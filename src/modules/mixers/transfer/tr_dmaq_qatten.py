import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, layer, last_layer_bias=True):
        super().__init__()
        self.net = []
        for _ in range(layer - 1):
            self.net.append(nn.Linear(in_dim, hidden_dim))
            self.net.append(nn.ReLU())
            in_dim = hidden_dim
        self.net.append(nn.Linear(hidden_dim, out_dim, bias=last_layer_bias))
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        return self.net(x)

class DMAQ_SI_Weight(nn.Module):
    def __init__(self, args, n_actions_no_attack):
        super(DMAQ_SI_Weight, self).__init__()
        self.args = args
        self.n_actions_no_attack = n_actions_no_attack
        mixing_head = args.mixing_head
        hypernet_embed_dim = args.hypernet_embed
        entity_embed_dim = args.entity_embed_dim
        n_layer = args.hypernet_layers
        
        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        for _ in range(mixing_head):
            self.key_extractors.append(Extractor(entity_embed_dim, 1, hypernet_embed_dim, n_layer))
            self.agents_extractors.append(Extractor(2 * entity_embed_dim, 1, hypernet_embed_dim, n_layer))
            self.action_extractors.append(Extractor(2 * entity_embed_dim + n_actions_no_attack + 1, 1, hypernet_embed_dim, n_layer))
        
    def forward(self, states, entity_states, actions, task_decomposer):
        # states: [bs, T, d], entity_states: [bs, T, n, d], actions: [bs, T, n, n_act]
        bs, seq_len, n_agents, _ = entity_states.size()
        states = states.reshape(bs * seq_len, -1)
        entity_states = entity_states.reshape(bs * seq_len, n_agents, -1)
        actions = actions.reshape(bs * seq_len, n_agents, -1)
        _, _, compact_action_info = task_decomposer.decompose_action_info(actions)
        state_actions = th.cat([entity_states, compact_action_info], dim=-1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors] # v: (bsT, 1)
        all_head_agents = [k_ext(entity_states).squeeze(-1) for k_ext in self.agents_extractors] # phi: (bsT, n)
        all_head_actions = [sel_ext(state_actions).squeeze(-1) for sel_ext in self.action_extractors] # lambda: (bsT, n)
        
        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_actions):
            x_key = th.abs(curr_head_key).repeat(1, n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action # (bsT, n)
            head_attend_weights.append(weights)
            
        head_attend = th.stack(head_attend_weights, dim=1) # (bsT, K, n)
        # head_attend = head_attend.view(-1, self.num_kernel, n_agents)
        head_attend = th.sum(head_attend, dim=1) # (bsT, n)

        return head_attend

class Qatten_Weight(nn.Module):
    def __init__(self, args):
        super(Qatten_Weight, self).__init__()
        self.mixing_head = args.mixing_head
        hypernet_embed_dim = args.hypernet_embed
        self.attn_dim = hypernet_embed_dim
        entity_embed_dim = args.entity_embed_dim
        n_layer = args.hypernet_layers
        self.attend_reg_coef = args.attend_reg_coef
        self.mask_dead = args.mask_dead
        self.weighted_head = args.weighted_head
        
        self.selector_extractors = nn.ModuleList()
        self.key_extractors = nn.ModuleList()
        for _ in range(self.mixing_head):
            self.selector_extractors.append(Extractor(entity_embed_dim, hypernet_embed_dim, hypernet_embed_dim, n_layer, last_layer_bias=False))
            self.key_extractors.append(nn.Linear(entity_embed_dim, hypernet_embed_dim, bias=False))
        if self.weighted_head:
            self.hyper_w_head = Extractor(entity_embed_dim, self.mixing_head, hypernet_embed_dim, 2)
            
        self.V = Extractor(entity_embed_dim, 1, hypernet_embed_dim, 2)
        
    def forward(self, agent_qs, states, ally_states, entity_states, actions):
        # agent_qs: [bs, T, n, 1|d_phi], states: [bs, T, d], entity_states: [bs, T, n, d], actions: [bs, T, n, n_act]
        bst, n_agents = agent_qs.size()[:2]
        states = states.reshape(bst, -1)
        ally_states = ally_states.reshape(bst, n_agents, -1)
        
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors] # (bsT, d)
        all_head_keys = [k_ext(ally_states) for k_ext in self.key_extractors] # (bsT, n, d)
        
        head_attend_logits, head_attend_weights = [], []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # (bsT, 1, d) * (bsT, d, n) -> (bsT, 1, n) -> (bsT, n)
            attend_logits = th.bmm(
                curr_head_selector.unsqueeze(1),
                curr_head_keys.transpose(1, 2),
            ).squeeze(1)
            scaled_attend_logits = attend_logits / (self.attn_dim ** .5)
            
            if self.mask_dead:
                raise NotImplementedError
            attend_weights = F.softmax(scaled_attend_logits, dim=-1)
            
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)
            
        head_attend = th.stack(head_attend_weights, dim=1) # (bsT, K, n)
        
        v = self.V(states) # (bsT, 1)
        if self.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bsT, K)
            w_head = w_head.unsqueeze(-1).repeat(1, 1, n_agents)  # w_head: (bsT, K, n)
            head_attend *= w_head
            
        head_attend = head_attend.sum(1) # (bsT, n)
        
        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies

class MTDMAQQattnMixer(nn.Module):
    def __init__(self, surrogate_decomposer, main_args):
        super(MTDMAQQattnMixer, self).__init__()
        self.main_args = main_args
        self.embed_dim = main_args.mixing_embed_dim
        self.attn_embed_dim = main_args.attn_embed_dim
        self.entity_embed_dim = main_args.entity_embed_dim
        self.gamma = self.main_args.gamma

        match self.main_args.env:
            case "sc2":
                state_nf_al, state_nf_en, timestep_state_dim = (
                    surrogate_decomposer.aligned_state_nf_al,
                    surrogate_decomposer.aligned_state_nf_en,
                    surrogate_decomposer.timestep_number_state_dim,
                )
            case 'gymma':
                state_nf_al, state_nf_en, timestep_state_dim = \
                    surrogate_decomposer.state_nf_al, surrogate_decomposer.state_nf_en, surrogate_decomposer.timestep_number_state_dim
        # timestep_state_dim = 0/1 denote whether encode the "t" of s
        # get detailed state shape information
        self.state_last_action, self.state_timestep_number = surrogate_decomposer.state_last_action, surrogate_decomposer.state_timestep_number

        # get action dimension information
        self.n_actions_no_attack = surrogate_decomposer.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            state_nf_al += self.n_actions_no_attack + 1
        self.ally_encoder = nn.Linear(state_nf_al, self.entity_embed_dim)
        self.enemy_encoder = nn.Linear(state_nf_en, self.entity_embed_dim)

        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

        mixing_input_dim = self.entity_embed_dim
        entity_mixing_input_dim = self.entity_embed_dim + self.entity_embed_dim
        if self.state_timestep_number:
            mixing_input_dim += timestep_state_dim
            entity_mixing_input_dim += timestep_state_dim

        self.attn_weight = Qatten_Weight(main_args)
        self.si_weight = DMAQ_SI_Weight(main_args, self.n_actions_no_attack)
        self.q_residual = Extractor(main_args.entity_embed_dim, 1, main_args.entity_embed_dim, 2)

    def calc_v(self, agent_qs):
        v_tot = th.sum(agent_qs, dim=-1) # over agent
        return v_tot

    def calc_adv(self, agent_qs, states, entity_states, actions, max_q_i, task_decomposer):
        adv_q = (agent_qs - max_q_i).detach()
        adv_w_final = self.si_weight(states, entity_states, actions, task_decomposer)
        if len(agent_qs.shape) == 3:
            adv_w_final = adv_w_final.unsqueeze(-1)
        adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1) # over agent
        return adv_tot

    def forward(self, agent_qs, states, actions, max_q_i, task_decomposer, is_phi=False):
        # agent_qs: [batch_size, seq_len, n_agents, (d_phi)]
        # states, actions: [batch_size, seq_len, dim]
        # max_q_i: [bs, seq_len, n_agents]
        bs, seq_len, n_agents = agent_qs.size()[:3]
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=2)  # [bs, seq_len, n_agents, state_nf_al]
        enemy_states = th.stack(enemy_states, dim=2)  # [bs, seq_len, n_enemies, state_nf_en]

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=2) # (bs, seq_len, n_agents, n_actions)
            _, _, last_compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, last_compact_action_states], dim=-1)

        # do inference and get entity_embed
        ally_embed = self.ally_encoder(ally_states) # [bs, seq_len, n_agents, entity_embed_dim]
        enemy_embed = self.enemy_encoder(enemy_states) # [bs, seq_len, n_enemies, entity_embed_dim]

        # we ought to do self-attention
        entity_embed = th.cat([ally_embed, enemy_embed], dim=2) # [bs, seq_len, n_entity, entity_embed_dim]

        # do attention
        proj_query = self.query(entity_embed).reshape(bs * seq_len, n_entities, self.attn_embed_dim)
        proj_key = self.key(entity_embed).reshape(bs * seq_len, n_entities, self.attn_embed_dim)
        energy = th.bmm(proj_query, proj_key.transpose(1, 2)) / (self.attn_embed_dim ** (1 / 2))
        score = F.softmax(energy, dim=-1) # (bs*seq_len, n_entities, n_entities)
        proj_value = entity_embed.reshape(bs * seq_len, n_entities, self.entity_embed_dim)
        out = th.bmm(score, proj_value) # (bs * seq_len, n_entities, entity_embed_dim)
        # mean pooling over entity
        out = out.mean(dim=1).reshape(bs, seq_len, self.entity_embed_dim)

        # concat timestep information
        if self.state_timestep_number:
            raise Exception(f"Not Implemented")
        # else:
        #     pass

        entity_mixing_input = th.cat(
            [
                out.unsqueeze(2).repeat(1, 1, n_agents, 1),
                ally_embed,
            ],
            dim=-1,
        )  # (bs, T, n, d)
        mixing_input = out # (bs, T, d)
        
        if is_phi:
            phi_dim = agent_qs.size(-1)
            agent_qs = agent_qs.reshape(bs * seq_len, n_agents, phi_dim)
            w_final, v, regs, ents = self.attn_weight(agent_qs, mixing_input, ally_embed, entity_mixing_input, actions) # w: [bsT, n], v: [bsT, 1]
            w_final = w_final + 1e-10
            w_final = w_final.unsqueeze(-1).repeat(1, 1, phi_dim)
            v = v.view(-1, 1, 1)
            v /= n_agents * phi_dim
            agent_qs = w_final * agent_qs + v
            max_q_i = th.zeros_like(agent_qs) # then adv_tot == linear transform of phi's, instead of affine
            adv_tot = self.calc_adv(agent_qs, mixing_input, entity_mixing_input, actions, max_q_i, task_decomposer)
            adv_tot = adv_tot.view(bs, seq_len, -1)
            return adv_tot, mixing_input
        else:
            agent_qs = agent_qs.view(bs * seq_len, n_agents)
            w_final, v, regs, ents = self.attn_weight(agent_qs, mixing_input, ally_embed, entity_mixing_input, actions) # w: [bsT, n], v: [bsT, 1]
            w_final = w_final + 1e-10
            v = v.view(-1, 1)
            v /= n_agents
            agent_qs = w_final * agent_qs + v
            max_q_i = max_q_i.view(bs * seq_len, n_agents)
            max_q_i = w_final * max_q_i + v  
            adv_tot = self.calc_adv(agent_qs, mixing_input, entity_mixing_input, actions, max_q_i, task_decomposer)
            q_sum = self.calc_v(agent_qs)
            q_tot = adv_tot + q_sum
            q_tot = q_tot.view(bs, seq_len, 1)
            return q_tot, regs, ents
    