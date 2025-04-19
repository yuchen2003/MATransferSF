import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MTDMAQQattnMixer(nn.Module):
    def __init__(self, surrogate_decomposer, main_args):
        super(MTDMAQQattnMixer, self).__init__()
        self.main_args = main_args
        self.embed_dim = main_args.mixing_embed_dim
        self.attn_embed_dim = main_args.attn_embed_dim
        self.entity_embed_dim = main_args.entity_embed_dim

        match self.main_args.env:
            case 'sc2' | 'sc2_v2':
                state_nf_al, state_nf_en, timestep_state_dim = \
                    surrogate_decomposer.aligned_state_nf_al, surrogate_decomposer.aligned_state_nf_en, surrogate_decomposer.timestep_number_state_dim
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
        
        if getattr(main_args, "hypernet_layers", 1) == 1:
            self.trfm_w = nn.Linear(entity_mixing_input_dim, 1)
            self.hyper_w_1 = nn.Linear(entity_mixing_input_dim, 1)
        elif getattr(main_args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.main_args.hypernet_embed
            self.trfm_w = nn.Sequential(
                nn.Linear(entity_mixing_input_dim, hypernet_embed),
                nn.LeakyReLU(),
                nn.Linear(hypernet_embed, 1),
            )
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(entity_mixing_input_dim, hypernet_embed),
                nn.LeakyReLU(),
                nn.Linear(hypernet_embed, 1),
            )
        elif getattr(main_args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.") 
        
        # V(s) instead of a bias for the last layers
        self.hyper_b_1 = nn.Sequential(nn.Linear(mixing_input_dim, self.embed_dim),
                               nn.LeakyReLU(),
                               nn.Linear(self.embed_dim, 1))
    
    def forward(self, agent_qs, max_qs, states, task_decomposer):
        # agent_qs: [batch_size, seq_len, n_agents, d_phi]
        # states: [batch_size, seq_len, state_dim]
        if len(agent_qs.shape) == 4:
            bs, seq_len, n_agents, phi_dim = agent_qs.size()
        else:
            bs, seq_len, n_agents = agent_qs.size()
            phi_dim = 1
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
        enemy_embed = self.enemy_encoder(enemy_states)

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

        entity_mixing_input = th.cat(
            [out.unsqueeze(2).repeat(1, 1, n_agents, 1), 
             ally_embed], 
            dim=-1
        ).reshape(bs*seq_len, n_agents, -1)  # (bs, seq_len, n_agents, x)
        mixing_input = out.reshape(bs*seq_len, -1)

        agent_qs = agent_qs.reshape(bs * seq_len, n_agents, phi_dim)
        max_qs = max_qs.reshape(bs * seq_len, n_agents, phi_dim)
        w = th.abs(self.trfm_w(entity_mixing_input)) + 1e-10 # (bsT, n, 1)
        b = self.hyper_b_1(mixing_input).unsqueeze(-1) / n_agents # (bsT, 1, 1)
        agent_qs = w * agent_qs + b
        max_qs = w * max_qs + b
        adv_q = agent_qs - max_qs # [bsT, n, d_phi]
        
        w1 = th.abs(self.hyper_w_1(entity_mixing_input)) + 1e-10 # [bsT, n, 1]
        # adv_tot = th.sum((w1 - 1.) * adv_q, dim=1) # (bs*seq_len)
        adv_tot = th.bmm(adv_q.transpose(1, 2), (w1 - 1.)) # [bsT, d_phi, 1]
        
        q_sum = agent_qs.sum(1).unsqueeze(-1)
        
        q_tot = (adv_tot + q_sum).reshape(bs, seq_len, -1)
        
        return q_tot