import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed

class TrSFRNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, 
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args) -> None:
        super(TrSFRNNAgent, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in
                                       task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.hidden_dim = args.rnn_hidden_dim
        
        self.phi_dim = args.phi_dim

        #### define various networks
        ## networks for attention
        self._build_attention(surrogate_decomposer)
        self._build_policy(surrogate_decomposer)
        
        ## networks for successor features
        self._build_SF(surrogate_decomposer)
    
    def init_hidden(self):
        # make hidden states on the same device as model
        return self.wo_action_layer.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, cur_action, hidden_state, task):
        attn_feature, enemy_feats = self._get_attn_feature(inputs, task) # (bs * n_agents, entity_embed_dim * 3 * n_heads); last step actions
        cur_no_attack_action, cur_attack_action, cur_compact_action = self.task2decomposer[task].decompose_action_info(cur_action)
        # compact_action = no_attack + 1(bin_attack)
        # enemy_feats (bs*n_agents, n_enemy, x)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in) # (bs*n_agents, h)
        
        tau_a_inputs = th.cat([h, cur_compact_action], dim=-1) # (bsn, h + no_att + 1)
        # FIXME check all shape 
        phi = self._phi_enc(tau_a_inputs) # (bsn, d_phi)
        tau_phi_inputs = th.cat([h, phi], dim=-1) # (bsn, h+d_phi)
        recon_action = self._phi_dec(h, enemy_feats, phi) # (bsn, n_act)
        
        phi_hat = self.phi_enc2(tau_phi_inputs)
        group_idx = None # TODO input adv infos and calc this
        phi_tilde = self._group_self_attn(group_idx, phi.detach()) # TODO check if detach is needed
        psi_tilde = self.psi(tau_a_inputs)
        
        w = self.wGen(tau_phi_inputs) # (bsn, d_phi)
        r_hat = self._uvf(phi_tilde.detach(), w) # (bsn,)
        q_hat = self._uvf(psi_tilde, w) # (bsn,)
        
        return h, q_hat, r_hat, w, psi_tilde, phi_tilde, phi_hat, recon_action
    
    def pretrain_forward(self, inputs, cur_action, hidden_state, task):
        attn_feature, enemy_feats = self._get_attn_feature(inputs, task)
        cur_no_attack_action, cur_attack_action, cur_compact_action = self.task2decomposer[task].decompose_action_info(cur_action)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in)
        tau_a_inputs = th.cat([h, cur_compact_action], dim=-1)
        phi, mu, logvar = self._phi_enc(tau_a_inputs) # (bsn, d_phi)
        
        r_shaping = self.r_shaping(h).squeeze(-1) # (bsn,)
        
        action_recon = self._phi_dec(h, enemy_feats, phi) # （bsn, n_act)
        
        return h, phi, mu, logvar, r_shaping, action_recon
    
    def forward_enc2(self, inputs, hidden_state, task):
        pass

    def _get_attn_feature(self, inputs, task):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim+last_action_shape], inputs[:, obs_dim+last_action_shape:]
        
        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)    
        # own_obs: [bs*self.n_agents, own_obs_dim]
        # enemy_feats: list(bs * n_agents, obs_nf_en/obs_en_dim), len=n_enemies
        bs = int(own_obs.shape[0]/task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs 
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info (n_enemies, bs*n_agents, 1) into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        if self.have_attack_action:
            enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1).transpose(0, 1) 
        else:
            enemy_feats = th.stack(enemy_feats, dim=0).transpose(0, 1)
        # (bs*n_agents, n_enemies, obs_nf_en+1)
        ally_feats = th.stack(ally_feats, dim=0).transpose(0, 1)


        # compute k, q, v for (multi-head) attention
        own_feature = self.own_value(own_obs) #(bs * n_agents, entity_embed_dim * n_heads)
        # assert own_feature.size() == (bs * task_n_agents, self.entity_embed_dim * self.args.head), print("own feature size:", own_feature.size())
        
        query = self.query(own_obs)

        ally_keys = self.ally_key(ally_feats)  # (bs*n_agents, n_ally, attn_dim *n_heads)
        enemy_keys = self.enemy_key(enemy_feats)
        ally_values = self.ally_value(ally_feats)
        enemy_values = self.enemy_value(enemy_feats)

        if self.args.head == 1:
            ally_feature = self.attention(query, ally_keys, ally_values, self.attn_embed_dim)
            enemy_feature = self.attention(query, enemy_keys, enemy_values, self.attn_embed_dim)
        else:
            ally_feature = self.multi_head_attention(query, ally_keys, ally_values, self.attn_embed_dim)
            enemy_feature = self.multi_head_attention(query, enemy_keys, enemy_values, self.attn_embed_dim)
        
        attn_feature = th.cat([own_feature, ally_feature, enemy_feature], dim=-1)

        return attn_feature, enemy_feats

    def multi_head_attention(self, q, k, v, attn_dim):
        """
            q: [bs*n_agents, attn_dim*n_heads]
            k: [bs*n_agents,n_entity, attn_dim*n_heads]
            v: [bs*n_agents, n_entity, value_dim*n_heads]
        """
        bs = q.shape[0]
        q = q.unsqueeze(1).view(bs, 1, self.args.head, self.attn_embed_dim)
        k = k.view(bs, -1, self.args.head, self.attn_embed_dim)
        v = v.view(bs, -1, self.args.head, self.entity_embed_dim)

        q = q.transpose(1, 2).contiguous().view(bs*self.args.head, 1, self.attn_embed_dim)
        k = k.transpose(1, 2).contiguous().view(bs*self.args.head, -1, self.attn_embed_dim)
        v = v.transpose(1, 2).contiguous().view(bs*self.args.head, -1, self.entity_embed_dim)
        
        energy = th.bmm(q, k.transpose(1, 2)) / (attn_dim ** (1 / 2))
        assert energy.shape[0] == bs * self.args.head and energy.shape[1] == 1
        # shape[2] == n_entity
        score = F.softmax(energy, dim=-1)
        score = th.bmm(score, v).view(bs, self.args.head, 1, self.entity_embed_dim) # (bs*head, 1, entity_embed_dim) 
        out = out.transpose(1, 2).contiguous().view(bs, 1, self.entity_embed_dim * self.args.head).squeeze(1)
        return out
    
    def attention(self, q, k, v, attn_dim):
        """
            q: [bs*n_agents, attn_dim]
            k: [bs*n_agents,n_entity, attn_dim]
            v: [bs*n_agents, n_entity, value_dim]
        """
        assert self.args.head == 1
        energy = th.bmm(q.unsqueeze(1), k.transpose(1, 2))/(attn_dim ** (1 / 2))
        score = F.softmax(energy, dim=-1)
        out = th.bmm(score, v).squeeze(1)
        return out

    def _build_attention(self, surrogate_decomposer):
        ## get obs shape information
        match self.args.env:
            case "sc2":
                obs_own_dim = surrogate_decomposer.aligned_own_obs_dim
                obs_en_dim, obs_al_dim = surrogate_decomposer.aligned_obs_nf_en, surrogate_decomposer.aligned_obs_nf_al        
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
                wrapped_obs_own_dim = obs_own_dim + self.args.id_length + n_actions_no_attack + 1
                ## enemy_obs ought to add attack_action_infos
                obs_en_dim += 1
            case "gymma":
                obs_own_dim, obs_en_dim, obs_al_dim = surrogate_decomposer.own_obs_dim, surrogate_decomposer.obs_nf_en, surrogate_decomposer.obs_nf_al
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
                wrapped_obs_own_dim = obs_own_dim + self.args.id_length + n_actions_no_attack # see gymma_offline.yaml
                ## enemy_obs ought to add attack_action_infos
                obs_en_dim += surrogate_decomposer.n_actions_attack
            case _:
                raise NotImplementedError
            
        assert self.attn_embed_dim % self.args.head == 0
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim * self.args.head)
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim * self.args.head)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim * self.args.head)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim * self.args.head)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim * self.args.head)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim * self.args.head)
        
        # SF self attention
        self.phi_query = nn.Linear(self.phi_dim, self.attn_embed_dim * self.args.head)
        self.phi_key = nn.Linear(self.phi_dim, self.attn_embed_dim * self.args.head)
        self.phi_value = nn.Linear(self.phi_dim, self.phi_dim)

    def _build_policy(self, surrogate_decomposer):
        ## get obs shape information
        match self.args.env:
            case "sc2":
                obs_en_dim = surrogate_decomposer.aligned_obs_nf_en      
                obs_en_dim += 1
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
            case "gymma":
                obs_en_dim = surrogate_decomposer.obs_nf_en + surrogate_decomposer.n_actions_attack
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
            case _:
                raise NotImplementedError
        

        self.rnn = nn.GRUCell(self.entity_embed_dim * self.args.head * 3, self.args.rnn_hidden_dim)
        
        self.wo_action_layer = nn.Linear(self.args.rnn_hidden_dim + self.phi_dim, n_actions_no_attack)
        ## attack action networks
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )
        self.attack_action_layer = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.args.rnn_hidden_dim + self.phi_dim, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, 1)
        )
    
    # For transfer learning
    def off(self):
        # TODO build offline learning required modules and pseudo-functions
        pass
    
    def on(self):
        # TODO build rectifier, critic, optionally new actor (Q-actor for discrete; old dec actor, new actor for disc and cont act)
        pass
    
    def _build_SF(self, surrogate_decomposer): 
        # TODO 
        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        
        # include a shared local feature encoder-decoder, (local decoder is _build_policy); => \phi
        self.phi_enc1 = FCNet(self.hidden_dim + n_actions_no_attack, 2 * self.phi_dim) # traj_hid + no_attack_actions + 1(compact attack) -> mu, logstd
        self.phi_enc2 = FCNet(self.hidden_dim + self.phi_dim, self.phi_dim) # local inference: \phi -> \hat \phi
        
        self.r_shaping = FCNet(self.hidden_dim, 1)
        
        # \psi networks; local shared
        self.psi = FCNet(self.hidden_dim + n_actions_no_attack, self.phi_dim) # same as phi
        
        # QPLEX, or other mixing module with #!GAE => Advantage networks (learn additional V-net)
            # an attn-based critic ? for population-inv #inputs
        # sub-group advantage-based self-attention module 
        
        # local weightGen, phi (enc), pi (dec) should be #!inherited
        self.wGen = FCNet(self.hidden_dim + self.phi_dim, self.phi_dim) # local weight generator that infer global weight
        
    def _phi_enc(self, inputs):
        emb1 = self.phi_enc1(inputs) # (bsn, 2*hid_dim)
        mu, logvar = emb1[:, :self.phi_dim], emb1[:, self.phi_dim:]
        reparam = self._reparam(mu, logvar)
        return reparam, mu, logvar
        
    def _reparam(self, mu, logvar):
        eps = th.randn_like(mu)
        std = th.exp( .5 * logvar)
        return mu + std * eps
        
    def _phi_dec(self, no_attack_in, enemy_feats, phi):
        no_attack_in = th.cat([no_attack_in, phi], dim=-1) # (bsn, hidden+d_phi)
        no_attack_q = self.wo_action_layer(no_attack_in)
        
        if self.have_attack_action:
            enemy_features = self.enemy_embed(enemy_feats)
            attack_in = th.cat([enemy_features, 
                                no_attack_in.unsqueeze(1).repeat(1, enemy_feats.size(1), 1),
                                phi.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)], dim=-1) 
            # (bs*n_agent, n_enemy, 2*hidden+d_phi)
            attack_q = self.attack_action_layer(attack_in).squeeze(-1) # (bs*n_agent, n_enemy)
            q = th.cat([no_attack_q, attack_q], dim=-1)
        else:
            q = no_attack_q
            
        return q
    
    def _group_self_attn(self, group_idx, phi): # FIXME check dims
        sub_phi = phi[:, group_idx, :] # (bs, n_agents -> n_subgroup, d_phi)
        query = self.phi_query(sub_phi)
        key = self.phi_key(sub_phi)
        value = self.phi_value(sub_phi)
        
        if self.args.head == 1:
            attn_feature = self.attention(query, key, value, self.attn_embed_dim)
        else:
            attn_feature = self.multi_head_attention(query, key, value, self.attn_embed_dim)
            
        return attn_feature
    
    def _calc_group_idx(self, adv):
        pass
    
    def _uvf(self, feat, w):
        return th.sum(th.multiply(feat, w), dim=-1)

class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=1, hidden_dim=1024, use_leaky_relu=True, use_last_activ=False):
        super().__init__()
        if use_leaky_relu:
            self.activ = nn.LeakyReLU()
        else:
            self.activ = nn.ReLU()
            
        layers = [nn.Linear(in_dim, hidden_dim), self.activ]
        for l in range(hidden_layer - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activ])
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        if use_last_activ:
            layers.append(nn.Sigmoid())
            
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.layers(x)
    