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
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
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
        self.phi_hidden = args.phi_hidden
        
        self.task_emb_std = args.task_emb_std

        #### define various networks
        ## networks for attention
        self._build_attention(surrogate_decomposer)
        self._build_policy(surrogate_decomposer)

        ## networks for successor features
        self._build_SF(surrogate_decomposer)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.wo_action_layer_psi.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def pretrain_forward(self, inputs, cur_action, hidden_state, task):
        bsn = inputs.shape[0]
        attn_feature, enemy_feats = self._get_attn_feature(inputs, task)
        _, _, cur_compact_action = self.task2decomposer[task].decompose_action_info(cur_action)
        cur_compact_action = cur_compact_action.reshape(-1, cur_compact_action.shape[-1])
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in)
        tau_a_inputs = th.cat([h, cur_compact_action], dim=-1)

        r_shaping = self.r_shaping(h).squeeze(-1) # (bsn,)
        phi_hid, mu, logvar = self._phi_enc(tau_a_inputs) # (bsn, d_phi, phi_hid)
        
        ## decoder />
        wo_action_layer_in = th.cat([h, phi_hid.reshape(-1, self.phi_dim * self.phi_hidden)], dim=-1) # FIXME duplicated h
        wo_action_a = self.wo_action_layer_phi(wo_action_layer_in)
        
        enemy_feature = self.enemy_embed(enemy_feats) # (bsn, n_enemy, hid)
        
        if self.have_attack_action:
            phi_feature = phi_hid.reshape(bsn, -1).unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
            attack_action_input = th.cat([enemy_feature, phi_feature], dim=-1)
            attack_action_a = self.attack_action_layer_phi(attack_action_input).squeeze(-1) # (bsn, n_enemy)
            action_recon = th.cat([wo_action_a, attack_action_a], dim=-1)
        else:
            action_recon = wo_action_a
        ## </decoder
        
        phi_hid = phi_hid.view(-1, self.phi_dim, self.phi_hidden)
        phi = self.phi_proj(phi_hid).squeeze(-1) # (bsn, d_phi)

        return h, phi, mu, logvar, r_shaping, action_recon

    def forward(self, inputs, cur_action, hidden_state, task, task_weight): # psi forward
        bsn = inputs.shape[0]
        n_agents = self.task2n_agents[task]
        attn_feature, enemy_feats = self._get_attn_feature(inputs, task)
        _, _, cur_compact_action = self.task2decomposer[task].decompose_action_info(cur_action)
        cur_compact_action = cur_compact_action.reshape(-1, cur_compact_action.shape[-1])
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in) # (bsn, hid)
        task_emb = self._task_enc(task_weight.repeat(h.shape[0], 1)) # (bsn, hid)
        
        ## forward psi
        psi_hid = self.psi(th.cat([h, task_emb], dim=-1)).view(-1, self.phi_dim, self.phi_hidden) # (bsn, d_phi, phi_hid)
        wo_action_layer_in = psi_hid
        wo_action_psi = self.wo_action_layer_psi(wo_action_layer_in) # (bsn, d_phi, n_wo_action)

        enemy_feature = self.enemy_embed(enemy_feats) # (bsn, n_enemy, hid)

        if self.have_attack_action:
            attack_action_input = th.cat(
                [
                    enemy_feature.unsqueeze(1).repeat(1, self.phi_dim, 1, 1),
                    psi_hid.unsqueeze(2).repeat(1, 1, enemy_feats.size(1), 1),
                ],
                dim=-1,
            )  # (bsn, d_phi, n_enemy, hid+phi_hid)
            attack_action_psi = self.attack_action_layer_psi(attack_action_input).squeeze(-1) # (bsn, d_phi, n_enemy)
            psi = th.cat([wo_action_psi, attack_action_psi], dim=-1)
        else:
            psi = wo_action_psi
        # psi: (bsn, d_phi, n_act) -> (bs, n, d_phi, n_act)
        psi = psi.view(-1, n_agents, self.phi_dim, cur_action.shape[-1])
        psi_v = psi.mean(dim=-1, keepdim=True)
        psi_a = (psi - psi_v).detach() # TODO why need (no) detach ?
        
        select_action = cur_action.unsqueeze(2).repeat(1, 1, self.phi_dim, 1) # (bs, n, n_act) -> (bs, n, d_phi, n_act)
        select_action_adv = (psi_a * select_action).sum(-1) # (bs, n, d_phi)
        adv_mask = (select_action_adv > 0).float() # according policy, (bs, n, d_phi) \in {0,1}^~; # or softly weighting agents' value/contrib in each subtask ?
        
        ## forward phi with masking
        tau_a_inputs = th.cat([h, cur_compact_action], dim=-1)
        phi_hid, _, _ = self._phi_enc(tau_a_inputs) # (bsn, d_phi, phi_hid)
        phi = self.phi_proj(phi_hid).squeeze(-1) # (bsn, d_phi)
        
        phi_hid_input = phi_hid.detach().reshape(-1, n_agents, self.phi_dim, self.phi_hidden)
        phi_tilde_hid = self._group_self_attn(adv_mask, phi_hid_input, n_agents) # (bs, n, d_phi, phi_hid)
        phi_tilde = self.phi_proj(phi_tilde_hid.reshape(-1, self.phi_dim, self.phi_hidden)).squeeze(-1)
        # prophet, oracle; do not transmit back to phi enc
        
        r_shaping = self.r_shaping(h).squeeze(-1) # (bsn,)
        
        return h, psi, phi, phi_tilde, r_shaping

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
        if self.have_attack_action:
            attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
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
                obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.aligned_obs_nf_en,
                    surrogate_decomposer.aligned_obs_nf_al,
                )
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
                wrapped_obs_own_dim = (
                    obs_own_dim + self.args.id_length + n_actions_no_attack + 1
                )
                ## enemy_obs ought to add attack_action_infos
                obs_en_dim += 1
            case "gymma":
                obs_own_dim, obs_en_dim, obs_al_dim = surrogate_decomposer.own_obs_dim, surrogate_decomposer.obs_nf_en, surrogate_decomposer.obs_nf_al
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
                wrapped_obs_own_dim = obs_own_dim + self.args.id_length + n_actions_no_attack + 1 # see the decomposers
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
        self.phi_query = nn.Linear(self.phi_hidden, self.attn_embed_dim * self.args.head)
        self.phi_key = nn.Linear(self.phi_hidden, self.attn_embed_dim * self.args.head)
        self.phi_value = nn.Linear(self.phi_hidden, self.phi_hidden)

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

        self.wo_action_layer_psi = nn.Linear(self.phi_hidden, n_actions_no_attack)
        self.wo_action_layer_phi = FCNet(self.args.rnn_hidden_dim + self.phi_dim * self.phi_hidden, n_actions_no_attack)
        ## attack action networks
        self.attack_action_layer_psi = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.phi_hidden, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, 1)
        )
        self.attack_action_layer_phi = FCNet(self.args.rnn_hidden_dim + self.phi_dim * self.phi_hidden, 1)
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )

    # For transfer learning
    def _build_SF(self, surrogate_decomposer): 
        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack

        # include a shared local feature encoder-decoder, (local decoder is _build_policy); => \phi
        self.phi_enc = FCNet(self.hidden_dim + n_actions_no_attack + 1, 2 * self.phi_dim * self.phi_hidden) # traj_hid + no_attack_actions + 1(compact attack) -> mu, logstd (multi headed)
        self.phi_proj = nn.Linear(self.phi_hidden, 1)
        
        self.r_shaping = FCNet(self.hidden_dim, 1)
        self.task_enc = FCNet(self.phi_dim, self.hidden_dim) # z(w)
        self.psi = FCNet(self.hidden_dim * 2, self.phi_dim * self.phi_hidden) # h, z(w) -> psi_hid
        
    def _task_enc(self, inputs):
        eps = th.randn_like(inputs).clamp(min=-0.5, max=0.5)
        inputs += eps * self.task_emb_std
        emb = self.task_enc(inputs)
        return emb
        
    def _phi_enc(self, inputs):
        emb1 = self.phi_enc(inputs).view(-1, 2, self.phi_dim, self.phi_hidden) # (bsn, 2, d_phi, phi_hid)
        mu, logvar = emb1[:, 0], emb1[:, 1]

        eps = th.randn_like(mu)
        std = th.exp(.5 * logvar)
        reparam = mu + std * eps

        return reparam, mu, logvar # (bsn, d_phi, phi_hid)

    def _group_self_attn(self, adv_mask, phi_hid, n_agents):
        """ Apply subgroup-based self attention on phi.

        Args:
            adv_mask (Tensor): (bs, n, d_phi) \in {0,1}^*
            phi_hid (Tensor): (bs, n, d_phi, phi_hid)

        Returns:
            phi_tilde_hid: (bs, n, d_phi, phi_hid)
        """
        adv_mask = adv_mask.transpose(1, 2).reshape(-1, n_agents).unsqueeze(-1).repeat(1, 1, self.phi_hidden) # (bs*d_phi, n, phi_hid)
        phi_hid = phi_hid.transpose(1, 2).reshape(-1, n_agents, self.phi_hidden) # (bs*d_phi, n, phi_hid)
        phi_hid_adv = phi_hid * adv_mask
        phi_hid_dis = phi_hid * (1. - adv_mask)
        
        # Apply (single head) self attention on phi_hid_adv, (bs*d_phi, n, attn_emb|phi_hid)
        # NOTE or more expressive transformer may be needed !
        assert self.args.head == 1
        q = self.phi_query(phi_hid_adv)
        k = self.phi_key(phi_hid_adv)
        v = self.phi_value(phi_hid_adv)
        energy = th.bmm(q, k.transpose(1, 2)) / (self.attn_embed_dim ** (1/2)) # (bs*d_phi, n, n)
        score = F.softmax(energy, dim=-1)
        attn = th.bmm(score, v) # self-attended(phi_hid_adv) (bs*d_phi, n, phi_hid)
        
        out = attn * adv_mask + phi_hid_dis
        
        phi_tilde_hid = out.view(-1, self.phi_dim, n_agents, self.phi_hidden).transpose(1, 2) # (bs, n, d_phi, phi_hid)
        return phi_tilde_hid

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
            layers.append(nn.Softmax(dim=-1))
            
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.layers(x)
