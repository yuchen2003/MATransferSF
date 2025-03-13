import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed, onehot_embed
from utils.calc import count_total_parameters

class AttnFeatureExtractor(nn.Module):
    '''
    phi|feature extraction module, based on attention mechanism
    '''
    def __init__(self, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args): # TODO diff obs & obs_action inputs
        super().__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)
        self.state_last_action, self.state_timestep_number = surrogate_decomposer.state_last_action, surrogate_decomposer.state_timestep_number

        ## get obs shape information
        match self.args.env:
            case "sc2":
                obs_own_dim = surrogate_decomposer.aligned_own_obs_dim
                obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.aligned_obs_nf_en,
                    surrogate_decomposer.aligned_obs_nf_al,
                )
                state_en_dim, state_al_dim = (
                    surrogate_decomposer.state_nf_en,
                    surrogate_decomposer.state_nf_al,
                )
                wrapped_obs_own_dim = obs_own_dim + self.args.id_length
                ## enemy_obs ought to add attack_action_infos
                obs_en_dim += 1
            case "gymma":
                obs_own_dim, obs_en_dim, obs_al_dim = surrogate_decomposer.own_obs_dim, surrogate_decomposer.obs_nf_en, surrogate_decomposer.obs_nf_al
                state_en_dim, state_al_dim = surrogate_decomposer.state_nf_en, surrogate_decomposer.state_nf_al
                wrapped_obs_own_dim = obs_own_dim + self.args.id_length
                ## enemy_obs ought to add attack_action_infos
                obs_en_dim += surrogate_decomposer.n_actions_attack
            case _:
                raise NotImplementedError

        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        if self.args.obs_last_action:
            wrapped_obs_own_dim += n_actions_no_attack + 1

        if self.state_last_action:
            state_al_dim += self.n_actions_no_attack + 1

        assert self.attn_embed_dim % self.args.head == 0
        # obs self-attention modules
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim * self.args.head)
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim * self.args.head)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim * self.args.head)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim * self.args.head)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim * self.args.head)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim * self.args.head)
        # inverse dynamic predictor
        self.na_action_layer = nn.Sequential(
            nn.Linear(self.entity_embed_dim * self.args.head * 6, self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.rnn_hidden_dim, n_actions_no_attack)
        )
        # self.na_action_layer = nn.Linear(self.entity_embed_dim * self.args.head * 6, n_actions_no_attack)
        self.attack_action_layer = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim + self.entity_embed_dim * self.args.head * 6, self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.rnn_hidden_dim, 1)
        )
        # self.attack_action_layer = nn.Linear(self.args.rnn_hidden_dim + self.entity_embed_dim * self.args.head * 6, 1)
        self.enemy_embed = nn.Sequential( # NOTE it has a different semantic with that in ValueModule
            nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )

    def _multi_head_attention(self, q, k, v, attn_dim):
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

    def _attention(self, q, k, v, attn_dim):
        """
            q: [bs*n_agents, (query_len), attn_dim]
            k: [bs*n_agents, n_entity, attn_dim]
            v: [bs*n_agents, n_entity, value_dim]
        """
        assert self.args.head == 1
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        energy = th.bmm(q, k.transpose(1, 2))/(attn_dim ** (1 / 2))
        score = F.softmax(energy, dim=-1)
        out = th.bmm(score, v).squeeze(1)
        return out

    def _get_attn_feature(self, inputs, task):
        # inputs == obs: (bsn, d)
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
            ally_feature = self._attention(query, ally_keys, ally_values, self.attn_embed_dim)
            enemy_feature = self._attention(query, enemy_keys, enemy_values, self.attn_embed_dim)
        else:
            ally_feature = self._multi_head_attention(query, ally_keys, ally_values, self.attn_embed_dim)
            enemy_feature = self._multi_head_attention(query, enemy_keys, enemy_values, self.attn_embed_dim)

        attn_feature = th.cat([own_feature, ally_feature, enemy_feature], dim=-1)

        return attn_feature, enemy_feats

    # def _state_rep(self, states, obs, task):
    #     '''
    #     Represent obs with state: o^i_t = O^i(s_t) as common in Dec-POMDP
    #     states: (bsT, d)
    #     obs: (bsT, n, d)
    #     '''
    #     decomposer = self.task2decomposer[task]
    #     bst, n_agents, _ = obs.size()
    #     n_enemies = decomposer.n_enemies
    #     n_entities = n_agents + n_enemies

    #     agent_states, enemy_states, last_action_states, _ = decomposer.decompose_state(states)
    #     agent_states = th.stack(agent_states, dim=0) # [n_agents, bsT, state_nf_al]
    #     enemy_states = th.stack(enemy_states, dim=0).permute(1, 0, 2) # [n_enemies, bsT, state_nf_en]

    #     if self.state_last_action:
    #         last_action_states = th.stack(last_action_states, dim=0) # (n_agents, bsT, n_actions)
    #         _, _, last_compact_action_states = decomposer.decompose_action_info(last_action_states)
    #         agent_states = th.cat([agent_states, last_compact_action_states], dim=-1)

    #     agent_embed = self.agent_enc(agent_states).permute(1, 0, 2)
    #     enemy_embed = self.enemy_enc(enemy_states)
    #     entity_embed = th.cat([agent_embed, enemy_embed], dim=1) # [bsT, n_entity, entity_embed_dim]

    #     # encode with self-attention
    #     entity_query = self.state_query(entity_embed)
    #     entity_key = self.state_key(entity_embed)
    #     entity_value = self.state_value(entity_embed)

    #     if self.args.head == 1:
    #         state_attn_feature = self._attention(entity_query, entity_key, entity_value, self.attn_embed_dim)
    #     else:
    #         state_attn_feature = self._multi_head_attention(entity_query, entity_key, entity_value, self.attn_embed_dim)
    #     # mean pooling over entity
    #     state_attn_feature = state_attn_feature.mean(dim=1) # (bsT, entity_embed_dim)
    #     state_kv_in = th.cat([state_attn_feature.unsqueeze(1).repeat(1, n_agents, 1),
    #                              agent_embed], dim=-1) # (bsT, n_agents, 2d)
    #     # state_query = self.ca_state_query(state_query_in) # (bsT, n_agents, d_attn)
    #     state_key = self.ca_state_key(state_kv_in)
    #     state_value = self.ca_state_value(state_kv_in)

    #     obs = obs.reshape(bst * n_agents, -1)
    #     obs_attn_feature, _ = self._get_attn_feature(obs, task) # (bsTn, 3d)
    #     obs_attn_feature = obs_attn_feature.reshape(bst, n_agents, -1)
    #     obs_query = self.ca_obs_query(obs_attn_feature)
    #     # obs_key = self.ca_obs_key(obs_attn_feature)
    #     # obs_value = self.ca_obs_value(obs_attn_feature) # (bsT, n_agents, d)

    #     if self.args.head == 1:
    #         state_rep = self._attention(obs_query, state_key, state_value, self.attn_embed_dim)
    #     else:
    #         state_rep = self._multi_head_attention(obs_query, state_key, state_value, self.attn_embed_dim)

    #     return state_rep, enemy_states

    def invdyn_forward(self, states, obs, next_states, next_obs, task):
        # /> with state
        # state_rep, enemy_states = self._state_rep(states, obs, task)
        # next_states_rep, _ = self._state_rep(next_states, next_obs, task)
        # action_pred_in = th.cat([state_rep, next_states_rep], dim=-1) # (bsT, n, 2d)
        # </ with state
        # /> wo state
        bst, n_agents, _ = obs.size()
        obs = obs.reshape(bst * n_agents, -1)
        next_obs = next_obs.reshape(bst * n_agents, -1)
        obs_attn_feature, enemy_states = self._get_attn_feature(obs, task) # (bsTn, 3d)
        next_obs_attn_feature, _ = self._get_attn_feature(next_obs, task)
        action_pred_in = th.cat(
            [
                obs_attn_feature.reshape(bst, n_agents, -1),
                next_obs_attn_feature.reshape(bst, n_agents, -1),
            ],
            dim=-1,
        )  # (bsT, n, 6d)
        # </ wo state

        na_action_pred = self.na_action_layer(action_pred_in)
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_states) # (bsT, n_enemy, hid)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    action_pred_in.unsqueeze(1).repeat(1, enemy_states.size(1), 1)
                ],
                dim=-1
            ) # (bs, n_enemy, hid+2d)
            attack_action_pred = self.attack_action_layer(attack_action_input).squeeze(-1) # TODO need debug
            action_pred = th.cat([na_action_pred, attack_action_pred], dim=-1)
        else:
            action_pred = na_action_pred

        action_pred = F.softmax(action_pred, dim=-1)

        return action_pred

    def forward(self, inputs, task):
        attn_feature, enemy_feats = self._get_attn_feature(inputs, task)
        return attn_feature, enemy_feats

class ValueModule(nn.Module):
    '''
    compute psi|Q value based on 
    '''
    def __init__(self, surrogate_decomposer, args):
        super().__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.phi_dim = args.rnn_hidden_dim
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)
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
        self.n_actions_no_attack = n_actions_no_attack
        self.id_action_no_attack = th.eye(self.n_actions_no_attack, dtype=th.float32, device=args.device)
        self.rnn = nn.GRUCell(self.entity_embed_dim * self.args.head * 3, self.args.rnn_hidden_dim)

        ## attack action networks
        # self.psi = FCNet(self.hidden_dim + self.n_actions_no_attack, self.phi_dim, hidden_layer=2)
        self.no_attack_psi = nn.Sequential(
            nn.Linear(self.hidden_dim + self.n_actions_no_attack, 2 * self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * self.args.rnn_hidden_dim, self.phi_dim),
        )
        self.attack_action_layer_psi = nn.Sequential(
            nn.Linear(2 * self.args.rnn_hidden_dim, 2 * self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * self.args.rnn_hidden_dim, self.phi_dim)
        )
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )
    
    def forward(self, attn_feature, enemy_feats, hidden_state):
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in) # (bsn, hid)
        
        psi_input = h.unsqueeze(1).repeat(1, self.n_actions_no_attack, 1) # (bsn, n_no_attack, hid)
        id_action_no_attack = self.id_action_no_attack.repeat(h.size(0), 1, 1)
        psi_input = th.cat([psi_input, id_action_no_attack], dim=-1)
        wo_action_psi = self.no_attack_psi(psi_input) # (bsn, n_no_attack, d_phi)
        loss = 0
        loss_info = {}
        
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_feats) # (bsn, n_enemy, hid)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    h.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
                ],
                dim=-1,
            )  # (bsn, n_enemy, 2*hid)
            attack_action_psi = self.attack_action_layer_psi(attack_action_input) # (bsn, n_enemy, d_phi)
            psi = th.cat([wo_action_psi, attack_action_psi], dim=1)
        else:
            psi = wo_action_psi

        return h, psi, loss, loss_info

class TaskHead(nn.Module):
    '''
    compute w via | multitask regression | -task weight explainer-
    '''
    def __init__(self, n_train_task, args, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_train_task, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, args.rnn_hidden_dim),
            # nn.Tanh(),
            nn.Softmax(),
        )
        
    def forward(self, task_id: th.Tensor):
        eps = (th.randn_like(task_id) * 0.1).clamp(-0.1, 0.1)
        task_id = (task_id + eps).clamp(0, 1)
        # return 0.5 * (self.net(x) + 1)
        return self.net(task_id)


class TrSFRNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, n_train_task,
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

        # define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.hidden_dim = args.rnn_hidden_dim

        self.phi_dim = args.rnn_hidden_dim
        self.fixed_phi = th.randn(3 * self.hidden_dim, self.phi_dim, device=args.device).clamp(-1, 1)

        # define various networks
        self.attn_enc = AttnFeatureExtractor(task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args)
        self.value = ValueModule(surrogate_decomposer, args)
        self.task_explainer = TaskHead(n_train_task, args)
        
        count_total_parameters(self)

    def init_hidden(self):
        # make hidden states on the same device as model
        return th.zeros(1, self.args.rnn_hidden_dim, device=self.args.device)

    def pretrain_forward(self, states, obs, next_states, next_obs, task):
        action_pred = self.attn_enc.invdyn_forward(states, obs, next_states, next_obs, task)
        return action_pred
    
    def forward(self, inputs, hidden_state, task): # psi forward
        n_agents = self.task2n_agents[task]
        bs = inputs.shape[0] // n_agents
        attn_feature, enemy_feats = self.attn_enc(inputs, task)
        
        phi = th.matmul(attn_feature, self.fixed_phi).view(bs, n_agents, -1).clamp(-0.1, 0.1) # TODO param proj; or modify phi_dim
        h, psi, loss, loss_info = self.value(attn_feature, enemy_feats, hidden_state)
        
        # psi: (bsn, n_act, d_phi) -> (bsn, d_phi, n_act) -> (bs, n, d_phi, n_act)
        psi = psi.transpose(1, 2).view(bs, n_agents, self.phi_dim, -1)

        return h, phi, psi, loss, loss_info
        
    def explain_task(self, task_id):
        return self.task_explainer(task_id)
        
    def set_train_mode(self, mode):
        """ Set the training or evaluation mode of the agent. mode=
        
            pretrain: update only feature extractor modules;
            
            offline: update all modules except task weight head;
            
            online: update policy head;
            
            adapt: update only task weight head;
            
        Args:
            mode (str): agent running mode.
            
        Return: None
        """        
        # CHECK 试一下不同的参数冻结方法
        mode = mode.lower()
        assert mode in ['pretrain', 'offline', 'online', 'adapt']
        modules = [self.attn_enc, self.value, self.task_explainer]
        match mode:
            case 'pretrain':
                grad_set = [True, False, False]
            case 'offline': 
                grad_set = [True, True, True]
            case 'online':
                grad_set = [False, True, True]
            case 'adapt':
                grad_set = [False, False, True]
                
        for g_set, module in zip(grad_set, modules):
            for param in module.parameters():
                param.require_grad = g_set
                
        print(f"Model in [{mode}] training mode.")

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
