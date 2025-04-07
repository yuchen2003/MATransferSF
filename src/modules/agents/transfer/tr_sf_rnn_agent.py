import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed, onehot_embed
from utils.calc import count_total_parameters

class TrSFRNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, n_train_task, train_mode,
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

        self.phi_dim = args.entity_embed_dim * args.head * 2 # FIXME 优化一下维度
        # self.fixed_phi = th.randn(3 * self.hidden_dim, self.phi_dim, device=args.device).clamp(-1, 1) # imitate fronzen transformation as in Does Zeroshot RL...
        self.mode = None

        # define various networks
        self.state_decomposer = PIStateDecomposer(task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args)
        self.obs_decomposer = PIObsDecomposer(task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args)
        self.task_explainer = TaskHead(task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args, self.phi_dim)
        self.phi_gen = AttnFeatureExtractor(self.task_explainer, task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args, is_for_pretrain=True)
        self.attn_enc = AttnFeatureExtractor(self.task_explainer, task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args)
        self.value = ValueModule(surrogate_decomposer, args, self.phi_dim)
        
        self.set_train_mode(train_mode)
        
        for module in [self.task_explainer, self.phi_gen, self.attn_enc, self.value]:
            count_total_parameters(module)
        print("Agent: ")
        count_total_parameters(self, is_concrete=True)

    def init_hidden(self):
        # make hidden states on the same device as model
        return th.zeros(1, self.args.rnn_hidden_dim, device=self.args.device)

    def pretrain_forward(self, state, obs, task):
        r_pred, action_pred = self.phi_gen.pretrain_forward(state, obs, task)
        return r_pred, action_pred
    
    def forward(self, obs, next_obs, inputs, hidden_state, task, test_mode): # psi forward
        n_agents = self.task2n_agents[task]
        bs = inputs.shape[0] // n_agents

        if test_mode:
            phi = None
        else:
            phi_out = self.phi_gen.feature_forward(obs, task) # NOTE no grad flows back to phi
            phi_cur = phi_out[:, :-1]
            phi_next = phi_out[:, 1:] # CHECK detach ?
            phi = th.cat([phi_cur, phi_next], dim=-1)
            phi = phi.view(bs, n_agents, -1)
        
        attn_feature, enemy_feats = self.attn_enc.attn_forward(inputs, task) 
        if self.mode in ['online', 'adapt']:
            attn_feature = attn_feature.detach()
        h, psi = self.value.forward(attn_feature, enemy_feats, hidden_state)
        
        # psi: (bsn, n_act, d_phi) -> (bs, n, n_act, d_phi)
        psi = psi.view(bs, n_agents, -1, self.phi_dim)

        return h, phi, psi
        
    def explain_task(self, task_id): # FIXME modify controller
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
        mode = mode.lower()
        assert mode in ['pretrain', 'offline', 'online', 'adapt']
        print(f"Model in [{mode}] training mode.")
        self.mode = mode
        
    def save(self, path):
        assert self.mode in ['pretrain', 'offline', 'online', 'adapt']
        match self.mode:
            case 'pretrain':
                th.save(self.phi_gen.state_dict(), "{}/phi_gen.th".format(path))
            case 'offline' | 'online' | 'adapt':
                th.save(self.state_dict(), "{}/agent.th".format(path))
        
    def load(self, path):
        assert self.mode in ['pretrain', 'offline', 'online', 'adapt'] # NOTE only consider continuing pretraining
        match self.mode:
            case 'pretrain' | 'offline':
                self.phi_gen.load_state_dict(th.load("{}/phi_gen.th".format(path), map_location=lambda storage, loc: storage))
            case _: # can be `eval` mode 
                self.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

class PIObsDecomposer(nn.Module):
    ''' Popolation invariant decomposer: ``input(scalable w. n agents) -> output(fixed embeddings)``; Can decompose observations;
    '''
    def __init__(self, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args, concat_obs_act):
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
            case "sc2" | "sc2_v2":
                obs_own_dim, obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.aligned_own_obs_dim,
                    surrogate_decomposer.aligned_obs_nf_en,
                    surrogate_decomposer.aligned_obs_nf_al,
                )
                ## enemy_obs ought to add attack_action_infos
                if not concat_obs_act and self.args.obs_last_action:
                    obs_en_dim += 1
            case "gymma":
                obs_own_dim, obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.own_obs_dim,
                    surrogate_decomposer.obs_nf_en,
                    surrogate_decomposer.obs_nf_al,
                )
                ## enemy_obs ought to add attack_action_infos
                if not concat_obs_act and self.args.obs_last_action:
                    obs_en_dim += surrogate_decomposer.n_actions_attack
            case _:
                raise NotImplementedError

        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        wrapped_obs_own_dim = obs_own_dim + self.args.id_length

        if not concat_obs_act and self.args.obs_last_action:
            wrapped_obs_own_dim += n_actions_no_attack + 1

        assert self.attn_embed_dim % self.args.head == 0
        # obs self-attention modules
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim * self.args.head)
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim * self.args.head)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim * self.args.head)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim * self.args.head)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim * self.args.head)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim * self.args.head)
        
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
        out = th.bmm(score, v).view(bs, self.args.head, 1, self.entity_embed_dim) # (bs*head, 1, entity_embed_dim) 
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

    def get_obs_feature(self, obs, task):
        # obs: (bs|bsn|bsTn, *)
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        # obs_inputs, last_action_inputs, agent_id_inputs = th.split(obs, [obs_dim, last_action_shape, obs.size(-1) - obs_dim - last_action_shape], dim=-1)
        obs_inputs, last_action_inputs, agent_id_inputs = obs[:, :obs_dim], \
            obs[:, obs_dim:obs_dim+last_action_shape], obs[:, obs_dim+last_action_shape:]
        
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
        if not self.is_for_pretrain and self.have_attack_action: 
            attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
            enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1).transpose(0, 1) 
        else:
            enemy_feats = th.stack(enemy_feats, dim=0).transpose(0, 1)
        # (bs*n_agents, n_enemies, obs_nf_en+1)
        ally_feats = th.stack(ally_feats, dim=0).transpose(0, 1)

        # compute k, q, v for (multi-head) attention
        own_feature = self.own_value(own_obs) #(bs * n_agents, entity_embed_dim * n_heads)
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
    
class PIStateDecomposer(nn.Module):
    ''' Popolation invariant decomposer: ``input(scalable w. n agents) -> output(fixed embeddings)``; Can decompose states;
    '''
    def __init__(self, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args):
        super().__init__()
        self.args = args
        self.mixing_embed_dim = args.mixing_embed_dim
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

        match self.main_args.env:
            case 'sc2' | "sc2_v2":
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

        # mixing_input_dim = self.entity_embed_dim
        # entity_mixing_input_dim = self.entity_embed_dim + self.entity_embed_dim
        # if self.state_timestep_number:
        #     mixing_input_dim += timestep_state_dim
        #     entity_mixing_input_dim += timestep_state_dim
    
    def get_state_feature(self, state, task):
        # state: [bs, T, *] -> [bs, T, d]
        bs, t, _ = state.size()
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        n_enemies = task_decomposer.n_enemies
        n_entities = task_n_agents + n_enemies
        
        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(state)
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
        proj_query = self.query(entity_embed).reshape(bs * t, n_entities, self.attn_embed_dim)
        proj_key = self.key(entity_embed).reshape(bs * t, n_entities, self.attn_embed_dim)
        energy = th.bmm(proj_query, proj_key.transpose(1, 2)) / (self.attn_embed_dim ** (1 / 2))
        score = F.softmax(energy, dim=-1) # (bs*seq_len, n_entities, n_entities)
        proj_value = entity_embed.reshape(bs * t, n_entities, self.entity_embed_dim)
        out = th.bmm(score, proj_value) # (bs * seq_len, n_entities, entity_embed_dim)
        # mean pooling over entity
        out = out.mean(dim=1).reshape(bs, t, self.entity_embed_dim)
        
        entity_mixing_input = th.cat(
            [out.unsqueeze(2).repeat(1, 1, task_n_agents, 1), 
             ally_embed], 
            dim=-1
        )  # (bs, seq_len, n_agents, x)
        mixing_input = out
        
        return mixing_input, entity_mixing_input


class TaskHead(nn.Module): # TODO
    '''
    compute w via | multitask regression | -task weight explainer-
    '''
    def __init__(self, n_train_task, args, phi_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_train_task, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, phi_dim),
        )
        
    # def forward(self, task_id: th.Tensor):
    #     eps = (th.randn_like(task_id) * 0.1).clamp(-0.1, 0.1)
    #     task_id = (task_id + eps).clamp(0, 1)
    #     return self.net(task_id)
    
    def forward(self, state, task):
        pass

class AttnFeatureExtractor(nn.Module):
    '''
    phi|feature extraction module, based on attention mechanism
    '''
    def __init__(self, task_explainer, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args, is_for_pretrain=False):
        '''
        is_for_pretrain (bool): Default to ``False``, differentiate whether inputs are ``[obs]`` or ``[obs, last_action]``
        '''
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
        self.is_for_pretrain = is_for_pretrain
        self.hypernet_layers = args.hypernet_layers
        self.task_explainer = task_explainer

        ## get obs shape information
        match self.args.env:
            case "sc2" | "sc2_v2":
                obs_own_dim, obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.aligned_own_obs_dim,
                    surrogate_decomposer.aligned_obs_nf_en,
                    surrogate_decomposer.aligned_obs_nf_al,
                )
                ## enemy_obs ought to add attack_action_infos
                if not is_for_pretrain and self.args.obs_last_action:
                    obs_en_dim += 1
            case "gymma":
                obs_own_dim, obs_en_dim, obs_al_dim = (
                    surrogate_decomposer.own_obs_dim,
                    surrogate_decomposer.obs_nf_en,
                    surrogate_decomposer.obs_nf_al,
                )
                ## enemy_obs ought to add attack_action_infos
                if not is_for_pretrain and self.args.obs_last_action:
                    obs_en_dim += surrogate_decomposer.n_actions_attack
            case _:
                raise NotImplementedError

        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        wrapped_obs_own_dim = obs_own_dim + self.args.id_length

        if not is_for_pretrain and self.args.obs_last_action:
            wrapped_obs_own_dim += n_actions_no_attack + 1

        assert self.attn_embed_dim % self.args.head == 0
        # obs self-attention modules
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim * self.args.head)
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim * self.args.head)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim * self.args.head)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim * self.args.head)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim * self.args.head)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim * self.args.head)

        attn_feature_dim = self.entity_embed_dim * self.args.head * 3
        self.layernorm1 = nn.LayerNorm(attn_feature_dim)
        self.layernorm2 = nn.LayerNorm(self.entity_embed_dim * self.args.head)
        self.feat_trfm = FCNet(attn_feature_dim, self.entity_embed_dim * self.args.head, use_last_activ=True)
        self.phi_clamp = .1

        if is_for_pretrain:
            # inverse dynamic predictor
            self.na_action_layer = nn.Sequential(
                nn.Linear(self.entity_embed_dim * self.args.head * 2, self.args.rnn_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.args.rnn_hidden_dim, n_actions_no_attack)
            )
            self.attack_action_layer = nn.Sequential(
                nn.Linear(self.args.rnn_hidden_dim + self.entity_embed_dim * self.args.head * 2, self.args.rnn_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.args.rnn_hidden_dim, 1)
            )
            self.enemy_embed = nn.Sequential( # NOTE it has a different semantic to that in ValueModule
                nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
            )

    def _hyper_forward(self, x, z, net1: nn.Module, bias1, net2=None, bias2=None):
        w1 = th.abs(net1(z))
        b1 = bias1(z)
        hidden = F.elu(th.bmm(x, w1) + b1)
        w2 = th.abs(net2(z))
        b2 = bias2(z)
        out = th.bmm(hidden, w2) + b2 # FIXME fixdim
        return out

    def feature_forward(self, inputs, task): # \bar{\phi}
        # inputs: [bsn|bsTn, *]
        attn_feature, enemy_feat = self._get_attn_feature(inputs, task)
        attn_feature = self.layernorm1(attn_feature)
        out = self.layernorm2(self.feat_trfm(attn_feature))
        out = F.tanh(out) * self.phi_clamp
        return out, enemy_feat
    
    # Only for pretrain
    def pretrain_forward(self, state, obs, task):
        # state: [bs, T, *], obs: [bs, T, n, *]
        bs, t, n_agents, _ = obs.size()
        z_state, z_obs, ... = self.task_explainer(state, obs, task) # TODO may also attach mixing w; state is used as global trajectory
        # obs = obs.reshape(bs*t*n_agents, -1)
        attn_feature, enemy_states = self.feature_forward(obs, task) # [bs, T, n, d]
        phi_bar = attn_feature[:, :-1] # [bs, T-1, n, d] # FIXME 可能改下名字，局部叫phi，合并叫phibar
        next_phi_bar = attn_feature[:, 1:]

        # 1. phi mix
        mixing_w = self.phi_mix_w(z_state).unsqueeze(1) # [bs, 1, d] # CHECK add|learn noise ?
        mixing_n = self.phi_mix_n(z_obs).squeeze(-1) # [bs, n]
        r_bar = ((next_phi_bar - phi_bar) * mixing_w).sum(-1) # [bsT, n]
        r_hat = (r_bar * mixing_n).sum(-1, keepdim=True) # [bsT, 1]
        
        # 2. invdyn
        action_pred_in = th.cat([phi_bar, next_phi_bar], dim=-1)  # (bsTn, 2d)
        # na_action_pred = self.na_action_layer(action_pred_in)
        na_action_pred = self._hyper_forward(action_pred_in, z_obs, self.na_action_layer1, self.na_action_bias1, self.na_action_layer2, self.na_action_bias2)
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_states) # (bsTn, n_enemy, d_phi)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    action_pred_in.unsqueeze(1).repeat(1, enemy_states.size(1), 1)
                ],
                dim=-1
            ) # (bs, n_enemy, hid+2d)
            # attack_action_pred = self.attack_action_layer(attack_action_input).squeeze(-1)
            attack_action_pred = self._hyper_forward(attack_action_input, z_obs, self.attack_action_layer1, self.attack_action_bias1, self.attack_action_layer2, self.attack_action_bias2).squeeze(-1)
            action_pred = th.cat([na_action_pred, attack_action_pred], dim=-1)
        else:
            action_pred = na_action_pred

        action_pred = F.softmax(action_pred, dim=-1)
        
        return r_hat, action_pred

    # Only for RL arch
    def attn_forward(self, inputs, task): 
        feature, enemy_feat = self.feature_forward(inputs, task)
        return feature, enemy_feat
    

class ValueModule(nn.Module):
    '''
    compute psi|Q value based on 
    '''
    def __init__(self, surrogate_decomposer, args, phi_dim):
        super().__init__()
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.phi_dim = phi_dim
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)
        ## get obs shape information
        match self.args.env:
            case "sc2" | "sc2_v2":
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
        # self.rnn = nn.GRUCell(self.entity_embed_dim * self.args.head * 3, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.entity_embed_dim * self.args.head, self.args.rnn_hidden_dim)

        ## attack action networks
        # self.no_attack_psi = nn.Sequential(
        #     nn.Linear(self.hidden_dim + self.n_actions_no_attack, 2 * self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(2 * self.hidden_dim, self.phi_dim),
        # )
        # self.attack_psi = nn.Sequential(
        #     nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(2 * self.hidden_dim, self.phi_dim)
        # )
        # self.no_attack_psi = nn.Linear(self.hidden_dim + self.n_actions_no_attack, self.phi_dim)
        self.no_attack_psi = FCNet(self.hidden_dim + self.n_actions_no_attack, self.phi_dim)
        self.attack_psi = FCNet(2 * self.args.rnn_hidden_dim, self.phi_dim)
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
        
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_feats) # (bsn, n_enemy, hid)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    h.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
                ],
                dim=-1,
            )  # (bsn, n_enemy, 2*hid)
            attack_action_psi = self.attack_psi(attack_action_input) # (bsn, n_enemy, d_phi)
            psi = th.cat([wo_action_psi, attack_action_psi], dim=1)
        else:
            psi = wo_action_psi

        return h, psi

class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=2, hidden_dim=512, use_leaky_relu=True, use_last_activ=False):
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
            layers.append(self.activ)
            
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.layers(x)
