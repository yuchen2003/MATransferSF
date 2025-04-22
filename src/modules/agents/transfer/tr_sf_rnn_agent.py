import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed, onehot_embed, pad_shape
from utils.calc import count_total_parameters
from utils.rl_utils import RunningMeanStd, EMAMeanStd

# ---------------------- TranSSFer implement ----------------------

class TrSFRNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, all_task, train_mode,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args) -> None:
        """  
        Args:
            train_mode (str): Set the agent running mode.
        Modes:
            pretrain: update only feature extractor modules;
            offline: update all modules;
            online: update policy head;
            adapt: update only task weight head;
        """ 
        super(TrSFRNNAgent, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)
        self.all_task = all_task

        # define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.phi_dim = args.phi_dim

        # define various networks
        task_spec_in = [task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args, self.phi_dim]
        self.task_explainer = TaskHead(*task_spec_in, all_task) # should update slowly ~ target net -> wreg may have stablized it
        self.phi_gen = AttnFeatureExtractor(*task_spec_in, self.task_explainer, is_for_pretrain=True)
        self.attn_enc = AttnFeatureExtractor(*task_spec_in, None)
        self.value = ValueModule(surrogate_decomposer, args, self.phi_dim)
        
        for module in [self.task_explainer, self.phi_gen, self.attn_enc, self.value]:
            count_total_parameters(module)
        print("Agent: ")
        count_total_parameters(self, is_concrete=True)
        
        train_mode = train_mode.lower()
        assert train_mode in ['pretrain', 'offline', 'online', 'adapt']
        print(f"Model in [{train_mode}] training mode.")
        self.mode = train_mode

    def init_hidden(self):
        return th.zeros(1, self.args.rnn_hidden_dim, device=self.args.device)

    def pretrain_forward(self, state, state_mask, obs, task):
        r_pred, action_pred, mixing_w = self.phi_gen.pretrain_forward(state, state_mask, obs, task)
        return r_pred, action_pred, mixing_w
    
    def phi_forward(self, obs, task): # get phi upon RL stages
        n_agents = self.task2n_agents[task]
        bs = obs.shape[0] // n_agents
        phi_out, _ = self.phi_gen.feature_forward(obs, task) # NOTE no grad flows back to phi (stage >= off)
        phi_cur = phi_out[:, :-1]
        phi_next = phi_out[:, 1:]
        phi = phi_next - phi_cur
        
        return phi
    
    def forward(self, inputs, hidden_state, task, mixing_w): # psi forward
        n_agents = self.task2n_agents[task]
        bs = inputs.shape[0] // n_agents
        if len(mixing_w.shape) == 1: # [d,] | [bs, d] -> [n, d] | [bsn, d] == attn_feature.shape
            mixing_w = mixing_w.unsqueeze(0).expand(n_agents, -1)
        elif len(mixing_w.shape) == 2:
            mixing_w = mixing_w.unsqueeze(1).repeat(1, n_agents, 1).reshape(bs * n_agents, -1)
        else:
            raise NotImplementedError
        
        attn_feature, enemy_feats = self.attn_enc.attn_forward(inputs, task)
        if self.mode in ['online', 'adapt']:
            attn_feature = attn_feature.detach()
        h, psi = self.value.forward(attn_feature, enemy_feats, hidden_state, mixing_w)
        
        # psi: (bsn, n_act, d_phi) -> (bs, n, n_act, d_phi)
        psi = psi.view(bs, n_agents, -1, self.phi_dim)
        return h, psi

    def explain_task(self, task, state=None, state_mask=None, test_mode=True):
        if test_mode:
            assert state is None and state_mask is None
            # Only for inference
            mixing_w = self.task_explainer.read_explain(task)
            mixing_n = None
        else:
            # Only for RL train
            mixing_w, mixing_n = self.task_explainer(state, state_mask, task)
        return mixing_w, mixing_n
        
    def save(self, path):
        th.save(self.state_dict(), "{}/agent.th".format(path))
        th.save(self.task_explainer.task2w_ms, "{}/task_embed.th".format(path))
        
    def load(self, path):
        # More fine-grained save and load can be implemented
        missing, unexpected = self.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage), strict=False)
        print(f"Missing: {missing}, Unexpected: {unexpected}.")
                
        self.task_explainer.task2w_ms = th.load("{}/task_embed.th".format(path))
        for k, v in self.task_explainer.task2w_ms.items():
            if v.var is not None:
                print(f"read {k} std: {np.sqrt(np.mean(v.var))}")
        
        saved_task = set(self.task_explainer.task2w_ms.keys())
        all_task = set(self.all_task)
        new_task = all_task - saved_task
        self.task_explainer.register_task(list(new_task))

# ---------------------- Main components ----------------------

class PIObsDecomposer(nn.Module):
    ''' Population invariant decomposer: ``input(scalable w. n agents) -> output(fixed embeddings)``; Can decompose observations;
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
        self.concat_obs_act = concat_obs_act

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
            case "gymma" | "grid_mpe":
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
        
        self.obs_en_dim = obs_en_dim
        
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
        # obs: (~, *)
        shape = obs.shape
        obs = obs.reshape(np.prod(shape[:-1]), shape[-1])
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
        n_enemies = len(enemy_feats)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info (bs*n_agents, n_enemies, 1) into enemy_feats
        enemy_feats = th.stack(enemy_feats, dim=1)
        if not self.concat_obs_act and self.have_attack_action: 
            attack_action_info = attack_action_info.unsqueeze(-1)
            enemy_feats = th.cat([enemy_feats, attack_action_info], dim=-1)
        # (bs*n_agents, n_enemies, obs_nf_en+1)
        ally_feats = th.stack(ally_feats, dim=1)

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
        
        attn_feature = attn_feature.reshape(*shape[:-1], -1)
        enemy_feats = enemy_feats.reshape(*shape[:-1], n_enemies, -1)

        return attn_feature, enemy_feats
    
class TaskHead(nn.Module):
    ''' Popolation invariant decomposer: ``input(scalable w. n agents) -> output(fixed embeddings)``; Can decompose states;
    '''
    def __init__(self, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args, phi_dim,
                 all_task):
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
        self.phi_dim = phi_dim

        match self.args.env:
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
        self.agent_feat_trfm = FCNet(self.entity_embed_dim, self.entity_embed_dim, hidden_dim=2*self.entity_embed_dim)
        self.agent_ln1 = nn.LayerNorm(self.entity_embed_dim)
        self.agent_ln2 = nn.LayerNorm(self.entity_embed_dim)

        self.temp_query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.temp_key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.traj_feat_trfm = FCNet(self.entity_embed_dim, self.entity_embed_dim, hidden_dim=2*self.entity_embed_dim)
        self.traj_ln1 = nn.LayerNorm(self.entity_embed_dim)
        self.traj_ln2 = nn.LayerNorm(self.entity_embed_dim)
        
        self.w_proj = nn.Linear(self.entity_embed_dim, self.phi_dim)
        self.n_proj = FCNet(self.entity_embed_dim, 1, 2, self.entity_embed_dim)
        
        self.task2w_ms = {}
        self.register_task(all_task)
        
    def register_task(self, all_task):
        for task in all_task:
            wms = EMAMeanStd()
            self.task2w_ms[task] = wms
        
    def read_explain(self, task):
        m = self.task2w_ms[task].mean
        if m is None:
            return th.ones(self.phi_dim, device=self.args.device) / self.phi_dim
        else:
            return th.as_tensor(self.task2w_ms[task].mean, device=self.args.device)
        
    def _attention(self, q, k, v, attn_dim, mask=None):
        """
            q: [bsn, t, attn_dim]
            k: [bsn, t, attn_dim]
            v: [bsn, t, value_dim]
            mask: [bsn, t, 1]
        """
        assert self.args.head == 1
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        energy = th.bmm(q, k.transpose(1, 2))/(attn_dim ** (1 / 2))
        if mask is not None:
            mask = mask.float()
            attn_mask = 1 - th.bmm(mask, mask.transpose(1, 2)) # [bsn, t, t]
            energy = energy - attn_mask * (1e6)
        score = F.softmax(energy, dim=-1)
        out = th.bmm(score, v).squeeze(1)
        return out
    
    def forward(self, state, state_mask, task):
        # NOTE: state sequences are regarded as trajectories
        # state: [bs, T, *] -> [bs, d];
        bs, t, _ = state.size()
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        n_enemies = task_decomposer.n_enemies
        n_entities = task_n_agents + n_enemies
        state_mask = state_mask.unsqueeze(1).expand(-1, n_entities, -1, 1).reshape(bs * n_entities, t, 1)
        
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
        
        # do temporal attention on trajectory level
        entity_query = self.temp_query(entity_embed).permute(0, 2, 1, 3).reshape(bs * n_entities, t, self.attn_embed_dim)
        entity_key = self.temp_key(entity_embed).permute(0, 2, 1, 3).reshape(bs * n_entities, t, self.attn_embed_dim)
        entity_value = entity_embed.permute(0, 2, 1, 3).reshape(bs * n_entities, t, self.entity_embed_dim)
        entity_attn = self._attention(entity_query, entity_key, entity_value, self.attn_embed_dim, state_mask)
        entity_attn = entity_attn * state_mask # NOTE 原则上是对的
        entity_ln1 = self.traj_ln1(entity_attn + entity_value)
        entity_ln2 = self.traj_ln2(entity_ln1 + self.traj_feat_trfm(entity_ln1))
        entity_ln2 = entity_ln2.reshape(bs, n_entities, t, self.entity_embed_dim)
        entity_out = entity_ln2.mean(dim=2)

        # do entity-wise attention # TODO add multihead attn
        proj_query = self.query(entity_out)
        proj_key = self.key(entity_out)
        proj_value = entity_out
        attn = self._attention(proj_query, proj_key, proj_value, self.attn_embed_dim) # [bs, n_ent, d]
        attn_ln1 = self.agent_ln1(attn + proj_value)
        attn_ln2 = self.agent_ln2(attn_ln1 + self.agent_feat_trfm(attn_ln1))
        # mean pooling over entity
        out = attn_ln2.mean(dim=1) # [bs, d]
        
        mixing_w = self.w_proj(out).abs() # [bs, d_phi] # could be learnable Gaussian 
        mixing_n = self.n_proj(entity_ln2[:, :task_n_agents]).squeeze(-1).abs() # [bs, T, n]
        # mixing_n = self.n_proj(attn_ln2[:, :task_n_agents]).squeeze(-1).abs() # [bs, n]

        return mixing_w, mixing_n

class AttnFeatureExtractor(nn.Module):
    def __init__(self, task2input_shape_info,
                 task2decomposer, task2n_agents,
                 surrogate_decomposer, args, phi_dim,
                 task_explainer, is_for_pretrain=False):
        '''
        phi|feature extraction module, based on attention mechanism
        Args:
            is_for_pretrain (bool): Default to ``False``, differentiate whether inputs are ``[obs]`` or ``[obs, last_action]``
        '''
        super().__init__()
        self.args = args
        self.have_attack_action = (surrogate_decomposer.n_actions != surrogate_decomposer.n_actions_no_attack)
        self.obs_decomposer = PIObsDecomposer(task2input_shape_info, task2decomposer, task2n_agents, surrogate_decomposer, args, concat_obs_act=is_for_pretrain)
        obs_en_dim = self.obs_decomposer.obs_en_dim
        n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
        self.is_for_pretrain = is_for_pretrain
        
        if is_for_pretrain:
            self.task_explainer = task_explainer
            # inverse dynamic predictor
            invdyn_feature_dim = 2 * phi_dim
            self.na_action_layer = CondNet(invdyn_feature_dim, phi_dim, n_actions_no_attack, cond_type=args.invdyn_cond_type)
            self.attack_action_layer = CondNet(self.args.rnn_hidden_dim + invdyn_feature_dim, phi_dim, 1, cond_type=args.invdyn_cond_type)
            self.enemy_embed = nn.Sequential( # NOTE it has a different semantic to that in ValueModule
                nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
            )
            
            attn_feature_dim = self.args.entity_embed_dim * self.args.head * 3  
            self.layernorm1 = nn.LayerNorm(attn_feature_dim)
            self.layernorm2 = nn.LayerNorm(phi_dim)
            self.feat_trfm = FCNet(attn_feature_dim, phi_dim, use_last_activ=True)
        else:
            attn_feature_dim = self.args.entity_embed_dim * self.args.head * 3  
            self.layernorm1 = nn.LayerNorm(attn_feature_dim)
            self.layernorm2 = nn.LayerNorm(self.args.entity_embed_dim * self.args.head)
            self.feat_trfm = FCNet(attn_feature_dim, self.args.entity_embed_dim * self.args.head, use_last_activ=True)

    def feature_forward(self, inputs, task): # phi
        # inputs: [bsn|bsTn, *]
        attn_feature, enemy_feat = self.obs_decomposer.get_obs_feature(inputs, task)
        attn_feature = self.layernorm1(attn_feature)
        attn_feature = self.layernorm2(self.feat_trfm(attn_feature))
        out = attn_feature
        return out, enemy_feat
    
    # Only for pretrain
    def pretrain_forward(self, state, state_mask, obs, task):
        # state: [bs, T, *], obs: [bs, T, n, *]
        bs, t, n_agents, _ = obs.size()
        t = t - 1
        attn_feature, enemy_states = self.feature_forward(obs, task) # [bs, T, n, d_phi]
        phi_cur = attn_feature[:, :-1] # [bs, T-1, d_phi, n]
        phi_next = attn_feature[:, 1:]
        enemy_feat = enemy_states[:, :-1]

        # 1. phi mix
        mixing_w, mixing_n = self.task_explainer(state, state_mask, task) # [bs, d|n], [bs, n, d]
        phi_tilde = (phi_next - phi_cur).transpose(-1, -2)
        # mixing_n = mixing_n.view(bs, 1, 1, -1)
        mixing_n = mixing_n[:, :, :-1].transpose(-1, -2).unsqueeze(-2)
        phi_bar = (phi_tilde * mixing_n).sum(-1)   # [bs, T-1, d_phi]
        # r_hat = (phi_bar * mixing_w.view(bs, 1, -1)).sum(-1, keepdim=True) # [bs, T-1, 1]
        
        # 2. invdyn
        action_pred_in = th.cat([phi_cur, phi_next], dim=-1)  # (bs, T, n, 2d_phi)
        na_action_pred = self.na_action_layer(action_pred_in, mixing_w)
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_feat) # (bs, T, n, n_enemy, d)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    action_pred_in.unsqueeze(3).repeat(1, 1, 1, enemy_states.size(3), 1)
                ],
                dim=-1
            ) # (bs, n_enemy, hid+3d_phi)
            attack_action_pred = self.attack_action_layer(attack_action_input, mixing_w).squeeze(-1)
            action_pred = th.cat([na_action_pred, attack_action_pred], dim=-1)
        else:
            action_pred = na_action_pred

        action_pred = F.softmax(action_pred, dim=-1)
        
        return phi_bar, action_pred, mixing_w

    # Only for RL arch
    def attn_forward(self, inputs, task): 
        return self.feature_forward(inputs, task)
    

class ValueModule(nn.Module):
    '''
    compute psi|Q value
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
            case "gymma" | "grid_mpe":
                obs_en_dim = surrogate_decomposer.obs_nf_en + surrogate_decomposer.n_actions_attack
                n_actions_no_attack = surrogate_decomposer.n_actions_no_attack
            case _:
                raise NotImplementedError
        self.n_actions_no_attack = n_actions_no_attack
        self.id_action_no_attack = th.eye(self.n_actions_no_attack, dtype=th.float32, device=args.device)
        self.rnn = nn.GRUCell(self.entity_embed_dim * self.args.head, self.args.rnn_hidden_dim)

        ## attack action networks
        self.no_attack_psi = FCNet(self.hidden_dim + self.phi_dim + self.n_actions_no_attack, self.phi_dim, hidden_layer=3, use_leaky_relu=False)
        self.attack_psi = FCNet(2 * self.args.rnn_hidden_dim + self.phi_dim, self.phi_dim, hidden_layer=3, use_leaky_relu=False)
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, self.args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        )
    
    def forward(self, attn_feature, enemy_feats, hidden_state, mixing_w):
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(attn_feature, h_in) # (bsn, hid)
        traj_policy_emb = th.cat([h, mixing_w], dim=-1) # (bsn, hid+d_phi)
        
        psi_input = traj_policy_emb.unsqueeze(1).repeat(1, self.n_actions_no_attack, 1) # (bsn, n_no_attack, hid)
        id_action_no_attack = self.id_action_no_attack.repeat(h.size(0), 1, 1)
        psi_input = th.cat([psi_input, id_action_no_attack], dim=-1)
        wo_action_psi = self.no_attack_psi(psi_input) # (bsn, n_no_attack, d_phi)
        
        if self.have_attack_action:
            enemy_feature = self.enemy_embed(enemy_feats) # (bsn, n_enemy, hid)
            attack_action_input = th.cat(
                [
                    enemy_feature,
                    traj_policy_emb.unsqueeze(1).repeat(1, enemy_feats.size(1), 1),
                ],
                dim=-1,
            )  # (bsn, n_enemy, 2*hid)
            attack_action_psi = self.attack_psi(attack_action_input) # (bsn, n_enemy, d_phi)
            psi = th.cat([wo_action_psi, attack_action_psi], dim=1)
        else:
            psi = wo_action_psi
        # psi: [bsn, n_act, d_phi]
        return h, psi

# ---------------------- Other network utilities ----------------------

class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=2, hidden_dim=512, use_leaky_relu=True, use_last_activ=False):
        super().__init__()
            
        layers = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU() if use_leaky_relu else nn.ReLU()]
        for l in range(hidden_layer - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU() if use_leaky_relu else nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        if use_last_activ:
            layers.append(nn.LeakyReLU() if use_leaky_relu else nn.ReLU())
            
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.layers(x)

class CondNet(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim, hidden_layer=2, hidden_dim=64, cond_type: str="cat"):
        super().__init__()
        self.activ = F.tanh
        self.layers = []
        
        self.cond_type = cond_type.lower()
        match self.cond_type:
            case 'cat':
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                for l in range(hidden_layer - 1):
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Linear(hidden_dim, out_dim))
            case 'concat': 
                self.layers.append(nn.Linear(in_dim + cond_dim, hidden_dim))
                for l in range(hidden_layer - 1):
                    self.layers.append(nn.Linear(hidden_dim + cond_dim, hidden_dim))
                self.layers.append(nn.Linear(hidden_dim + cond_dim, out_dim))
            case 'film': 
                self.layers.append(nn.Linear(in_dim, hidden_dim))
                for l in range(hidden_layer - 1):
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Linear(hidden_dim, out_dim))
                self.film_net = FCNet(cond_dim, 2 * hidden_layer, 1, hidden_dim) # film generator; act as fine grained activ
                self.hidden_layer = hidden_layer
            case _:
                raise ValueError('Currently support directly [cat]enate, [concat]enate per layer and [FiLM] arch.')
        
        self.layers = nn.ModuleList(self.layers)
            
    def forward(self, x, z):
        """
        Args:
            x (Tensor): inputs [bs, ..., in_dim]
            z (Tensor): condition [bs, cond_dim] (film) or [bs, ..., cond_dim] (other)
        """
        match self.cond_type:
            case 'cat':
                for f in self.layers[:-1]:
                    x = f(x)
                    x = self.activ(x)
                x = self.layers[-1](x)
            case 'concat':
                z = pad_shape(z, x, pos=1).expand(list(x.shape[:-1]) + list([z.shape[-1]]))
                for f in self.layers[:-1]:
                    x = f(th.cat([x, z], dim=-1))
                    x = self.activ(x)
                x = self.layers[-1](th.cat([x, z], dim=-1))
            case 'film':
                film_gen = self.film_net(z) # [bs, 2n_layer]
                film_gen = pad_shape(film_gen, x, extra=1, pos=2) # [bs, 2n_layer, 1...] ~= x.shape
                gamma, beta = th.split(film_gen, self.hidden_layer, dim=1) # each [bs, n_layer, 1...]
                
                for i, f in enumerate(self.layers[:-1]):
                    x = f(x)
                    x = gamma[:, i] * x + beta[:, i]
                    x = self.activ(x)
                x = self.layers[-1](x)

        return x
    