import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GymDecomposer:
    def __init__(self, args):
        env, map_name = args.env_args['key'].split(':')
        self.env = env
        self.map_name = map_name
        match env:
            case "lbforaging":
                # Load map params
                self.n_agents = args.n_agents
                self.n_enemies = args.n_foods # should scale wrt foods as [bs, n_foods, 3] instead of [bs, 1, n_foods * 3]

                # state params
                self.state_last_action = False
                self.state_timestep_number = 0
                self.timestep_number_state_dim = 0

                tuple_dim = 3
                self.state_nf_al = tuple_dim
                # self.state_nf_en = tuple_dim * self.n_foods
                self.state_nf_en = tuple_dim

                # obs params
                self.own_obs_dim = tuple_dim
                self.obs_nf_al = tuple_dim
                # self.obs_nf_en = tuple_dim * self.n_foods
                # self.obs_dim = tuple_dim * self.n_agents + self.obs_nf_en
                self.obs_nf_en = tuple_dim
                self.obs_dim = tuple_dim * (self.n_agents + self.n_enemies)
                

                # Actions
                self.decomposed_obs = None
                # self.n_actions_no_attack = 5
                # self.n_actions_attack = 1 # pickup foods
                self.n_actions_no_attack = 6
                self.n_actions_attack = 0 # pickup corresponds to the agent's own behaviour
                self.n_actions = self.n_actions_no_attack + self.n_actions_attack
            case _:
                raise NotImplementedError("Not supported env: {}".format(env))


    def decompose_state(self, state_input):
        # state_input = [enemy_state(food), self_state, ally_state]
        # assume state_input.shape == [batch_size, state]
        
        # extract enemy_states
        enemy_states = [state_input[:, :, i * self.state_nf_en:(i + 1) * self.state_nf_en] for i in range(self.n_enemies)]
        # extract ally_states
        base = self.n_agents * self.state_nf_en
        agent_states = [state_input[:, :, base + i * self.state_nf_al: base + (i + 1) * self.state_nf_al] for i in range(self.n_agents)] # include self_state at [0]

        # extract last_action_states
        last_action_states = []
        # extract timestep_number_state
        timestep_number_state = []

        return agent_states, enemy_states, last_action_states, timestep_number_state

    def decompose_obs(self, obs_input):
        """
        obs_input: food_pos_level + self_pos_level + ally_pos_level
        env_obs = [move_feats, enemy_feats, ally_feats, own_feats]
        """
        
        # extract enemy_feats (foods)        
        enemy_feats = [obs_input[:, i * self.obs_nf_en : (i+1) * self.obs_nf_en] for i in range(self.n_enemies)] # (n_enemy, bs * n_agents, nf_en)
        
        # extract own (lbf: self player is always first)
        base = self.n_enemies * self.obs_nf_en
        own_obs = obs_input[:, base : base + self.own_obs_dim]
        
        # extrace ally_feats
        base += self.own_obs_dim
        ally_feats = [obs_input[:, base + i * self.obs_nf_al : base + (i+1) * self.obs_nf_al] for i in range(self.n_agents - 1)]
      
        return own_obs, enemy_feats, ally_feats

    def decompose_action_info(self, action_info):
        """
        action_info: shape [(bs), n_agent, n_action]
        """
        shape = action_info.shape
        bsn = np.prod(shape[:-1]) # bs x n_agent
        if len(shape) > 2:
            action_info = action_info.reshape(bsn, shape[-1])
        no_attack_action_info = action_info[:, :self.n_actions_no_attack]  
        attack_action_info = action_info[:, self.n_actions_no_attack:]
        bin_attack_info = no_attack_action_info[:, -1:]
        
        # recover shape: (bs, n, *)
        no_attack_action_info = no_attack_action_info.reshape(*shape[:-1], -1)    
        attack_action_info = attack_action_info.reshape(*shape[:-1], -1)
        bin_attack_info = bin_attack_info.reshape(*shape[:-1], -1)
        
        # get compact action
        compact_action_info = th.cat([no_attack_action_info, bin_attack_info], dim=-1) # NOTE picking-up food is regarded as agent' own action, not scalable wrt n_enemies, however also regarded as attack action as in compact_action
        
        return no_attack_action_info, attack_action_info, compact_action_info
    