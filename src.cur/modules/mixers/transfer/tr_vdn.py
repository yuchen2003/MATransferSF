import torch as th
import torch.nn as nn


class MTVDNMixer(nn.Module):
    '''For mixing SFs.'''
    def __init__(self, surrogate_decomposer, main_args):
        super(MTVDNMixer, self).__init__()
        match main_args.env:
            case "sc2":
                state_dim = surrogate_decomposer.aligned_state_dim
            case "gymma":
                state_dim = surrogate_decomposer.obs_dim
        embed_dim = main_args.mixing_embed_dim

        self.with_bias = False
        if self.with_bias:
            self.bias = nn.Sequential(
                nn.Linear(state_dim, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, 1)
            )
        self.gamma = main_args.gamma
        

    def forward(self, agent_qs, states, task=None, phi_mode=False, w_inv=None):
        mixed = th.sum(agent_qs, dim=2, keepdim=True) # (bs, T-1, 1, (d_phi))
        if self.with_bias:
            if phi_mode:
                cur_states = states[:, :-1] # (bs, T-1, dim)
                next_states = states[:, 1:]
                bias = self.bias(cur_states) - self.gamma * self.bias(next_states)
            else:
                bias = self.bias(states) # (bs, T-1, 1)
        else:
            bias = 0
            
        return mixed + bias * w_inv
