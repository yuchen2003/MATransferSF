import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.transfer.tr_vdn import MTVDNMixer
from modules.mixers.transfer.tr_qatten import MTQMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transfer.tr_dmaq_qatten import MTDMAQQattnMixer
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam, SGD
from components.standarize_stream import RunningMeanStd
import torch.nn.functional as F
import numpy as np

class TransferSFLearner:
    def __init__(self, mac, logger, main_args) -> None:
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())
        self.pretrain_steps = main_args.pretrain_steps
        self.pretrain_batch_size = main_args.pretrain_batch_size
        self.pretrain_last_log_t = 0
        self.num_ep_w = 50
        self.lr_w = 0.01
        self.offline_train_steps = main_args.offline_train_steps
        self.online_train_steps = main_args.t_max
        self.phi_dim = main_args.phi_dim
        
        # loss weights
        self.lambda_recon = 0.1 # lbf
        self.lambda_r = 1
        self.lambda_l1 = 1 # not sensitive
        self.vae_beta = 1
        self.lambda_align = 1

        self.mixer = None
        if main_args.mixer is not None:
            match main_args.mixer:
                case "tr_vdn":
                    self.mixer = MTVDNMixer(self.surrogate_decomposer, main_args)
                case "tr_qatten":
                    self.mixer = MTQMixer(self.surrogate_decomposer, main_args)
                case "tr_dmaq_qatten":
                    self.mixer = MTDMAQQattnMixer(self.surrogate_decomposer, main_args)
                case "qmix":
                    self.mixer = QMixer(main_args)
                case _:
                    raise ValueError("Mixer {} not recognised.".format(main_args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        match self.main_args.optim_type.lower():
            case "rmsprop":
                self.pretrain_optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
                self.optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
            case "adam":
                self.pretrain_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
                self.optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.task2train_info = {}
        for task, task_args in self.task2args.items():
            self.task2train_info[task] = {
                "log_stats_t": -task_args.learner_log_interval - 1,
                "training_steps": 0
            }
        self.last_target_update_episode = 0
        self.total_training_steps = 0

        device = "cuda" if main_args.use_cuda else "cpu"
        if self.main_args.standardise_returns:
            self.task2ret_ms, self.task2psi_ms = {}, {}
            for task in self.task2args.keys():
                self.task2ret_ms[task] = RunningMeanStd(shape=(self.task2n_agents[task], ), device=device)
                self.tasl2psi_ms[task] = RunningMeanStd(shape=(self.task2n_agents[task], ), device=device)

        if self.main_args.standardise_rewards:
            self.task2rew_ms = {}
            self.task2aw_rew_ms_list = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device)
    
    def _w_rectify(self, batch, phi_detach, r, task):
        '''
        phi: (bsn, T-1, d_phi)|(bs, T-1, n, d_phi)
        Clone a weight from mac.task.weights, optimize it, and upload it to mac. After call this function, a task weight is instantiated as optimized current task weight.
        '''
        weight = self.mac.task2weights[task].detach().clone().requires_grad_()
        # self.optim_w = Adam([weight], lr=self.lr_w)
        # for ep_w in range(self.num_ep_w):
        #     phi_hat = 
        #     r_hat = th.sum(phi_hat * weight, dim=-1) # (bs, T-1, 1)
        #     r_loss = th.mean(th.square(r_hat - r))
        #     w_reg = self.lambda_l1 * th.mean(th.sum(th.abs(weight), dim=-1))
        #     w_loss = r_loss + self.lambda_l1 * w_reg
            
        #     self.optim_w.zero_grad()
        #     w_loss.backward()
        #     self.optim_w.step()
            
        # self.mac.task2weights[task] = weight.detach().clone()
        return weight.detach()
    
    def pretrain(self, batch, t_env, episode, task: str):
        n_agents = self.task2n_agents[task]
        bs = batch.batch_size
        T_1 = batch.max_seq_length - 1
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        action_mask = mask.unsqueeze(1).repeat(1, n_agents, 1, 1).reshape(bs * n_agents, T_1, 1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        self.mac.init_hidden(bs, task)
        phi, mu, logvar, action_recon = [], [], [], []
        # for t in range(1, batch.max_seq_length):
        for t in range(batch.max_seq_length):
            phi_t, mu_t, logvar_t, action_recon_t = self.mac.pretrain_forward(batch, t, task)
            phi.append(phi_t)
            mu.append(mu_t)
            logvar.append(logvar_t)
            action_recon.append(action_recon_t)
        phi = th.stack(phi, dim=1)[: ,:-1] # Concat over time (bs, T-1, n, d_phi)
        mu = th.stack(mu, dim=1)[: ,:-1] # (bsn, T-1, d_phi)
        logvar = th.stack(logvar, dim=1)[: ,:-1]
        action_recon = th.stack(action_recon, dim=1)[: ,:-1] * action_mask
            
        a = actions_onehot.transpose(1, 2).reshape(bs * n_agents, T_1, -1) # (bs, T-1, n_agents, n_act) -> (bs, n, T-1, n_act) -> (bsn, T-1, n_act)
        
        phi_detach = phi.detach() # (bs, T-1, n, d_phi)
        weight = self._w_rectify(batch, phi_detach, rewards, task)
        phi_hat = self.mixer(phi, batch["state"], self.task2decomposer[task], phi_mode=True, w_inv=w_inv)
        r_hat = th.sum(phi_hat * weight, dim=-1) # (bs, T-1, 1)
        phi_loss = th.mean(th.square(r_hat - rewards))
        
        # recon_loss += F.binary_cross_entropy(action_recon, a)
        recon_loss = F.mse_loss(action_recon, a)
        KLD = -.5 * th.mean(th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        loss = phi_loss + self.lambda_recon * (recon_loss + self.vae_beta * KLD)

        self.pretrain_optimiser.zero_grad()
        loss.backward()
        self.pretrain_optimiser.step()

        if t_env - self.pretrain_last_log_t >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"pretrain/loss", loss.item(), t_env)
            self.logger.log_stat(f"pretrain/recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat(f"pretrain/phi_loss", phi_loss.item(), t_env)
            self.logger.log_stat(f"pretrain/KLD", KLD.item(), t_env)
            self.pretrain_last_log_t = t_env
            print("step: {}, loss: {:.6f}, recon: {:.6f}, phi: {:.6f}, KLD: {:.6f}".format(t_env, loss.item(), recon_loss.item(), phi_loss.item(), KLD.item()))
    
    def train(self, batch, t_env: int, episode_num: int, task: str, is_online=False):
        n_agents = self.task2args[task].n_agents
        rewards = batch["reward"][:, :-1] # (bs, T-1, *)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        psi_mask = mask.clone().detach().unsqueeze(-1)
        avail_actions = batch["avail_actions"]
        
        # (bs, T, n, 1) -> (bs, T, n, d_phi, 1) ~ psi_out[:, :, :, :, *]
        expand_actions = actions.unsqueeze(3).repeat(1, 1, 1, self.phi_dim, 1)
        # (bs, T, n, n_act) -> (bs, T, n, d_phi, n_act) ~ psi_out
        expand_avail_actions = avail_actions.unsqueeze(3).repeat(1, 1, 1, self.phi_dim, 1)
        # (bs, T, 1) -> (bs, T, 1, d_phi)
        expand_terminated = terminated.unsqueeze(3).repeat(1, 1, 1, self.phi_dim)
        expand_mask = mask.unsqueeze(3).repeat(1, 1, 1, self.phi_dim)
            
        if self.main_args.standardise_rewards:
            # NOTE phi is calculated by the encoder
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
        
        # Calculate estimated Q-Values -> psi-values, Q-values
        psi_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            psi = self.mac.forward(batch, t=t, task=task)
            psi_out.append(psi)
        psi_out = th.stack(psi_out, dim=1) # (bs, T, n, d_phi, n_act) 
        # weight = self._w_rectify(None, None, None, task)
        weight = 1
        
        psi_out[expand_avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_psi_na_vals = th.gather(psi_out[:, :-1], dim=-1, index=expand_actions).squeeze(-1)

        # Calculate the Q-Values necessary for the target
        target_psi_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_psi = self.target_mac.forward(batch, t=t, task=task)
            target_psi_out.append(target_psi)
            
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_psi_out = th.stack(target_psi_out[1:], dim=1) # (bs, T-1, n, d_phi, n_act)
        
        # Mask out unavailable actions
        target_psi_out[expand_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            psi_out_detach = psi_out.clone().detach()
            expand_cur_max_actions = psi_out_detach.max(dim=4, keepdim=True)[1] # (bs, T, n, d_phi, 1)
            target_max_na_psi_vals = th.gather(target_psi_out, 4, expand_cur_max_actions[:, 1:]).squeeze(4) # (bs, T-1, n, d_phi)
            # cons_max_na_psi_vals = th.gather(psi_out, 4, expand_cur_max_actions).squeeze(4)
        else:
            target_max_na_psi_vals = target_psi_out.max(dim=4)[0]
        
        # Mix
        if self.mixer is not None:
            chosen_action_psi_vals = self.mixer(chosen_action_psi_na_vals, batch["state"][:, :-1], self.task2decomposer[task])
            target_max_psi_vals = self.target_mixer(target_max_na_psi_vals, batch["state"][:, 1:], self.task2decomposer[task]) # (bs, T-1, 1, d_phi)
            # cons_max_psi_vals = self.mixer(cons_max_na_psi_vals, batch["state"], self.task2decomposer[task], w_inv=w_inv)
        
        agent_qs = (psi_out.transpose(-1, -2) * weight).sum(-1) # (bs, T, n, n_act) 
        chosen_action_na_qvals = (chosen_action_psi_na_vals * weight).sum(-1)
        chosen_action_qvals = (chosen_action_psi_vals * weight).sum(-1)
        target_max_qvals = (target_max_psi_vals * weight).sum(-1)
        
        if self.main_args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.task2ret_ms[task].var) + self.task2ret_ms[task].mean
        
        # Calculate 1-step Q-Learning targets
        # FIXME only for testing psi leaerning (so no phi learning occurred)
        targets = rewards + self.main_args.gamma * (1 - terminated) * target_max_qvals #(bs, T-1, 1)

        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
        
        # Td-error
        # psi_td_error = (chosen_action_psi_na_vals - psi_targets.detach())
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        
        # psi_mask = psi_mask.expand_as(psi_td_error)
        # masked_psi_td_error = psi_td_error * psi_mask

        # Normal L2 loss, take mean over actual data (1, 2.1)
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        
        # FIXME only for testing psi leaerning
        loss = td_loss
        
        # CQL: TO handle ood in offRL; may be replaced with qplex
        if not is_online:
            if self.main_args.cql_type == 'base':
                # CQL-error
                cql_error = th.logsumexp(agent_qs[:, :-1], dim=3) - chosen_action_na_qvals
                cql_mask = mask.expand_as(cql_error)
                cql_loss = (cql_error * cql_mask).sum() / mask.sum() # better use mask.sum() instead of cql_mask.sum()
            else:
                raise ValueError("Unknown cql_type: {}".format(self.main_args.cql_type))
            loss += self.main_args.cql_alpha * cql_loss
        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.optimiser.step()
        self.total_training_steps += 1
        self.task2train_info[task]["training_steps"] += 1

        if self.main_args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.main_args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.main_args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.main_args.target_update_interval_or_tau)
        
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            self.logger.log_stat(f"{task}/td_loss", td_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/psi_td_loss", psi_td_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            psi_mask_elems = psi_mask.sum().item()
            if not is_online:
                self.logger.log_stat(f"{task}/cql_loss", cql_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/psi_td_error_abs", (masked_psi_td_error.abs().sum().item()/psi_mask_elems), t_env)
            # self.logger.log_stat(f"{task}/r_loss", r_loss.item(), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        #self.logger.console_logger.info("Updated target network")

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        th.save(self.pretrain_optimiser.state_dict(), "{}/pt_opt.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        
    def load_models(self, path):
        self.pretrain_optimiser.load_state_dict(th.load("{}/pt_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            