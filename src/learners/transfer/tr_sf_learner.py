import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.multi_task.mt_vdn import MTVDNMixer
from modules.mixers.multi_task.mt_qatten import MTQMixer
from modules.mixers.qmix import QMixer
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
        self.lr_w = 0.2
        self.offline_train_steps = main_args.offline_train_steps
        self.online_train_steps = main_args.t_max
        self.phi_dim = main_args.phi_dim
        
        # loss weights
        self.lambda_recon = 0.1
        self.lambda_r = 1
        self.lambda_l1 = 1 # not sensitive
        self.vae_beta = 1
        self.lambda_align = 1

        self.mixer = None
        if main_args.mixer is not None:
            match main_args.mixer:
                case "mt_vdn":
                    self.mixer = MTVDNMixer()
                case "mt_qattn":
                    self.mixer = MTQMixer(self.surrogate_decomposer, main_args)
                case "qmix":
                    self.mixer = QMixer(main_args)
                case _:
                    raise ValueError("Mixer {} not recognised.".format(main_args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        match self.main_args.optim_type.lower():
            case "rmsprop":
                self.pretrain_optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
                self.offline_optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
                self.online_optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
            case "adam":
                self.pretrain_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
                self.offline_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
                self.online_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
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
    
    def _w_rectify(self, bs, T_1, n_agents, phi_detach, r_shap_detach, r, task):
        '''
        Clone a weight from mac.task.weights, optimize it, and upload it to mac. After call this function, a task weight is instantiated as optimized current task weight.
        '''
        weight = self.mac.task2weights[task].detach().clone().requires_grad_()
        self.optim_w = SGD([weight], lr=self.lr_w)
        for ep_w in range(self.num_ep_w):
            w_repeat = weight.unsqueeze(1).repeat(bs, n_agents, 1) # (1, d_phi) -> (bs, n, d_phi)
            w_repeat = w_repeat.view(-1, self.phi_dim).unsqueeze(1).repeat(1, T_1, 1) # (bsn, T-1, d_phi)
            r_hat = th.sum(phi_detach * w_repeat, dim=-1) + r_shap_detach # (bsn, T-1)
            r_hat = r_hat.view(bs, n_agents, -1).sum(dim=1) # (bsn, T-1) -> (bs, n, T-1) -> (bs, T-1) # FIXME a priori VDN
            r_loss = th.mean(th.square(r_hat - r))
            w_reg = self.lambda_l1 * th.mean(th.sum(th.abs(weight), dim=-1))
            w_loss = r_loss + self.lambda_l1 * w_reg
            
            self.optim_w.zero_grad()
            w_loss.backward()
            self.optim_w.step()
            
        self.mac.task2weights[task] = weight.detach().clone()
        return weight.detach()
    
    def pretrain(self, batch, t_env, episode, task: str):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        n_agents = self.task2n_agents[task]
        bs = batch.batch_size
        T_1 = batch.max_seq_length - 1
        
        self.mac.init_hidden(bs, task)
        phi, mu, logvar, r_shap, action_recon = [], [], [], [], []
        for t in range(1, batch.max_seq_length):
            phi_t, mu_t, logvar_t, r_shap_t, action_recon_t = self.mac.pretrain_forward(batch, t, task)
            phi.append(phi_t)
            mu.append(mu_t)
            logvar.append(logvar_t)
            r_shap.append(r_shap_t)
            action_recon.append(action_recon_t)
        phi = th.stack(phi, dim=1) # Concat over time #!(bsn, T-1, d_*)
        mu = th.stack(mu, dim=1)
        logvar = th.stack(logvar, dim=1)
        r_shap = th.stack(r_shap, dim=1)
        action_recon = th.stack(action_recon, dim=1)
        
        # r_shap = th.zeros_like(r_shap.detach(), device=r_shap.device) # test if r_shaping has some affect: r_loss cannot decrease (step 2k: 0.08 vs. 0.05)
            
        loss, phi_loss, recon_loss, KLD = 0, 0, 0, 0 
        r = rewards.squeeze(-1) # (bs, T-1)
        a = actions_onehot.transpose(1, 2).reshape(bs * n_agents, -1, actions_onehot.shape[-1]) # (bs, T-1, n_agents, n_act) -> (bs, n, T-1, n_act) -> (bsn, T-1, n_act)
        
        phi_detach = phi.detach() # (bsn, T-1, d_phi)
        r_shap_detach = r_shap.detach()
        weight = self._w_rectify(bs, T_1, n_agents, phi_detach, r_shap_detach, r, task)
        
        w_repeat_detach = weight.clone().detach().unsqueeze(1).repeat(bs, n_agents, 1) # (bs, n, d_phi)
        w_repeat_detach = w_repeat_detach.view(-1, self.phi_dim).unsqueeze(1).repeat(1, T_1, 1) # (bsn, T-1, d_phi)
        r_hat_out = th.sum(phi * w_repeat_detach, dim=-1) + r_shap # (bsn, T-1)
        r_hat_out = r_hat_out.view(bs, n_agents, -1).sum(dim=1) # (bsn, T-1) -> (bs, n, T-1) -> (bs, T-1)
        
        phi_loss += th.mean(th.square(r_hat_out - r))
        recon_loss += F.binary_cross_entropy(action_recon, a)
        KLD += -.5 * th.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = phi_loss + self.lambda_recon * (recon_loss + self.vae_beta * KLD)

        self.pretrain_optimiser.zero_grad()
        loss.backward()
        self.pretrain_optimiser.step()
        
        if t_env - self.pretrain_last_log_t >= 100:
            self.logger.log_stat(f"loss", loss.item(), t_env)
            self.logger.log_stat(f"recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat(f"phi_loss", phi_loss.item(), t_env)
            self.logger.log_stat(f"KLD", KLD.item(), t_env)
            self.pretrain_last_log_t = t_env
            print(f"step: {t_env}, loss: {loss.item()}, recon: {recon_loss.item()}, phi: {phi_loss.item()}, avg KLD: {KLD.item()}")
    
    def train(self, batch, t_env: int, episode_num: int, task: str, is_online=False):
        # TODO 或者把整个offline阶段都作为pretrain，类似于m3的形式；不过现在这样代码清晰一点
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
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
        n_agents = self.task2n_agents[task]
        bs = batch.batch_size
        T_1 = batch.max_seq_length - 1
        r = rewards.squeeze(-1) # (bs, T-1)
        a = actions_onehot.transpose(1, 2).reshape(bs * n_agents, -1, actions_onehot.shape[-1]) # (bs, T-1, n_agents, n_act) -> (bs, n, T-1, n_act) -> (bsn, T-1, n_act)
            
        if self.main_args.standardise_rewards:
            # NOTE phi is calculated by the encoder
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
        
        # Calculate estimated Q-Values -> psi-values, Q-values
        psi_out, phi_out, phi_hat_out, phi_tilde_out, r_shap_out = [], [], [], [], []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            psi, phi, phi_hat, phi_tilde, r_shap = self.mac.forward(batch, t=t, task=task)
            psi_out.append(psi)
            phi_out.append(phi)
            # phi_hat_out.append(phi_hat)
            phi_tilde_out.append(phi_tilde)
            r_shap_out.append(r_shap)
        psi_out = th.stack(psi_out, dim=1) # (bs, T, n, d_phi, n_act) 
        phi_out = th.stack(phi_out, dim=1)[:, :-1] # (bsn, T-1, d_phi) 
        # phi_hat_out = th.stack(phi_hat_out, dim=1)[:, :-1] 
        phi_tilde_out = th.stack(phi_tilde_out, dim=1)[:, :-1]
        r_shap_out = th.stack(r_shap_out, dim=1)[:, :-1]
        
        # Calc losses
        loss = 0
        ## 1. phi & w co-training
        # do not update phi encoder upon online learning | only psi network is updated
        phi_detach = phi_out.detach() # (bsn, T-1, d_phi)
        r_shap_detach = r_shap_out.detach()
        weight = self._w_rectify(bs, T_1, n_agents, phi_detach, r_shap_detach, r, task) # FIXME a priori VDN; should integrage mixing network for composing <phi^i, w> into r like QPLEX; #!这里的r如何mix决定了之后的Q应该如何mix (phi-mix <=> psi-mix <=> q-mix)
        # w: (1, d_phi) -> (d_phi, )
        weight = weight.detach().mean(dim=0)
        if not is_online: 
            w_repeat = weight.unsqueeze(1).repeat(bs, n_agents, 1) # (bs, d_phi) -> (bs, n, d_phi)
            w_repeat = w_repeat.view(-1, self.phi_dim).unsqueeze(1).repeat(1, T_1, 1) # (bsn, T-1, d_phi)
            
            r_hat_out = th.sum(phi_out * w_repeat, dim=-1) + r_shap_out # (bsn, T-1) # TODO should r_shap be updated in online ?
            r_hat_out = r_hat_out.view(bs, n_agents, -1).sum(dim=1) # (bsn, T-1) -> (bs, n, T-1) -> (bs, T-1) # FIXME a priori VDN (phi-mix)
            loss += th.mean(th.square(r_hat_out - r))
        
        ## 2. TD error for psi and Q
        psi_out[expand_avail_actions == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_psi_na_vals = th.gather(psi_out[:, :-1], dim=-1, index=expand_actions).squeeze(-1)

        # Calculate the Q-Values necessary for the target
        target_psi_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_psi, target_phi, target_phi_hat, target_phi_tilde, target_r_shap = self.target_mac.forward(batch, t=t, task=task)
            target_psi_out.append(target_psi)
            
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_psi_out = th.stack(target_psi_out[1:], dim=1) # (bs, T-1, n, d_phi, n_act), psi^i, corresponding reward/phi follows VDN composition
        
        # Mask out unavailable actions
        target_psi_out[expand_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            psi_out_detach = psi_out.clone().detach()
            expand_cur_max_actions = psi_out_detach.max(dim=4, keepdim=True)[1] # (bs, T, n, d_phi, 1)
            target_max_na_psi_vals = th.gather(target_psi_out, 4, expand_cur_max_actions[:, 1:]).squeeze(4)
            cons_max_psi_vals = th.gather(psi_out, 4, expand_cur_max_actions).squeeze(4)
        else:
            target_max_na_psi_vals = target_psi_out.max(dim=4)[0]
        
        if self.mixer is not None:
            chosen_action_psi_vals = self.mixer(chosen_action_psi_na_vals, batch["state"][:, :-1], self.task2decomposer[task])
            target_max_psi_vals = self.target_mixer(target_max_na_psi_vals, batch["state"][:, :-1], self.task2decomposer[task]) # (bs, T-1, 1, d_phi)
            cons_max_psi_vals = self.mixer(cons_max_psi_vals, batch["state"][:, :-1], self.task2decomposer[task])
        
        # Calculate 1-step Q-Learning targets
        # NOTE Q is calculated by psi, only forward mixer one time
        phi_tilde_out = phi_tilde_out.view(-1, n_agents, T_1, self.phi_dim).transpose(1, 2) # (bs, T-1, n_agents, d_phi)
        phi_tilde_mixed = phi_tilde_out.sum(2, keepdim=True) # FIXME phi-mix (bs, T-1, 1, d_phi)
        psi_targets = phi_tilde_mixed + self.main_args.gamma * (1 - expand_terminated) * target_max_psi_vals
        targets = rewards + self.main_args.gamma * (1 - terminated) * (target_max_psi_vals * weight).sum(-1) #(bs, T-1, 1)

        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            self.task2psi_ms[task].update(psi_targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
            psi_targets = (psi_targets - self.task2psi_ms[task].mean) / th.sqrt(self.task2psi_ms[task].var)
        
        # Td-error
        psi_td_error = (chosen_action_psi_vals - psi_targets.detach())
        chosen_action_qvals = (chosen_action_psi_vals * weight).sum(-1)
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        
        psi_mask = psi_mask.expand_as(psi_td_error)
        masked_psi_td_error = psi_td_error * psi_mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        psi_td_loss = (masked_psi_td_error ** 2).sum() / psi_mask.sum()
        
        loss = td_loss + psi_td_loss
        
        # Optimise
        self.offline_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.offline_optimiser.step()

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
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
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

    def save_models(self, path, stage):
        pass
        # FIXME currently not saving
        # match stage:
        #     case 0: 
        #         th.save(self.pretrain_optimiser.state_dict(), "{}/pt_opt.th".format(path))
        #     case 1: 
        #         th.save(self.offline_optimiser.state_dict(), "{}/off_opt.th".format(path))
        #     case 2: 
        #         th.save(self.online_optimiser.state_dict(), "{}/on_opt.th".format(path))
        #     case _: raise ValueError
        # self.mac.save_models(path) # each model is saved up to its training stage
        # if self.mixer is not None:
        #     th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        
    def load_models(self, path, stage):
        match stage:
            case 0:
                self.pretrain_optimiser.load_state_dict(th.load("{}/pt_opt.th".format(path), map_location=lambda storage, loc: storage))
            case 1: 
                self.offline_optimiser.load_state_dict(th.load("{}/off_opt.th".format(path), map_location=lambda storage, loc: storage))
            case 2:
                self.online_optimiser.load_state_dict(th.load("{}/on_opt.th".format(path), map_location=lambda storage, loc: storage))
            case _: raise ValueError
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            