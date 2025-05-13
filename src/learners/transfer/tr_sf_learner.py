import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.transfer.tr_vdn import MTVDNMixer
from modules.mixers.transfer.tr_qatten import MTQMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transfer.tr_dmaq_qatten import MTDMAQQattnMixer
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, AdamW, SGD
from torch.optim.lr_scheduler import StepLR
from components.standarize_stream import RunningMeanStd
import torch.nn.functional as F
import numpy as np
from utils.calc import compute_q_values
from utils.embed import pad_shape

class TransferSFLearner:
    def __init__(self, mac, logger, main_args) -> None:
        self.main_args = main_args
        self.mac = mac
        self.train_mode = mac.train_mode
        self.logger = logger

        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())
        self.phi_dim = main_args.phi_dim
        self.reward_scale = getattr(main_args, "reward_scale", 1.) # currenly only for sc2v2: 10

        self.mixer = MTDMAQQattnMixer(self.surrogate_decomposer, main_args)
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
                self.optimiser = RMSprop(
                    params=self.params,
                    lr=self.main_args.lr,
                    alpha=self.main_args.optim_alpha,
                    eps=self.main_args.optim_eps,
                    weight_decay=self.main_args.weight_decay,
                )
            case "adam":
                # self.optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
                self.optimiser = AdamW(params=self.params, lr=self.main_args.lr)
            case _:
                raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        
        if main_args.train_mode == 'pretrain':
            self.w_range_reg = main_args.w_range_reg
            self.w_range_gate = main_args.w_range_gate
            self.r_lambda = main_args.r_lambda
            # self.lr_sched = StepLR(self.optimiser, main_args.lr_decay_step_size, main_args.lr_decay_rate)
        
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
        self.ema_true_q = None

        device = "cuda" if main_args.use_cuda else "cpu"
        if self.main_args.standardise_returns:
            self.task2ret_ms, self.task2psi_ms = {}, {}
            for task in self.task2args.keys():
                self.task2ret_ms[task] = RunningMeanStd(shape=(self.phi_dim, ), device=device)

        if self.main_args.standardise_rewards:
            self.task2rew_ms = {}
            self.task2aw_rew_ms_list = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device) # TODO load for on
                
        # NOTE codes for analysis
        self.w_records = {}
        for task in self.task2args.keys():
            self.w_records[task] = [] # for visualizing task embeddings
    
    def pretrain(self, batch, t_env: int, episode_num: int, task: str):
        states = batch["state"]
        obs = batch["obs"]
        rewards = batch["reward"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        state_mask = batch["filled"]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        phi_bar, action_pred, mixing_w = self.mac.pretrain_forward(states, state_mask, obs, task)
        
        # self.w_records[task].append(mixing_w)
        # print(t_env)
        
        rewards = rewards / self.reward_scale
        if self.main_args.standardise_rewards:
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
    
        w_mean, w_var = self.mac.update_weight(mixing_w, task)
        r_loss = ((phi_bar - rewards * mixing_w.unsqueeze(1)).square() * mask).mean()
        mask = mask.unsqueeze(-1).repeat(1, 1, *actions_onehot.shape[-2:])
        a_loss = F.binary_cross_entropy(action_pred * mask, actions_onehot * mask)
        sgn = (mixing_w.detach() >= 0).float() * 2 - 1
        w_range_reg = (1 / (mixing_w + (1e-5) * sgn)).abs().mean() + mixing_w.abs().mean()
        w_range_reg_gated = w_range_reg * (w_range_reg.detach() >= self.w_range_gate).float()
        loss = a_loss + self.r_lambda * r_loss + self.w_range_reg * w_range_reg_gated
        
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip, error_if_nonfinite=True)
        self.optimiser.step()
        self.total_training_steps += 1
        self.task2train_info[task]["training_steps"] += 1
        
        if t_env % self.main_args.lr_decay_step_size == 0:
            lr_targ = self.main_args.lr * (self.main_args.lr_decay_rate ** (t_env / self.main_args.t_max)) # exponential decay
            # lr_targ = self.main_args.lr / (1 + self.lr_decay_rate * t_env) # reciprocal decay
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr_targ
            print(lr_targ)
        
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/r_loss", r_loss.item(), t_env)
            self.logger.log_stat(f"{task}/a_loss", a_loss.item(), t_env)
            self.logger.log_stat(f"{task}/w_reg", w_range_reg.item(), t_env)
            self.logger.log_stat(f"{task}/w_std", mixing_w.std(0).mean().item(), t_env)
            self.logger.log_stat(f"{task}/w_rec_std", np.mean(np.sqrt(w_var)), t_env)
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            self.task2train_info[task]["log_stats_t"] = t_env
            if grad_norm.item() == 0:
                print("ERROR... GRADS VANISH.")
                exit(-1)

    
    def train(self, batch, t_env: int, episode_num: int, task: str, mode='offline'):
        ''' mode = | offline | online '''
        bs = batch.batch_size
        n_agents = self.task2args[task].n_agents
        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        actions = batch["actions"][:, :-1] # (bs, T-1, n, 1)
        states = batch["state"] # (bs, T-1, ds)
        state_mask = batch["filled"]
        obs = batch["obs"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        rewards = rewards / self.reward_scale
        if self.main_args.standardise_rewards:
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
        
        # Calculate estimated Q-Values -> psi-values, Q-values
        psi_out = []
        mixing_w, mixing_n = self.mac.explain_task(task, states, state_mask)
        w_mean, w_var = self.mac.update_weight(mixing_w, task)
        
        # self.w_records[task].append(mixing_w)
        # print(t_env)
        
        self.mac.init_hidden(batch.batch_size, task)
        # psi_w_in = mixing_w.detach() if mode == 'offline' else mixing_w
        psi_w_in = mixing_w.detach()
        for t in range(batch.max_seq_length):
            psi = self.mac.forward(batch, t=t, task=task, mixing_w=psi_w_in)
            psi_out.append(psi)
        psi_out = th.stack(psi_out, dim=1) # (bs, T, n, n_act, d_phi)
        mac_out = psi_out.transpose(-1, -2) # (bs, T, n, d_phi, n_act)
        avail_actions = avail_actions.unsqueeze(-2).expand_as(mac_out)
        actions = actions.unsqueeze(-2).expand(-1, -1, -1, self.phi_dim, -1)
        mac_out[avail_actions == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_na_qvals = th.gather(mac_out[:, :-1], dim=-1, index=actions).squeeze(-1) # (bs, T-1, n, d_phi)

        # Calculate the Q-Values necessary for the target
        target_psi_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_psi = self.target_mac.forward(batch, t=t, task=task, mixing_w=psi_w_in)
            target_psi_out.append(target_psi)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_psi_out = th.stack(target_psi_out[1:], dim=1) # (bs, T-1, n, n_act, d_phi)
        target_mac_out = target_psi_out.transpose(-1, -2) # (bs, T-1, n, d_phi, n_act)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]
            target_max_na_qvals = th.gather(target_mac_out, -1, cur_max_actions[:, 1:]).squeeze(-1)
        else:
            target_max_na_qvals = target_mac_out.max(dim=-1)[0]

        # Agent mixing: (bs, T-1, n, d_phi) -> (bs, T-1, d_phi)
        if isinstance(self.mixer, MTDMAQQattnMixer):
            max_action_qvals = mac_out[:, :-1].max(dim=-1)[0]
            target_max_action_qvals = target_mac_out.max(dim=-1)[0]
            chosen_action_qvals = self.mixer(chosen_action_na_qvals, max_action_qvals, states[:, :-1], self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_na_qvals, target_max_action_qvals, states[:, 1:], self.task2decomposer[task])
        else:
            chosen_action_qvals = self.mixer(chosen_action_na_qvals, states[:, :-1], self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_na_qvals, states[:, 1:], self.task2decomposer[task])
            
        # chosen_qs_global = ((1 / (mixing_w * self.phi_dim)).view(bs, 1, self.phi_dim) * chosen_action_qvals).sum(-1)
        # q_mask = mask.squeeze(-1)
        # print("q_esti_mean: ", (chosen_qs_global * q_mask).sum().item() / q_mask.sum().item())
            
        loss = 0
        # Calculate 1-step Q-Learning targets
        
        if mode == 'offline':
            phi_tilde = self.mac.phi_forward(obs, task).transpose(-1, -2) # [bs, T-1, d_phi, n]
            mixing_n = mixing_n[:, :, :-1].transpose(-1, -2).unsqueeze(-2)
            phi_bar = (phi_tilde * mixing_n).sum(-1)
            phi = phi_bar
        elif mode == 'online':
            phi_tilde = self.mac.phi_forward(obs, task).transpose(-1, -2) # [bs, T-1, d_phi, n]
            mixing_n = mixing_n[:, :, :-1].transpose(-1, -2).unsqueeze(-2)
            phi_bar = (phi_tilde * mixing_n).sum(-1)
            phi_on = rewards * mixing_w.unsqueeze(1)
            rew_alpha = 1 - np.exp(- self.main_args.rew_beta * t_env / self.main_args.rew_step)
            phi = rew_alpha * phi_on + (1-rew_alpha) * phi_bar
            
        targets = phi + self.main_args.gamma * (1 - terminated) * target_max_qvals
        
        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
        
        td_error = (chosen_action_qvals - targets.detach())
        
        # 0-out the targets that came from padded data
        td_mask = mask.expand_as(td_error)
        masked_td_error = td_error * td_mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()

        loss = loss + td_loss
        
        mode_cql = ['offline']
        if mode in mode_cql:
            if self.main_args.cql_type == 'base':
                # CQL-error
                cql_error = th.logsumexp(mac_out[:, :-1], dim=-1) - chosen_action_na_qvals
                cql_mask = mask.unsqueeze(-2).expand_as(cql_error)
                cql_loss = (cql_error * cql_mask).sum() / cql_mask.sum()
                
                loss = loss + self.main_args.cql_alpha * cql_loss

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
            if mode in mode_cql:
                self.logger.log_stat(f"{task}/cql_loss", cql_loss.item(), t_env)
            if mode == "online":
                # self.logger.log_stat(f"{task}/r_loss", r_loss.item(), t_env)
                # self.logger.log_stat(f"{task}/w_reg", w_range_reg.item(), t_env)
                self.logger.log_stat(f"{task}/w_rec_std", np.mean(np.sqrt(w_var)), t_env)
            
            chosen_qs_global = ((1 / (mixing_w * self.phi_dim)).view(bs, 1, self.phi_dim) * chosen_action_qvals).sum(-1)
            q_mask = mask.squeeze(-1)
            self.logger.log_stat(f"{task}/esti_q_mean", (chosen_qs_global * q_mask).sum().item() / q_mask.sum().item(), t_env)
            
            true_q = (compute_q_values(rewards, self.main_args.gamma) * mask).sum().item() / mask.sum().item()
            if self.ema_true_q is None:
                self.ema_true_q = true_q
            self.ema_true_q = self.ema_true_q * 0.9 + true_q * 0.1
            self.logger.log_stat(f"{task}/true_Q", self.ema_true_q, t_env)
            
            self.task2train_info[task]["log_stats_t"] = t_env
        
    def adapt(self, batch, t_env: int, episode_num: int, task: str):
        ''' mode = | adapt | '''
        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        states = batch["state"] # (bs, T-1, ds)
        state_mask = batch["filled"]
        obs = batch["obs"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 1. phi
        mixing_w, mixing_n = self.mac.explain_task(task, states, state_mask)
        phi_tilde = self.mac.phi_forward(obs, task).transpose(-1, -2) # [bs, T-1, d_phi, n]
        mixing_n = mixing_n[:, :, :-1].transpose(-1, -2).unsqueeze(-2)
        phi_bar = (phi_tilde * mixing_n).sum(-1) # [bs, T-1, d_phi]
        
        # 2. solve and store w
        w_hat = (rewards * phi_bar).mean(1).abs() / (rewards.square().mean(1) + 1e-5)
        w_hat_eff = w_hat[~(rewards == 0).all(1).squeeze(-1)] # exclude r=0 trajs
        w_mean, w_var = self.mac.update_weight(w_hat_eff, task)
        # w_mean, w_var = self.mac.update_weight(mixing_w, task)
        
        rloss = ((phi_bar - rewards * mixing_w.unsqueeze(1)).square() * mask).mean()
        
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/rloss", rloss.item(), t_env)
            self.logger.log_stat(f"{task}/w_std", np.mean(np.sqrt(w_var)), t_env)
            
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
        # if self.main_args.train_mode in ['offline', 'online']:
        #     th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        self.mac.save_models(path)
        # if self.main_args.standardise_rewards:
        #     th.save(self.task2rew_ms, f"{path}/rew_ms.th")
            
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.target_mixer.state_dict(), "{}/target_mixer.th".format(path))
        
    def load_models(self, path):
        # if self.main_args.train_mode in ['online']:
        #     self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        # if self.main_args.standardise_rewards:
        #     self.task2rew_ms = th.load(f"{path}/rew_ms.th")
        #     for task in self.task2args.keys():
        #         if task not in self.task2rew_ms.keys():
        #             self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=self.main_args.device)
        if self.main_args.train_mode in ['online', 'adapt']:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
                self.target_mixer.load_state_dict(th.load("{}/target_mixer.th".format(path), map_location=lambda storage, loc: storage))
            