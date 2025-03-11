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
from utils.calc import compute_q_values

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
        self.num_ep_w = 100
        self.lr_w = 0.01
        self.offline_train_steps = main_args.offline_train_steps
        self.online_train_steps = main_args.t_max
        self.phi_dim = main_args.rnn_hidden_dim
        
        # loss weights
        self.lambda_r = 1
        self.lambda_l1 = 1 # not sensitive

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
                self.optimiser = RMSprop(
                    params=self.params,
                    lr=self.main_args.lr,
                    alpha=self.main_args.optim_alpha,
                    eps=self.main_args.optim_eps,
                    weight_decay=self.main_args.weight_decay,
                )
            case "adam":
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
        self.ema_true_q = None

        device = "cuda" if main_args.use_cuda else "cpu"
        if self.main_args.standardise_returns:
            self.task2ret_ms, self.task2psi_ms = {}, {}
            for task in self.task2args.keys():
                self.task2ret_ms[task] = RunningMeanStd(shape=(self.task2n_agents[task], ), device=device)

        if self.main_args.standardise_rewards:
            self.task2rew_ms = {}
            self.task2aw_rew_ms_list = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device)
    
    def pretrain(self, batch, t_env: int, episode_num: int, task: str):
        pass
    
    def train(self, batch, t_env: int, episode_num: int, task: str, mode: str):
        bs = batch.batch_size
        n_agents = self.task2args[task].n_agents
        rewards = batch["reward"][:, :-1] # (bs, T-1, *)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # (bs, T, n, 1) -> (bs, T, n, d_phi, 1) ~ psi_out[:, :, :, :, *]
        expand_actions = actions.unsqueeze(3).repeat(1, 1, 1, self.phi_dim, 1)
        # (bs, T, n, n_act) -> (bs, T, n, d_phi, n_act) ~ psi_out
        expand_avail_actions = avail_actions.unsqueeze(3).repeat(1, 1, 1, self.phi_dim, 1)
        # (bs, T, 1) -> (bs, T, 1, d_phi)
        expand_terminated = terminated.unsqueeze(3).repeat(1, 1, 1, self.phi_dim)

        if self.main_args.standardise_rewards:
            # NOTE phi is calculated by the encoder
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
        
        # Calculate estimated Q-Values -> psi-values, Q-values
        phi_out, psi_out = [], []
        self.mac.init_hidden(batch.batch_size, task)
        agent_loss = 0
        loss_info = {}
        flag = 0
        for t in range(batch.max_seq_length):
            phi, psi, a_loss, a_loss_info = self.mac.forward(batch, t=t, task=task)
            phi_out.append(phi)
            psi_out.append(psi)
            agent_loss += a_loss
            if flag == 0:
                flag = 1
                loss_info.update(a_loss_info)
            else:
                for k, v in a_loss_info.items():
                    loss_info[k] += v
        phi_out = th.stack(phi_out, dim=1)[:, :-1] # (bs, T-1, n, d_phi)
        psi_out = th.stack(psi_out, dim=1) # (bs, T, n, d_phi, n_act) 
        agent_loss /= bs
        for k in a_loss_info.keys():
            loss_info[k] /= bs

        psi_out[expand_avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        # (bs, T, n, d_phi)
        chosen_action_psi_na_vals = th.gather(psi_out[:, :-1], dim=-1, index=expand_actions).squeeze(-1)

        # Calculate the Q-Values necessary for the target
        target_psi_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            _, target_psi, _, _ = self.target_mac.forward(batch, t=t, task=task)
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

        # NOTE these two mixing can exchange, equivalent if with linear agent mixing
        # Agent mixing: 1.(bs, T-1, n, d_phi) -> (bs, T-1, 1, d_phi)
        if self.mixer is not None:
            chosen_action_psi_vals = self.mixer(chosen_action_psi_na_vals, batch["state"][:, :-1], self.task2decomposer[task])
            target_max_psi_vals = self.target_mixer(phi_out + self.main_args.gamma * (1 - expand_terminated) * target_max_na_psi_vals, 
                                                    batch["state"][:, 1:], self.task2decomposer[task])


        # Subtask mixing: 2.(bs, T-1, 1, d_phi) -> (bs, T-1, 1)
        # weight = self._w_rectify(phi_hat.detach(), rewards, mask, task, t_env, is_online)
        weight = self.mac.explain_task(task)
        
        chosen_action_qvals = (chosen_action_psi_vals * weight).sum(-1)
        target_max_qvals = (target_max_psi_vals * weight).sum(-1)
        
        if self.main_args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.task2ret_ms[task].var) + self.task2ret_ms[task].mean
        
        # Calculate 1-step Q-Learning targets
        # targets = rewards + self.main_args.gamma * (1 - terminated) * target_max_qvals #(bs, T-1, 1)
        targets = target_max_qvals

        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data (1, 2.1)
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_loss + agent_loss
        
        # CQL: TO handle ood in offRL; may be replaced with qplex
        if not is_online:
            if self.main_args.cql_type == 'base':
                agent_qs = (psi_out.transpose(-1, -2) * weight).sum(-1) # (bs, T, n, n_act) 
                chosen_action_na_qvals = (chosen_action_psi_na_vals * weight).sum(-1)
                # CQL-error
                cql_error = th.logsumexp(agent_qs[:, :-1], dim=3) - chosen_action_na_qvals
                cql_mask = mask.expand_as(cql_error)
                cql_loss = (cql_error * cql_mask).sum() / mask.sum() # better use mask.sum() instead of cql_mask.sum()
            else:
                raise ValueError("Unknown cql_type: {}".format(self.main_args.cql_type))
            loss += self.main_args.cql_alpha * cql_loss
        
        # Optimise
        if is_online:
            task_id = self.mac.task2ids[task]
            optim_id = Adam([task_id], lr=self.lr_w)
            optim_id.zero_grad()
            loss.backward()
            optim_id.step()
            grad_norm = 0
        else:
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
            # self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            if not is_online:
                self.logger.log_stat(f"{task}/cql_loss", cql_loss.item(), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat(f"{task}/phi_mean", phi_out.mean().item(), t_env)
            self.logger.log_stat(f"{task}/phi_std", phi_out.std().item(), t_env)
            self.logger.log_stat(f"{task}/weight_mean", weight.mean().item(), t_env)
            self.logger.log_stat(f"{task}/weight_std", weight.std().item(), t_env)
            # true Q-value
            true_q = (compute_q_values(rewards, self.main_args.gamma) * mask).sum().item()/(mask_elems * n_agents)
            if self.ema_true_q is None:
                self.ema_true_q = true_q
            self.ema_true_q = self.ema_true_q * 0.9 + true_q * 0.1
            self.logger.log_stat(f"{task}/true_Q", self.ema_true_q, t_env)
            
            for k, v in loss_info.items():
                self.logger.log_stat(f"{task}/{k}", v, t_env)
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
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        
    def load_models(self, path):
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            