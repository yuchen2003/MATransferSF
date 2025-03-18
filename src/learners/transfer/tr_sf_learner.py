import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.transfer.tr_vdn import MTVDNMixer
from modules.mixers.transfer.tr_qatten import MTQattnMixer
from modules.mixers.qmix import QMixer
from modules.mixers.transfer.tr_dmaq_qatten import MTDMAQQattnMixer, Extractor
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam, SGD
from components.standarize_stream import RunningMeanStd, RunningMean, EMAMean
import torch.nn.functional as F
import numpy as np
from utils.calc import compute_q_values

class TransferSFLearner:
    def __init__(self, mac, logger, main_args, train_mode) -> None:
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.mac.agent.set_train_mode(train_mode)
        self.params = list(mac.parameters())
        # self.offline_train_steps = main_args.offline_train_steps
        # self.online_train_steps = main_args.t_max

        # task weight regression
        self.phi_dim = main_args.rnn_hidden_dim
        self.phi_cov = th.zeros(self.phi_dim, self.phi_dim, dtype=th.float32, device=main_args.device)
        self.phi_cov_ema_beta = 0.99
        self.last_weight = None

        # loss weights
        self.lambda_r = 1
        self.lambda_l1 = 1 # not sensitive

        self.mixer = None
        if main_args.mixer is not None:
            match main_args.mixer:
                case "tr_vdn":
                    self.mixer = MTVDNMixer(self.surrogate_decomposer, main_args)
                case "tr_qatten":
                    self.mixer = MTQattnMixer(self.surrogate_decomposer, main_args)
                case "tr_dmaq_qatten":
                    self.mixer = MTDMAQQattnMixer(self.surrogate_decomposer, main_args)
                case "qmix":
                    self.mixer = QMixer(main_args)
                case _:
                    raise ValueError("Mixer {} not recognised.".format(main_args.mixer))
        assert self.mixer is not None
        
        self.phi_residual = Extractor(main_args.entity_embed_dim, self.phi_dim, main_args.entity_embed_dim, 2)
        
        if train_mode in ['offline', 'online']:
            self.params += list(self.mixer.parameters())
            self.params += list(self.phi_residual.parameters())
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
            case "sgd":
                self.optimiser = SGD(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
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
            self.task2phi_mean = {}
            self.task2aw_rew_ms_list = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device)
                self.task2phi_mean[task] = RunningMean(shape=(self.phi_dim, ), device=device)
                

    def pretrain(self, batch, t_env: int, episode_num: int, task: str):
        obs = batch["obs"][:, :-1]
        next_obs = batch["obs"][:, 1:]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        action_pred = self.mac.pretrain_forward(obs, next_obs, task)
        mask = mask.unsqueeze(-1).repeat(1, 1, *actions_onehot.shape[-2:])
        loss = F.binary_cross_entropy(action_pred * mask, actions_onehot * mask)

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip, error_if_nonfinite=True)
        self.optimiser.step()
        self.total_training_steps += 1
        self.task2train_info[task]["training_steps"] += 1

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            self.task2train_info[task]["log_stats_t"] = t_env
            if grad_norm.item() == 0:
                print("ERROR...")

    def _w_solver(self, phi, rewards):
        # phi: [bs, T, d_phi], rewards: [bs, T, 1]
        # phi = phi.reshape(-1, self.phi_dim).unsqueeze(-1) # (bsT, d_phi, 1)
        # phi_cov_cur = th.bmm(phi, phi.transpose(1, 2)) # (bsT, d_phi, d_phi)
        # self.phi_cov = self.phi_cov * self.phi_cov_ema_beta + phi_cov_cur.mean(0) * (1. - self.phi_cov_ema_beta)
        # CHECK 对比batch求解lstsq问题与估算cov(进一步考虑间隔更新)两种方式

        phi = phi.reshape(-1, self.phi_dim) # (bsT, d_phi)
        rewards = rewards.reshape(-1, 1) # (bsT, 1)
        result = th.linalg.lstsq(phi, rewards)
        print("w solver residual {:.4f}".format(result.residuals.item() / phi.size(0)))
        w = result.solution.squeeze(1)
        return w

    def train(self, batch, t_env: int, episode_num: int, task: str, mode: str):
        ''' mode = | offline | online | adapt | '''
        bs = batch.batch_size
        n_agents = self.task2args[task].n_agents
        rewards = batch["reward"][:, :-1] # (bs, T-1, *)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values -> psi-values, Q-values
        phi_out, psi_out = [], []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            phi, psi = self.mac.forward(batch, t=t, task=task)
            phi_out.append(phi)
            psi_out.append(psi)
        phi_out = th.stack(phi_out, dim=1)[:, :-1] # (bs, T-1, n, d_phi)
        psi_out = th.stack(psi_out, dim=1) # (bs, T, n, n_act, d_phi) 

        phi_mix_value, state_embed = self.target_mixer(
            phi_out,
            batch["state"][:, :-1],
            actions_onehot,
            None,
            self.task2decomposer[task],
            is_phi=True,
        ) # (bs, T-1, d_phi)
        # phi_tilde = phi_mix_value.detach() + self.phi_residual(state_embed.detach())
        # weight = self._w_solver(phi_tilde, rewards).detach()
        # reward_hat = (phi_tilde * weight).sum(-1)
        # r_loss = (reward_hat - rewards).square().mean() 
        
        if self.main_args.standardise_rewards:
            self.task2rew_ms[task].update(rewards)
            self.task2phi_mean[task].update(phi_mix_value.detach())
            r_mean, r_std, phi_mean = self.task2rew_ms[task].mean, th.sqrt(self.task2rew_ms[task].var), self.task2phi_mean[task].mean
            rewards = (rewards - r_mean) / r_std
            phi_mix_value = (phi_mix_value - phi_mean) / r_std
        weight = self._w_solver(phi_mix_value, rewards).detach()
        self.mac.task2weight_ms[task].update(weight)
        reward_hat = (phi_mix_value * weight).sum(-1, keepdim=True) # (bs, T-1, 1)
        
        print("reward approx error (MAE): {:.4f}".format((reward_hat - rewards).abs().mean().item())) 
        
        mac_out = (psi_out * weight).sum(-1) # (bs, T, n, n_act)
        mac_out[avail_actions == 0] = -9999999
        max_action_qvals = mac_out[:, :-1].max(dim=3)[0]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_na_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) # (bs, T-1, n)

        # Calculate the Q-Values necessary for the target
        target_psi_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            _, target_psi = self.target_mac.forward(batch, t=t, task=task)
            target_psi_out.append(target_psi)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_psi_out = th.stack(target_psi_out[1:], dim=1) # (bs, T-1, n, n_act, d_phi)
        target_mac_out = (target_psi_out * weight).sum(-1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_max_action_qvals = target_mac_out.max(dim=3)[0]

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            # mac_out_detach[avail_actions == 0] = -9999999 already done before
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_na_qvals = th.gather(target_mac_out, 3, cur_max_actions[:, 1:]).squeeze(3)
            # cons_max_qvals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_na_qvals = target_mac_out.max(dim=3)[0]

        # NOTE these two mixing can exchange, equivalent if with linear agent mixing
        # Agent mixing: (bs, T-1, n) -> (bs, T-1, 1)
        chosen_action_qvals, regs, ents = self.mixer(
            chosen_action_na_qvals,
            batch["state"][:, :-1],
            actions_onehot,
            max_action_qvals,
            self.task2decomposer[task],
        )
        target_max_qvals, _, _ = self.target_mixer(
            target_max_na_qvals,
            batch["state"][:, 1:],
            actions_onehot,
            target_max_action_qvals,
            self.task2decomposer[task],
        )

        if self.main_args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.task2ret_ms[task].var) + self.task2ret_ms[task].mean

        # Calculate 1-step Q-Learning targets
        # CHECK use rewards or reward_hat(Eq.6 in ZSRL...) ?
        targets = reward_hat + self.main_args.gamma * (1 - terminated) * target_max_qvals # (bs, T-1, 1)
        # targets = target_max_qvals

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

        loss = td_loss + regs

        # CQL: TO handle ood in offRL; may be replaced with qplex
        if mode == 'offline':
            if self.main_args.cql_type == 'base':
                # CQL-error
                cql_error = th.logsumexp(mac_out[:, :-1], dim=3) - chosen_action_na_qvals
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

        # mask_elems = mask.sum().item()
        # true_q = (compute_q_values(rewards, self.main_args.gamma) * mask).sum().item()/(mask_elems * n_agents)
        # if self.ema_true_q is None:
        #     self.ema_true_q = true_q
        # self.ema_true_q = self.ema_true_q * 0.9 + true_q * 0.1
        # print("{:.4f}, {:.4f}".format(true_q, self.ema_true_q))

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
            self.logger.log_stat(f"{task}/td_loss", td_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/psi_td_loss", psi_td_loss.item(), t_env)
            mask_elems = mask.sum().item()
            # self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            if mode == 'offline':
                self.logger.log_stat(f"{task}/cql_loss", cql_loss.item(), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat(f"{task}/phi_mean", phi_out.mean().item(), t_env)
            self.logger.log_stat(f"{task}/phi_std", phi_out.std().item(), t_env)
            # self.logger.log_stat(f"{task}/weight_mean", weight.mean().item(), t_env) # insignificant if use softmax head
            self.logger.log_stat(f"{task}/weight_std", weight.std().item(), t_env)
            # true Q-value
            true_q = (compute_q_values(rewards, self.main_args.gamma) * mask).sum().item()/(mask_elems * n_agents)
            if self.ema_true_q is None:
                self.ema_true_q = true_q
            self.ema_true_q = self.ema_true_q * 0.9 + true_q * 0.1
            self.logger.log_stat(f"{task}/true_Q", self.ema_true_q, t_env)

            if self.last_weight is None:
                self.last_weight = weight
            self.logger.log_stat(f"{task}/weight_tvd", .5 * (weight - self.last_weight).abs().sum().item(), t_env)
            self.last_weight = weight

            self.task2train_info[task]["log_stats_t"] = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.console_logger.info("Updated target network")

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
        # FIXME disable  this when continue training with a different optimizer
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mac.load_models(path)
        # FIXME Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
