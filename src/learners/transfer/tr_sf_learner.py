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

        self.params = list(mac.parameters()) # TODO pretrain enc-dec need separate optim; offline train need one; online train need one
        self.pretrain_steps = main_args.pretrain_steps
        self.pretrain_batch_size = main_args.pretrain_batch_size
        self.pretrain_last_log_t = 0
        self.offline_train_steps = main_args.offline_train_steps
        self.online_train_steps = main_args.t_max
        self.phi_dim = main_args.phi_dim
        
        # loss weights
        self.lambda_recon = 0.1
        self.lambda_r = 1
        self.lambda_l1 = 1

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
        
        # separate reward shaping auxiliary function, just used for pretraining
        # self.r_shaping = nn.Sequential(nn.Linear(main_args.rnn_hidden_dim, 512), nn.LeakyReLU(),
        #                                nn.Linear(main_args.rnn_hidden_dim, 1))
        # self.optim_rshap = Adam(list(self.r_shaping.parameters()), lr=self.main_args.lr)
        
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
            self.task2ret_ms = {}
            for task in self.task2args.keys():
                self.task2ret_ms[task] = RunningMeanStd(shape=(self.task2n_agents[task], ), device=device)

        if self.main_args.standardise_rewards:
            self.task2rew_ms = {}
            self.task2aw_rew_ms_list = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device)
                
    def train(self, batch, t_env: int, episode_num: int, task: str):
        # FIXME DEPRECATED
        
        # TODO pretrain encoder
        # TODO Offline train
        # TD learn with reward => encoder train + sparse(min L1) weight train
        # encoder-decoder recon
        # subgroup-local encoder inference
        # TODO Online transfer
        # try to learn based on local decoders, i.e., (multi-task) policies, 
        # with value estimation|guidance from SF, weight generator and #!lasso retifier
        pass
    
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
        # avail_actions = batch["avail_actions"]
        
        self.w = th.randn(bs, self.phi_dim, requires_grad=True, device=th.device(batch.device)) # FIXME bs for one task ? or 1 for one task?
        self.optim_w = SGD([self.w], lr=0.2)
        self.num_ep_w = 50 # TODO lr=0.1

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
            
        loss, phi_loss, recon_loss, KLD = 0, 0, 0, 0 
        r = rewards.squeeze(-1) # (bs, T-1)
        a = actions_onehot.transpose(1, 2).reshape(bs * n_agents, -1, actions_onehot.shape[-1]) # (bs, T-1, n_agents, n_act) -> (bs, n, T-1, n_act) -> (bsn, T-1, n_act)
        
        phi_detach = phi.detach() # (bsn, T-1, d_phi)
        r_shap_detach = r_shap.detach()
        for ep_w in range(self.num_ep_w):
            w_repeat = self.w.unsqueeze(1).repeat(1, n_agents, 1) # (bs, d_phi) -> (bs, n, d_phi)
            w_repeat = w_repeat.view(-1, self.phi_dim).unsqueeze(1).repeat(1, T_1, 1) # (bsn, T-1, d_phi)
            r_hat = th.sum(phi_detach * w_repeat, dim=-1) + r_shap_detach # (bsn, T-1)
            r_hat = r_hat.view(bs, n_agents, -1).sum(dim=1) # (bsn, T-1) -> (bs, n, T-1) -> (bs, T-1)
            r_loss = th.mean(th.square(r_hat - r))
            w_reg = self.lambda_l1 * th.mean(th.sum(th.abs(self.w), dim=-1))
            w_loss = r_loss + self.lambda_l1 * w_reg
            
            self.optim_w.zero_grad()
            w_loss.backward()
            self.optim_w.step()
            
            # if ep_w % 10 == 0:
            #     print(f"Epoch {ep_w}, reward pred Loss: {r_loss.item()}")
                
        w_repeat_detach = self.w.detach().unsqueeze(1).repeat(1, n_agents, 1)
        w_repeat_detach = w_repeat_detach.view(-1, self.phi_dim).unsqueeze(1).repeat(1, T_1, 1) # (bsn, T-1, d_phi)
        r_hat_out = th.sum(phi * w_repeat_detach, dim=-1) + r_shap # (bsn, T-1)
        r_hat_out = r_hat_out.view(bs, n_agents, -1).sum(dim=1) # (bsn, T-1) -> (bs, n, T-1) -> (bs, T-1)
        phi_loss += th.mean(th.square(r_hat_out - r))
        
        recon_loss += th.mean(th.square(action_recon - a))
        KLD += -.5 * th.mean(1 + logvar - mu.pow(2) - logvar.exp())

        self.pretrain_optimiser.zero_grad()
        loss = phi_loss + self.lambda_recon * (recon_loss + KLD)
        loss.backward()
        self.pretrain_optimiser.step()
        
        if t_env - self.pretrain_last_log_t >= 50:
            self.logger.log_stat(f"loss", loss.item(), t_env)
            self.logger.log_stat(f"recon_loss", recon_loss.item(), t_env)
            self.logger.log_stat(f"phi_loss", phi_loss.item(), t_env)
            self.logger.log_stat(f"r_loss", r_loss.item(), t_env)
            self.logger.log_stat(f"KLD", KLD.item(), t_env)
            self.pretrain_last_log_t = t_env
            print(f"step: {t_env}, loss: {loss.item()}, recon: {recon_loss.item()}, phi: {phi_loss.item()}, w: {r_loss.item()}, KLD: {KLD.item()}")
    
    def offline_train(self, batch, t_env: int, episode_num: int, task: str):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
            
        if self.main_args.standardise_rewards:
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
            
        # TODO construct SF and calc psi, Q; learn with TD error
        
        # Calculate estimated Q-Values -> psi-values, Q-values
        mac_out = []
        r_out, w_out, psi_out, phi_tilde_out, phi_hat_out, recon_action_out = [], [], [], [], [], []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs, r_hat, w, psi_tilde, phi_tilde, phi_hat, recon_action = self.mac.forward(batch, t=t, task=task)
            mac_out.append(agent_outs)
            r_out.append(r_hat)
            w_out.append(w)
            psi_out.append(psi_tilde)
            phi_tilde_out.append(phi_tilde)
            phi_hat_out.append(phi_hat)
            recon_action_out.append(recon_action)
        # (bs, seq_len, n_agents, n_action | 1 | d_phi (x4) | n_actions)
        mac_out = th.stack(mac_out, dim=1) 
        r_out = th.stack(r_out, dim=1)
        w_out = th.stack(w_out, dim=1)
        psi_out = th.stack(psi_out, dim=1)
        phi_tilde_out = th.stack(phi_tilde_out, dim=1)
        phi_hat_out = th.stack(phi_hat_out, dim=1)
        recon_action_out = th.stack(recon_action_out, dim=1)
        
        # Calc losses
        loss = 0
        ## 1. reconstruction loss #FIXME done in pretrain
        loss += .5 * self.lambda_recon * th.mean(th.square(recon_action_out - actions))
        
        ## 2. reward approximation loss #FIXME done in pretrain
        loss += .5 * self.lambda_r * th.mean(th.square(r_out - rewards)) + self.lambda_l1 * th.mean(th.abs(w))
        
        ## 3. subgroup-local sf alignment
        loss += .5 * self.lambda_align * th.mean(th.square(phi_tilde_out - phi_hat_out))
        
        ## 4. TD error for psi

        ## 5. TD error for Q
        mac_out[avail_actions == 0] = -9999999

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_na_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim, (bs, seq_len, n_agents)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t, task=task)
            target_mac_out.append(target_agent_outs)
        
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            #mac_out_detach[avail_actions == 0] = -9999999 already done before
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_na_qvals = th.gather(target_mac_out, 3, cur_max_actions[:, 1:]).squeeze(3)
            cons_max_qvals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_na_qvals = target_mac_out.max(dim=3)[0]
        
        if self.mixer is not None: # TODO 结合QPLEX，QAttn等搞清楚mixing过程
            chosen_action_qvals = self.mixer(chosen_action_na_qvals, batch["state"][:, :-1], self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_na_qvals, batch["state"][:, 1:], self.task2decomposer[task])
            cons_max_q_vals = self.mixer(cons_max_qvals, batch["state"], self.task2decomposer[task])
        

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.main_args.gamma * (1 - terminated) * target_max_qvals

        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        
        loss = td_loss


        if "cql" in self.main_args.name:
            match self.main_args.cql_type:
                case "base":
                    # CQL-error
                    assert (th.logsumexp(mac_out[:, :-1], dim=3).shape == chosen_action_na_qvals.shape)
                    cql_error = th.logsumexp(mac_out[:, :-1], dim=3) - chosen_action_na_qvals
                    cql_mask = mask.expand_as(cql_error)
                    cql_loss = (cql_error * cql_mask).sum() / mask.sum() # better use mask.sum() instead of cql_mask.sum()
                case "odis":
                    assert cons_max_q_vals[:, :-1].shape == chosen_action_qvals.shape
                    cql_error = cons_max_q_vals[:, :-1] - chosen_action_qvals
                    cql_loss = (cql_error * mask).sum() / mask.sum()
                case _:
                    raise ValueError("Unknown cql_type: {}".format(self.main_args.cql_type))
            loss += self.main_args.cql_alpha * cql_loss
        
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
            if "cql" in self.main_args.name:
                self.logger.log_stat(f"{task}/cql_loss", cql_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean", (targets * mask).sum().item()/(mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env
    
    def online_train(self, batch, t_env: int, episode_num: int, task: str):
        pass

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
        match stage:
            case 0: 
                th.save(self.pretrain_optimiser.state_dict(), "{}/pt_opt.th".format(path))
            case 1: 
                th.save(self.offline_optimiser.state_dict(), "{}/off_opt.th".format(path))
            case 2: 
                th.save(self.online_optimiser.state_dict(), "{}/on_opt.th".format(path))
            case _: raise ValueError
        self.mac.save_models(path) # each model is saved up to its training stage
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

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
            