import os, sys, time
import argparse

from training_utils import TrainingProgress, timeSince, load_conf, check_path
from agent import DDPGfDAgent, DATA_RUNTIME, DATA_DEMO
from mjrl.utils.gym_env import GymEnv
import mjrl.envs
import mj_envs
import mj_allegro_envs
import torch, joblib
import torch.nn as nn
import numpy as np
from logger import logger_setup
import logging
from os.path import join as opj
import gym
import wandb

np.set_printoptions(suppress=True, precision=4)

# Used loggers
DEBUG_LLV = 5  # for masked
loggers = ['RLTrainer', 'DDPGfD', 'TP']
# logging.addLevelName(DEBUG_LLV, 'DEBUGLLV')  # Lower level debugging info
logging_level = logging.DEBUG  # logging.DEBUG

# wandb.init(project='dex_manip_ddpgfd')

def fetch_obs(obs):
    return np.r_[obs['observation'], obs['achieved_goal'], obs['desired_goal']]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class RLTrainer:
    def __init__(self, conf_path, eval=False):
        self.full_conf = load_conf(conf_path)
        self.conf = self.full_conf.train_config
        
        wandb.init(project='dex_manip_ddpgfd', config=self.full_conf)
        wandb.config.timestr = timestr

        # progress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')
        # result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        # Store in xxxx_dir/exp_name+exp_idx/...
        self.tp = TrainingProgress(progress_dir, result_dir, self.conf.exp_name)

        logger_setup(os.path.join(self.tp.result_path, self.conf.exp_name + '-log.txt'), loggers, logging_level)
        self.logger = logging.getLogger('RLTrainer')

        if torch.cuda.is_available():
            torch.cuda.set_device(self.conf.device)  # default 0
            # cudnn.benchmark = True # Faster only for fixed runtime size
            self.logger.info('Use CUDA Device ' + self.conf.device)
        
        self.device = self.conf.device

        if self.conf.seed == -1:
            self.conf.seed = os.getpid() + int.from_bytes(os.urandom(4), byteorder="little") >> 1
            self.logger.info('Random Seed={}'.format(self.conf.seed))
        # Random seed
        torch.manual_seed(self.conf.seed)  # cpu
        np.random.seed(self.conf.seed)  # numpy

        # Backup environment config
        if not eval:
            self.tp.backup_file(conf_path, 'training.yaml')

        # Construct Env
        # self.env = gym.make('FetchReach-v1')
        self.env = GymEnv('block-v3')
        self.logger.info('Environment Loaded')

        self.agent = DDPGfDAgent(self.full_conf.agent_config, self.device)
        self.agent.to(self.device)

        if self.conf.restore:
            self.restore_progress(eval)
        else:
            self.episode = 1
        self.set_optimizer()
        # Loss Function setting
        reduction = 'none'

        if self.conf.mse_loss:
            self.q_criterion = nn.MSELoss(reduction=reduction)
        else:
            self.q_criterion = nn.SmoothL1Loss(reduction=reduction)
        self.demo2memory_block()
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.full_conf.agent_config.action_dim),
                                                             self.full_conf.agent_config.action_noise_std)

    def restore_progress(self, eval=False):
        self.tp.restore_progress(self.conf.tps)  # tps only for restore process from conf
        self.agent.actor_b.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_b'))
        self.agent.actor_t.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='actor_t'))
        self.agent.critic_b.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_b'))
        self.agent.critic_t.load_state_dict(
            self.tp.restore_model_weight(self.conf.tps, self.device, prefix='critic_t'))

        self.episode = self.tp.get_meta('saved_episode') + 1
        np.random.set_state(self.tp.get_meta('np_random_state'))
        torch.random.set_rng_state(self.tp.get_meta('torch_random_state'))
        self.logger.info('Restore Progress,Episode={}'.format(self.episode))

    def summary(self):
        # call Test/Evaluation here
        self.tp.add_meta(
            {'saved_episode': self.episode, 'np_random_state': np.random.get_state(),
             'torch_random_state': torch.random.get_rng_state()})  # , 'validation_loss': self.valid_loss})
        self.save_progress(display=True)

    def save_progress(self, display=False):
        self.tp.save_model_weight(self.agent.actor_b, self.episode, prefix='actor_b')
        self.tp.save_model_weight(self.agent.actor_t, self.episode, prefix='actor_t')
        self.tp.save_model_weight(self.agent.critic_b, self.episode, prefix='critic_b')
        self.tp.save_model_weight(self.agent.critic_t, self.episode, prefix='critic_t')

        self.tp.save_progress(self.episode)
        self.tp.save_conf(self.conf.to_dict())
        if display:
            self.logger.info('Config name ' + self.conf.exp_name)
            self.logger.info('Progress Saved, current episode={}'.format(self.episode))
        
        wandb.config.exp_name = self.conf.exp_name

    def set_optimizer(self):
        # self.optimizer = getattr(optim, self.conf.optim)(
        #     filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.conf.lr_rate,
        #     weight_decay=self.conf.w_decay)  # default Adam

        self.optimizer_actor = torch.optim.Adam(self.agent.actor_b.parameters(), lr=self.conf.lr_rate,
                                                weight_decay=self.conf.w_decay)
        self.optimizer_critic = torch.optim.Adam(self.agent.critic_b.parameters(), lr=self.conf.lr_rate,
                                                 weight_decay=self.conf.w_decay)

    def demo2memory(self):
        dconf = self.full_conf.demo_config
        if dconf.load_demo_data:
            for f_idx in range(dconf.load_N):
                self.agent.episode_reset()
                fname = opj(dconf.demo_dir, dconf.prefix + str(f_idx) + '.pkl')
                data = joblib.load(fname)
                for exp in data:
                    s, a, r, s2, done = exp
                    s_tensor = torch.from_numpy(s).float()
                    s2_tensor = torch.from_numpy(s2).float()
                    action = torch.from_numpy(a).float()
                    if not done or self.agent.conf.N_step == 0:
                        self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(), s2_tensor,
                                               torch.tensor([self.agent.conf.gamma]),
                                               DATA_DEMO))  # Add one-step to memory, last step added in pop with done=True

                    # Add new step to N-step and Pop N-step data to memory
                    if self.agent.conf.N_step > 0:
                        self.agent.backup.add_exp(
                            (s_tensor, action, torch.tensor([r]).float(), s2_tensor))  # Push to N-step backup
                        self.agent.add_n_step_experience(DATA_DEMO, done)
            self.logger.info('{}/{} Demo Trajectories Loaded. Total Experience={}'.format(dconf.load_N, dconf.demo_N,
                                                                                          len(self.agent.memory)))
            self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info('No Demo Trajectory Loaded')

    def demo2memory_block(self):
        dconf = self.full_conf.demo_config
        if dconf.load_demo_data:
            for f_idx in range(dconf.load_N):
                self.agent.episode_reset()
                fname = opj(dconf.demo_dir, dconf.prefix + str(f_idx) + '.pkl')
                # fname = '/scratch/ax2030/dex_manip/dapg/dapg/demonstrations/dec-block3.pickle'
                data = joblib.load(fname)
                for exp in data:
                    s, a, r, s2, done = exp
                    # terminated, init_state_dict, env_infos, observations, rewards, actions = joblib.load(fname)
                    s_tensor = torch.from_numpy(s).float()
                    s2_tensor = torch.from_numpy(s2).float()
                    action = torch.from_numpy(a).float()
                    if not done or self.agent.conf.N_step == 0:
                        self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(), s2_tensor,
                                               torch.tensor([self.agent.conf.gamma]),
                                               DATA_DEMO))  # Add one-step to memory, last step added in pop with done=True

                    # Add new step to N-step and Pop N-step data to memory
                    if self.agent.conf.N_step > 0:
                        self.agent.backup.add_exp(
                            (s_tensor, action, torch.tensor([r]).float(), s2_tensor))  # Push to N-step backup
                        self.agent.add_n_step_experience(DATA_DEMO, done)
            self.logger.info('{}/{} Demo Trajectories Loaded. Total Experience={}'.format(dconf.load_N, dconf.demo_N,
                                                                                          len(self.agent.memory)))
            self.agent.memory.set_protect_size(len(self.agent.memory))
        else:
            self.logger.info('No Demo Trajectory Loaded')



    def update_agent(self, update_step):  # update_step iteration
        # 2. Sample experience and update
        losses_critic = []
        losses_actor = []
        demo_cnt = []
        batch_sz = 0
        if self.agent.memory.ready():
            for _ in range(update_step):
                (batch_s, batch_a, batch_r, batch_s2, batch_gamma,
                 batch_flags), weights, idxes = self.agent.memory.sample(
                    self.conf.batch_size)
                batch_s, batch_a, batch_r, batch_s2, \
                batch_gamma, weights = batch_s.to(self.device), batch_a.to(self.device), batch_r.to(
                    self.device), batch_s2.to(self.device), batch_gamma.to(self.device), torch.from_numpy(
                    weights.reshape(-1, 1)).float().to(self.device)

                batch_sz += batch_s.shape[0]
                with torch.no_grad():
                    action_tgt = self.agent.actor_t(batch_s)
                    y_tgt = batch_r + batch_gamma * self.agent.critic_t(torch.cat((batch_s, action_tgt), dim=1))

                self.agent.zero_grad()
                # Critic loss
                self.optimizer_critic.zero_grad()
                Q_b = self.agent.critic_b(torch.cat((batch_s, batch_a), dim=1))
                loss_critic = (self.q_criterion(Q_b, y_tgt) * weights).mean()
                # Record Demo count
                d_flags = torch.from_numpy(batch_flags)
                demo_select = d_flags == DATA_DEMO
                N_act = demo_select.sum().item()
                demo_cnt.append(N_act)
                loss_critic.backward()
                self.optimizer_critic.step()

                # Actor loss
                self.optimizer_actor.zero_grad()
                action_b = self.agent.actor_b(batch_s)
                Q_act = self.agent.critic_b(torch.cat((batch_s, action_b), dim=1))
                loss_actor = -torch.mean(Q_act)
                loss_actor.backward()
                self.optimizer_actor.step()

                priority = ((Q_b.detach() - y_tgt).pow(2) + Q_act.detach().pow(
                    2)).cpu().numpy().ravel() + self.agent.conf.const_min_priority
                priority[batch_flags == DATA_DEMO] += self.agent.conf.const_demo_priority
                if not self.agent.conf.no_per:
                    self.agent.memory.update_priorities(idxes, priority)

                losses_actor.append(loss_actor.item())
                losses_critic.append(loss_critic.item())
        if np.sum(demo_cnt) == 0:
            demo_n = 1e-10
        else:
            demo_n = np.sum(demo_cnt)
        return np.sum(losses_critic), np.sum(losses_actor), demo_n, batch_sz

    def pretrain(self):
        assert self.full_conf.demo_config.load_demo_data
        self.agent.train()
        start_time = time.time()
        self.logger.info('Run Pretrain')
        for step in np.arange(self.conf.pretrain_save_step, self.conf.pretrain_step + 1, self.conf.pretrain_save_step):
            losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(self.conf.pretrain_save_step)
            self.logger.info(
                '{}-Pretrain Step {}/{},(Mean):actor_loss={:.8f}, critic_loss={:.8f}, batch_sz={}, Demo_ratio={:.8f}'.format(
                    timeSince(start_time), step, self.conf.pretrain_step, losses_actor / batch_sz,
                                                                          losses_critic / batch_sz, batch_sz,
                                                                          demo_n / batch_sz))
            wandb.log({'Pretrain Step': step, 'Actor loss': losses_actor / batch_sz, 'Critic loss': losses_critic / batch_sz, 'Batch size': batch_sz, 'Demo ratio': demo_n / batch_sz})
                
            self.tp.record_step(step, 'pre_train',
                                {'actor_loss_mean': losses_actor / batch_sz,
                                 'critic_loss_mean': losses_critic / batch_sz,
                                 'batch_sz': batch_sz,
                                 'Demo_ratio': demo_n / batch_sz
                                 }, display=False)
            self.episode = 'pre_{}'.format(step)
            self.summary()
            self.tp.plot_data('pre_train', self.conf.pretrain_save_step, step,
                              'result-pretrain-{}.png'.format(self.episode),
                              self.conf.exp_name + str(self.conf.exp_idx) + '-Pretrain', grid=False,
                              ep_step=self.conf.pretrain_save_step)

        self.episode = 1  # Restore

    def train(self):
        self.agent.train()
        # Define criterion

        start_time = time.time()
        while self.episode <= self.conf.n_episode:  # self.iter start from 1
            # Episodic statistics
            eps_since = time.time()
            eps_reward = eps_length = eps_actor_loss = eps_critic_loss = eps_batch_sz = eps_demo_n = 0
            print('self.env.env.reset():', self.env.env.reset())
            # s0 = fetch_obs(self.env.env.reset())
            # self.env.env.set_target_orien()
            s0 = self.env.env.reset()

            self.agent.episode_reset()
            self.action_noise.reset()

            done = False
            s_tensor = self.agent.obs2tensor(s0)

            while not done:
                # 1. Run environment step
                with torch.no_grad():
                    # s_tensor = self.agent.obs2tensor(state)
                    action_noise = torch.from_numpy(self.action_noise()).float()
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu() + action_noise
                    s2, r, done, _ = self.env.env.step(action.numpy())
                    # s2 = fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    if not done or self.agent.conf.N_step == 0:  # For last step, not duplicate to the last pop from N_step
                        self.agent.memory.add((s_tensor, action, torch.tensor([r]).float(), s2_tensor,
                                               torch.tensor([self.agent.conf.gamma]),
                                               DATA_RUNTIME))  # Add one-step to memory

                # Add new step to N-step and Pop N-step data to memory
                if self.agent.conf.N_step > 0:
                    self.agent.backup.add_exp(
                        (s_tensor, action, torch.tensor([r]).float(), s2_tensor))  # Push to N-step backup
                    self.agent.add_n_step_experience(DATA_RUNTIME, done)  # Pop one

                losses_critic, losses_actor, demo_n, batch_sz = self.update_agent(self.conf.update_step)

                # 3. Record episodic statistics
                eps_reward += r
                eps_length += 1
                eps_actor_loss += losses_actor
                eps_critic_loss += losses_critic
                eps_batch_sz += batch_sz
                eps_demo_n += demo_n

                # Next step
                s_tensor = s2_tensor

            self.logger.info(
                '{}: Episode {}-Last:{}: Actor_loss={:.8f}, Critic_loss={:.8f}, Step={}, Reward={}, Demo_ratio={:.8f}'.format(
                    timeSince(start_time),
                    self.episode,
                    timeSince(eps_since),
                    eps_actor_loss / eps_batch_sz,
                    eps_critic_loss / eps_batch_sz,
                    eps_length, eps_reward, eps_demo_n / eps_batch_sz))

            wandb.log({'Episode': self.episode, 'Actor loss': eps_actor_loss / eps_batch_sz, 'Critic loss': eps_critic_loss / eps_batch_sz, 'Step': eps_length, 'Reward': eps_reward, 'Demo ratio': eps_demo_n / eps_batch_sz})
                
            # Update target
            self.agent.update_target(self.agent.actor_b, self.agent.actor_t, self.episode)
            self.agent.update_target(self.agent.critic_b, self.agent.critic_t, self.episode)

            self.tp.record_step(self.episode, 'episode',
                                {'total_reward': eps_reward, 'length': eps_length,
                                 'avg_reward': eps_reward / eps_length,
                                 'elapsed_time': timeSince(eps_since, return_seconds=True),
                                 'actor_loss_mean': eps_actor_loss / eps_batch_sz,
                                 'critic_loss_mean': eps_critic_loss / eps_batch_sz,
                                 'eps_length': eps_length,
                                 'Demo_ratio': eps_demo_n / eps_batch_sz,
                                 }, display=False)

            if self.episode % self.conf.save_every == 0:
                self.eval()  # Run before summary
                self.summary()
                self.tp.plot_data('episode', 1, self.episode, 'result-train-{}.png'.format(self.episode),
                                  self.conf.exp_name + str(self.conf.exp_idx) + '-Episode', grid=False)

            self.episode += 1

    def eval(self, save_fig=True):
        self.agent.eval()

        all_length = []
        all_reward = []

        # Backup Environment state
        for eps in range(self.conf.eval_episode):  # self.iter start from 1
            # Episodic statistics
            eps_reward = eps_length = 0
            # self.env.env.set_target_orien()
            #s0 = fetch_obs(self.env.env.reset())
            s0 = self.env.env.reset()
            done = False
            s_tensor = self.agent.obs2tensor(s0)
            while not done:
                # 1. Run environment step
                with torch.no_grad():
                    # s_tensor = self.agent.obs2tensor(state)
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu()
                    s2, r, done, _ = self.env.env.step(action.numpy())
                    # s2 = fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    # self.env.render()

                # 3. Record episodic statistics
                eps_reward += r
                eps_length += 1

                # Next step
                s_tensor = s2_tensor
            all_length.append(eps_length)
            all_reward.append(eps_reward)

        self.tp.record_step(self.episode, 'eval', {'Mean Length': np.mean(all_length), 'Std Length': np.std(all_length),
                                                   'Mean Reward': np.mean(all_reward),
                                                   'Std Reward': np.std(all_reward)})
        self.logger.info(
            'Eval Episode-{}: Mean Reward={:.3f}, Mena Length={:.3f}'.format(self.episode, np.mean(all_reward),
                                                                             np.mean(all_length)))

        wandb.log({'Eval episode': self.episode, 'Mean reward': np.mean(all_reward), 'Mean Length': np.mean(all_length)})
            
        if save_fig:
            self.tp.plot_data('eval', self.conf.save_every, self.episode, 'result-eval-{}.png'.format(self.episode),
                              self.conf.exp_name + str(self.conf.exp_idx) + '-Evaluate',
                              self.conf.save_every)  # start from self.conf.save_every
        self.agent.train()

    def collect_block_demo(self, n_collect):
        self.agent.eval()

        demo_record = []  # list of tuple (s,a,r,s',Terminal)
        file_idx = 0
        # check_path(self.full_conf.demo_config.demo_dir)
        fname = '/scratch/ax2030/dex_manip/dapg/dapg/demonstrations/dec-block3.pickle'
        data = joblib.load(fname)
        eps = 0
        print(len(data))

        with torch.no_grad():
            for exp in data: # len(data) = 14
                # s, a, r, s2, done = exp
                terminated, observations, rewards, actions = exp['terminated'], exp['observations'], exp['rewards'], exp['actions'] # len(observations) = 680 (or varying)
                print(len(observations))
                for i in range(1, len(observations)-1):
                    if i != len(observations)-1:
                        terminated = False
                    else:
                        terminated = True

                    s = observations[i-1]
                    s2 = observations[i]
                    demo_record.append((s, actions[i], rewards[i], s2, terminated))

                save_name = opj(self.full_conf.demo_config.demo_dir,
                                            self.full_conf.demo_config.prefix + str(eps) + '.pkl')
                joblib.dump(demo_record, save_name)
                self.logger.info('Terminate: Record {} saved'.format(save_name))
                demo_record = []
                file_idx += 1
                eps += 1

        print('Collect {} Demo'.format(n_collect))

    def collect_demo(self, n_collect):
        self.agent.eval()

        demo_record = []  # list of tuple (s,a,r,s',Terminal)
        file_idx = 0
        check_path(self.full_conf.demo_config.demo_dir)

        with torch.no_grad():
            for eps in range(n_collect):  # self.iter start from 1
                # Episodic statistics
                # self.env.env.set_target_orien()
                s = fetch_obs(self.env.env.reset())
                done = False
                s_tensor = self.agent.obs2tensor(s)
                while not done:
                    # s_tensor = self.agent.obs2tensor(state)
                    action = self.agent.actor_b(s_tensor.to(self.device)[None])[0].cpu().numpy()
                    s2, r, done, _ = self.env.env.step(action)
                    s2 = fetch_obs(s2)
                    s2_tensor = self.agent.obs2tensor(s2)
                    # self.env.render()

                    demo_record.append((s, action, r, s2, done))
                    if done:
                        save_name = opj(self.full_conf.demo_config.demo_dir,
                                        self.full_conf.demo_config.prefix + str(eps) + '.pkl')
                        joblib.dump(demo_record, save_name)
                        self.logger.info('Terminate: Record {} saved'.format(save_name))
                        demo_record = []
                        file_idx += 1

                    # Next step
                    s_tensor = s2_tensor
                    s = s2
        print('Collect {} Demo'.format(n_collect))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', help='Training Configuration', type=str)
    parser.add_argument('--eval', help='Evaluation mode', action='store_true', default=False)
    parser.add_argument('--collect', help='Collect Demonstration Data', action='store_true', default=False)
    parser.add_argument('-n_collect', help='Number of episode for demo collection', type=int, default=100)

    args = parser.parse_args()
    conf_path = args.conf

    trainer = RLTrainer(conf_path, args.eval)
   
    if args.eval:
        trainer.eval(save_fig=False)
    elif args.collect:
        trainer.collect_block_demo(args.n_collect)
    else:
        if trainer.conf.pretrain_demo:
            trainer.pretrain()
        trainer.train()


def analysis():
    # from numba import njit
    import matplotlib.pyplot as plt
    # @njit
    def calc_ewma_reward(reward):
        reward_new = np.zeros(len(reward) + 1)
        reward_new[0] = -50  # Min reward of the env
        ewma_reward = -50  # Min reward of the env
        idx = 1
        for r in reward:
            ewma_reward = 0.05 * r + (1 - 0.05) * ewma_reward
            reward_new[idx] = ewma_reward
            idx += 1
        return reward_new

    from matplotlib import colors as cl
    global_colors = [cl.cnames['aqua'], cl.cnames['orange']]

    configs = [
        'block_s0.yaml',
        'block_s1.yaml'
    ]
    show_names = [
        'No Demo (block_s0.yaml)',
        'With Demo (block_s1.yaml)',
    ]
    conf_base = './config'

    data_plot = {}
    for c, name in zip(configs, show_names):
        full_conf = load_conf(opj(conf_base, c))
        conf = full_conf.train_config
        # progress_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress')
        # result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        tp = TrainingProgress(progress_dir, result_dir, conf.exp_name)
        tp.restore_progress(1800)
        reward = tp.get_step_data('total_reward', 'episode', 1, 1801)
        ewma_reward = calc_ewma_reward(np.asarray(reward))
        data_plot[name] = np.asarray([0] + reward)
        data_plot[name + '-ewma'] = ewma_reward
        print('Done Processing {},avg_step={}'.format(name, tp.get_step_data('Mean Length', 'eval', 1800, 1801, 1)))

    fig = plt.figure(dpi=300, figsize=(6, 3))
    fig.suptitle('Total Reward-{}'.format('FetchReach-v1'))
    x_ticks = list(range(0, 1800 + 1, 1))
    # for i, (k, v) in enumerate(append_dict.items()):
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)
    # ax.set_xticks(x_ticks)
    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)

    c_idx = 0
    for name in show_names:
        color = global_colors[c_idx]
        v1 = data_plot[name]
        ax.plot(x_ticks, v1, linewidth=1, color=color, alpha=0.2)
        v2 = data_plot[name + '-ewma']
        ax.plot(x_ticks, v2, label=name, linewidth=1, color=color)
        c_idx += 1
        ax.legend(fontsize='x-small', loc='lower right')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('./plot_-{}.jpg'.format('FetchReach-v1'))
    plt.clf()
    plt.close(fig)


if __name__ == '__main__':
    os.putenv('DISPLAY', ':0')

    progress_base = './progress/block_s1/'
    timestr = time.strftime('%Y%m%d-%H%M%S')
    progress_dir = progress_base + timestr

    result_base = './result/block_s1/'
    result_dir = result_base + timestr

    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
        print('Directory', progress_dir, 'created')
    else:
        print('Directory', progress_dir, 'already exists')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print('Directory', result_dir, 'created')
    else:
        print('Directory', result_dir, 'already exists')

    main()
    # analysis()
