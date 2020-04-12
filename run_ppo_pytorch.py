#!/usr/bin/python3
import argparse
import gym
import numpy as np
# import tensorflow as tf
import torch
from network_models.policy_net_pytorch import Policy_net,Value_net
from algo.ppo_pytorch import PPOTrain

class DiscretizedActions(gym.ActionWrapper):
    def set_action_space(self,num_actions):
        self.high = self.action_space.high.item()
        self.low  = self.action_space.low.item()
        self.num_actions = num_actions

        self.deltas = (self.high - self.low) / self.num_actions

        self.high_list = self.low + np.array(range(self.num_actions), dtype = np.float32)  * self.deltas + self.deltas
        self.low_list  = self.low + np.array(range(self.num_actions), dtype = np.float32)  * self.deltas
        self.mid_list  = self.low + np.array(range(self.num_actions), dtype = np.float32)  * self.deltas + self.deltas/2

    def action(self, action, deterministic = True):
        if deterministic:
            return self.mid_list[action]
        else:
            return np.random.uniform(self.low_list[action] , self.high_list[action])


def argparser():
    import sys
    # sys.argv=['--logdir log/train/ppo']
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e6), type=int)
    parser.add_argument('--num_actions', default = int(10) , type=int)
    parser.add_argument('--n-episode',default=int(5) , type= int)
    parser.add_argument('--batch-size', default=int(64) , type = int)
    parser.add_argument('--learning-rate' , default=float(5e-5), type = float)
    parser.add_argument('--cuda' , default=True)
    return parser.parse_args()
args = argparser()

def main(args):
    # env = gym.make('CartPole-v0')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Pendulum-v0') 
    env = DiscretizedActions(gym.make('Pendulum-v0') )
    env.set_action_space(args.num_actions)

    env.seed(0)
    ob_space = env.observation_space
    act_space =env.action_space
    
    state_dim = ob_space.shape[0]
    # action_dim = act_space.shape[0]
    action_dim = args.num_actions

    policy = Policy_net(state_dim,action_dim,hidden=64, disttype = "categorical")
    old_policy = Policy_net(state_dim,action_dim,hidden=64, disttype = "categorical")
    value = Value_net(state_dim,action_dim,hidden=64)

    if args.cuda:
        policy = policy.cuda()
        old_policy = old_policy.cuda()
        value = value.cuda()
        # D = D.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() & args.cuda else "cpu")


    PPO = PPOTrain(policy, old_policy, value, gamma=args.gamma, lr = args.learning_rate)
    obs = env.reset()
    success_num = 0
    for iteration in range(args.iteration):
        observations = []
        actions = []
        rewards = []
        v_preds = []
        episode_length = 0
        while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
            # env.render()
            episode_length += 1
            obs = torch.Tensor(obs).unsqueeze(0)
            action = policy.act(obs.to(device))
            v_pred = value.forward(obs.to(device)).item()

            next_obs, reward, done, info = env.step(np.array([action]))

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            v_preds.append(v_pred)

            if done:
                next_obs = torch.Tensor(next_obs).unsqueeze(0)
                v_pred = value.forward(next_obs.to(device))
                v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                obs = env.reset()
                break
            else:
                obs = next_obs

        # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
        #                     , iteration)
        # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
        #                     , iteration)

        print("Total Reward = {:.2f}".format(np.sum(rewards) ))
        PPO.summary.add_scalar('reward', sum(rewards) ,PPO.summary_cnt )

        gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

        # convert list to numpy array for feeding tf.placeholder
        observations = torch.cat(observations)
        actions = torch.Tensor(actions)
        gaes = torch.Tensor(gaes)
        gaes = (gaes-gaes.mean())/gaes.std()
        rewards = torch.Tensor(rewards)
        v_preds_next = torch.Tensor(v_preds_next)

        PPO.hard_update(old_policy , policy)
        inp = [observations, actions, gaes, rewards, v_preds_next]
        # train
        for epoch in range(6):
            # sample indices from [low, high)
            # sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.batch_size)
            sample_indices = np.array(range(200))
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            PPO.train(obs       =sampled_inp[0].to(device),
                      actions   =sampled_inp[1].to(device),
                      gaes      =sampled_inp[2].to(device),
                      rewards   =sampled_inp[3].to(device),
                      v_preds_next=sampled_inp[4].to(device))




if __name__ == '__main__':
    args = argparser()
    main(args)
