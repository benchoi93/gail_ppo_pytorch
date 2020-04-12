#!/usr/bin/python3
import argparse
import gym
import numpy as np
import gc
# import tensorflow as tf
import torch
from network_models.policy_net_pytorch import Policy_net,Value_net
from network_models.discriminator_pytorch import Discriminator
from algo.ppo_pytorch import PPOTrain


def argparser():
    import sys
    sys.argv=['--logdir log/train/ppo']
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    parser.add_argument('--n-episode',default=int(5) , type= int)
    parser.add_argument('--batch-size', default=int(64) , type = int)
    parser.add_argument('--cuda' , default=True)
    parser.add_argument('--hidden' , default = int(64) , type = int)
    return parser.parse_args()
args = argparser()

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        return action

def main(args):
    env =  NormalizedActions(gym.make('Pendulum-v0'))
    # env = gym.make('CartPole-v0')
    # env = gym.make('MountainCar-v0')
    env.seed(0)
    ob_space = env.observation_space
    
    policy = Policy_net(env, hidden = args.hidden, disttype="normal")
    old_policy = Policy_net(env, hidden = args.hidden, disttype="normal")
    value = Value_net(env, hidden = args.hidden)

    PPO = PPOTrain(policy, old_policy, value, gamma=args.gamma)
    obs = env.reset()
    success_num = 0

    # D = Discriminator(env, hidden = 64, disttype="normal")
    # discrim_opt = torch.optim.Adam(D.parameters(),lr = PPO.lr,eps=PPO.eps)

    if args.cuda:
        policy = policy.cuda()
        old_policy = old_policy.cuda()
        value = value.cuda()
        # D = D.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() & args.cuda else "cpu")

    # expert = np.load("trajectory/mountain_car_expert_demo.npy")
    # expert = np.reshape(expert, newshape = (expert.shape[0]*expert.shape[1] , expert.shape[2]))
    # expert = expert[expert[:,2] != -1.0,:]

    # expert_observations = expert[:,0:2]
    # expert_actions = expert[:,2]
    
    # expert_observations = np.load("trajectory/pendulum_expert_states.npy")
    # expert_actions      = np.load("trajectory/pendulum_expert_action.npy")

    # expert_observations = np.genfromtxt('trajectory/observations.csv')
    # expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    for iteration in range(args.iteration):
        trajs = []
        obs = env.reset()
        for _ in range(args.n_episode):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            policy_outs = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                # env.render()
                episode_length += 1
                obs = torch.Tensor(obs).unsqueeze(0)

                action_dist = policy.forward(obs.to(device))
                sample = action_dist.rsample()
                action = torch.tanh(sample)
                # policy_out = torch.Tensor([policy.act(obs.to(device))])
                # action = torch.tanh(policy_out).item()
                v_pred = value.forward(obs.to(device)).item()

                next_obs, reward, done, info = env.step(action.item())

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                v_preds.append(v_pred)
                policy_outs.append(sample)

                if done:
                    next_obs = torch.Tensor(next_obs).unsqueeze(0)
                    v_pred = value.forward(next_obs.to(device))
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs = env.reset()
                    break
                else:
                    obs = next_obs
            trajs.append([observations, actions, rewards, v_preds,v_preds_next,policy_outs])
        
        avg_reward = np.mean([np.sum(x[2]) for x in trajs])
        print("Total Avg. Reward = {:.2f}".format( avg_reward ) )
        PPO.summary.add_scalar('reward', avg_reward ,PPO.summary_cnt )

        inp=[]
        for i in range(args.n_episode):
            obs, act, reward, v_pred,v_pred_next,policy_outs = trajs[i]
            obs = torch.cat(obs).float()
            act = torch.cat(act).float().squeeze(1)
            # act = torch.Tensor(act).float().to(device).unsqueeze(1)
            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_pred, v_preds_next=v_pred_next)
            gaes = torch.Tensor(gaes)
            gaes = (gaes-gaes.mean())/gaes.std()
            rewards = torch.Tensor(rewards)
            v_pred_next = torch.Tensor(v_pred_next)
            # policy_outs = torch.Tensor(policy_outs)
            policy_outs = torch.cat(policy_outs).squeeze(1)
            inp.append( [obs, act, gaes, rewards, v_pred_next,policy_outs] ) 
        
        PPO.hard_update(old_policy , policy)

        obs = torch.cat([x[0] for x in inp])
        act = torch.cat([x[1] for x in inp])
        gaes = torch.cat([x[2] for x in inp])
        rewards = torch.cat([x[3] for x in inp])
        v_pred_next = torch.cat([x[4] for x in inp])
        policy_outs = torch.cat([x[5] for x in inp])
        
        inp = [obs, act, gaes, rewards, v_pred_next, policy_outs]
        
        # train
        # for _ in range(20):
        #     sample_indices = np.random.randint(low=0, high=obs.shape[0], size=args.batch_size)
        #     # sampled_inp = [np.take(a=a.cpu(), indices=sample_indices, axis=0) for a in inp]  # sample training data

        #     PPO.train(obs       = obs[sample_indices].to(device)               ,
        #               actions   = act[sample_indices].to(device)               ,
        #               gaes      = gaes[sample_indices].to(device)              ,
        #               rewards   = rewards[sample_indices].to(device)           ,
        #               v_preds_next= v_pred_next[sample_indices].to(device)     ,
        #               policy_outs = policy_outs[sample_indices].to(device) 
        #               ) 

        PPO.train(obs       = obs.to(device)               ,
                  actions   = act.to(device)               ,
                  gaes      = gaes.to(device)              ,
                  rewards   = rewards.to(device)           ,
                  v_preds_next= v_pred_next.to(device)     ,
                  policy_outs = policy_outs.to(device) 
                  ) 

            # PPO.train_trajs(trajs, device)
        # gc.collect()


if __name__ == '__main__':
    args = argparser()
    main(args)
