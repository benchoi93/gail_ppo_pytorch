#!/usr/bin/python3
import argparse
import gym
import numpy as np
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
    
    policy = Policy_net(env, hidden = 64, disttype="normal")
    old_policy = Policy_net(env, hidden = 64, disttype="normal")
    value = Value_net(env, hidden = 64)

    PPO = PPOTrain(policy, old_policy, value, gamma=args.gamma)
    obs = env.reset()
    success_num = 0

    D = Discriminator(env, hidden = 64, disttype="normal")
    discrim_opt = torch.optim.Adam(D.parameters(),lr = PPO.lr,eps=PPO.eps)

    if args.cuda:
        policy = policy.cuda()
        old_policy = old_policy.cuda()
        value = value.cuda()
        D = D.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() & args.cuda else "cpu")

    # expert = np.load("trajectory/mountain_car_expert_demo.npy")
    # expert = np.reshape(expert, newshape = (expert.shape[0]*expert.shape[1] , expert.shape[2]))
    # expert = expert[expert[:,2] != -1.0,:]

    # expert_observations = expert[:,0:2]
    # expert_actions = expert[:,2]
    
    expert_observations = np.load("trajectory/pendulum_expert_states.npy")
    expert_actions      = np.load("trajectory/pendulum_expert_action.npy")

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
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                # env.render()
                episode_length += 1
                obs = torch.Tensor(obs).unsqueeze(0)
                action = torch.tanh(torch.Tensor([policy.act(obs.to(device))])).item()
                v_pred = value.forward(obs.to(device)).item()

                next_obs, reward, done, info = env.step(action)

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
            trajs.append([observations, actions, rewards, v_preds,v_preds_next])
        
        avg_reward = np.mean([np.sum(x[2]) for x in trajs])
        print("Total Avg. Reward = {:.2f}".format( avg_reward ) )
        PPO.summary.add_scalar('reward', avg_reward ,PPO.summary_cnt )

        # convert list to numpy array for feeding tf.placeholder
        obs_temp = sum([x[0] for x in trajs],[])
        act_temp = sum([x[1] for x in trajs],[])

        learner_obs = torch.cat(obs_temp).float().to(device)
        learner_act = torch.Tensor(act_temp).float().to(device).unsqueeze(1)

        for _ in range(3):
            idx = np.random.choice(expert_observations.shape[0], learner_obs.shape[0]*10, replace=False)
            sampled_exp_obs = expert_observations[idx, :]
            sampled_exp_act = expert_actions[idx, :]

            expert_obs = torch.from_numpy(sampled_exp_obs).float().to(device)
            expert_act = torch.from_numpy(sampled_exp_act).float().to(device)

            expert_prob = D.forward(expert_obs,expert_act)
            learner_prob = D.forward(learner_obs , learner_act)

            expert_target  = torch.zeros_like(expert_prob)
            learner_target = torch.ones_like(learner_prob)

            criterion = torch.nn.BCELoss()
            discrim_loss = criterion(expert_prob , expert_target) + \
                criterion(learner_prob , learner_target)

            discrim_opt.zero_grad()
            discrim_loss.backward()    
            discrim_opt.step()

            expert_acc  = ((expert_prob < 0.5).float()).mean()
            learner_acc = ((learner_prob > 0.5).float()).mean()

        PPO.summary.add_scalar('loss/discrim',discrim_loss.item() ,PPO.summary_cnt )
        PPO.summary.add_scalar('accuracy/expert',expert_acc.item() ,PPO.summary_cnt )
        PPO.summary.add_scalar('accuracy/learner',learner_acc.item() ,PPO.summary_cnt )

        inp=[]
        for i in range(args.n_episode):
            obs, act, reward, v_pred,v_pred_next = trajs[i]
            obs = torch.cat(obs).float().to(device)
            act = torch.Tensor(act).float().to(device).unsqueeze(1)

            d_reward = D.get_reward(obs , act)
            d_reward = d_reward.squeeze(1)

            gaes = PPO.get_gaes(rewards=d_reward, v_preds=v_pred, v_preds_next=v_pred_next)
            gaes = torch.Tensor(gaes)
            gaes = (gaes-gaes.mean())/gaes.std()
            rewards = torch.Tensor(rewards)
            v_pred_next = torch.Tensor(v_pred_next)
            inp.append( [obs, act, gaes, d_reward, v_pred_next] ) 
        
        avg_d_reward = np.mean([torch.sum(x[3]).item() for x in inp])
        step_avg_d_reward = np.mean([torch.mean(x[3]).item() for x in inp])
        PPO.summary.add_scalar('reward/d_reward', avg_d_reward ,PPO.summary_cnt )
        PPO.summary.add_scalar('reward/step_d_reward', step_avg_d_reward ,PPO.summary_cnt )

        PPO.hard_update(old_policy , policy)

        obs = torch.cat([x[0] for x in inp])
        act = torch.cat([x[1] for x in inp])
        gaes = torch.cat([x[2] for x in inp])
        d_reward = torch.cat([x[3] for x in inp])
        v_pred_next = torch.cat([x[4] for x in inp])
        
        inp = [obs, act, gaes, d_reward, v_pred_next]
        
        # train
        for _ in range(15):
            sample_indices = np.random.randint(low=0, high=obs.shape[0], size=args.batch_size)
            sampled_inp = [np.take(a=a.cpu(), indices=sample_indices, axis=0) for a in inp]  # sample training data
            PPO.train(obs       =sampled_inp[0].to(device),
                      actions   =sampled_inp[1].to(device),
                      gaes      =sampled_inp[2].to(device),
                      rewards   =sampled_inp[3].to(device),
                      v_preds_next=sampled_inp[4].to(device))


if __name__ == '__main__':
    args = argparser()
    main(args)
