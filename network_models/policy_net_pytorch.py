# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim ,hidden:int, disttype = "categorical"):
        super(Policy_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """

        # self.ob_space = env.observation_space
        # self.act_space = env.action_space
        
        # self.state_dim = self.ob_space.shape[0]
        # if disttype == "categorical":
        #     self.action_dim = self.act_space.n
        # elif disttype == "normal":        
        #     self.action_dim = self.act_space.shape[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.fc1 = nn.Linear(in_features = self.state_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)

        if disttype == "categorical":
            self.fc4 = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
        elif disttype == "normal":
            self.policy_mu = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
            self.policy_sigma = nn.Linear(in_features = self.hidden , out_features=self.action_dim)
        else:
            raise ValueError
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))


        if self.disttype == "categorical":
            prob = torch.softmax(self.fc4(x) , dim = 1)
            action_dist = torch.distributions.Categorical(prob)
        elif self.disttype == "normal":
            mu = self.policy_mu(x)
            sigma = F.softplus(self.policy_sigma(x))
            action_dist = torch.distributions.Normal(mu,sigma)

        return action_dist

    def act(self, state, stochastic= True):
        if stochastic:
            dist = self.forward(state)
            action = dist.sample().item()
            return action
        else:
            raise ValueError
            dist = self.forward(state)
            action = torch.argmax(dist.probs).item()
            return action

class Value_net(nn.Module):
    def __init__(self,state_dim,action_dim , hidden:int):
        super(Value_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden = 20

        self.fc1 = nn.Linear(in_features = self.state_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc3 = nn.Linear(in_features = self.hidden , out_features=self.hidden)
        self.fc4 = nn.Linear(in_features = self.hidden , out_features=1)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.fc4(x)
        return value





