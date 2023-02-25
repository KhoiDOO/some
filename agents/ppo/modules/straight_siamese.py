import os, sys
sys.path.append(os.getcwd())
from agents.ppo.modules.core import *

import numpy as np
import torch.nn as nn

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticSiameseV1(nn.Module):
    def __init__(self, num_actions: int, stack_size: int):
        super().__init__()

        self.actor = nn.Sequential(
            _layer_init(nn.Conv2d(stack_size, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
            _layer_init(nn.Linear(512, num_actions), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Conv2d(stack_size, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
            _layer_init(nn.Linear(512, 1))
        )

    def act(self, obs: torch.Tensor):
        action_probs = self.actor(obs/255)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        obs_val = self.critic(obs/255)

        return action.detach(), action_logprob.detach(), obs_val.detach()

    def forward(self, obs: torch.Tensor):
        action_probs = self.actor(obs/255)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        obs_val = self.critic(obs/255)

        return action.detach(), action_logprob.detach(), obs_val.detach()
    
    def evaluate(self, obs, action):
        action_probs = self.actor(obs/255)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        obs_val = self.critic(obs/255)
        
        return action_logprobs, obs_val, dist_entropy