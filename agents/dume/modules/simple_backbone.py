import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from torchvision.transforms import Resize
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any
from envs.warlords.warlord_env import wardlord_env_build
from utils.batchify import *
from agents.dume.modules.mlp_backbone import *
import matplotlib.pyplot as plt

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight) #he normal
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# class SimpleEncoder(nn.Module):
#     def __init__(self, inchannel: int, outchannel: int) -> None:        
#         super().__init__()
#         self.network = nn.Sequential(
#             # inchannel * 16 * 16
#             _layer_init(nn.Conv2d(inchannel, 32, 3, padding=1)),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             # 32 * 8 * 8
#             _layer_init(nn.Conv2d(32, 64, 3, padding=1)),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             # 64 * 4 * 4
#             _layer_init(nn.Conv2d(64, 128, 3, padding=1)),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             # 128 * 2 * 2
#             nn.Flatten(),
#             _layer_init(nn.Linear(128 * 4 * 4, outchannel)),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.network(x)

# class SimpleDecoder(nn.Module):
#     def __init__(self, inchannel: int) -> None:
#         super().__init__()
#         self.network = nn.Sequential(
#             _layer_init(nn.Linear(inchannel, 128 * 4 * 4)),
#             nn.Unflatten(1, (128, 4, 4)),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             #128 * 4 * 4
#             _layer_init(nn.ConvTranspose2d(128, 64, 3, padding=1)),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             #64 * 8 * 8
#             _layer_init(nn.ConvTranspose2d(64, 32, 3, padding=1)),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             #32 * 16 * 16
#             _layer_init(nn.ConvTranspose2d(32, 4, 3, padding=1)),
#             #4 * 16 * 16
#         )
    
#     def forward(self, x):
#         return self.network(x)

class SimpleEncoder(nn.Module):
    def __init__(self, inchannel: int, outchannel: int) -> None:        
        super().__init__()
        self.network = nn.Sequential(
            Resize(size = 32),
            # inchannel * 32 * 32
            _layer_init(nn.Conv2d(inchannel, 8, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 8 * 14 * 14
            _layer_init(nn.Conv2d(8, 16, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 16 * 5 * 5
            _layer_init(nn.Conv2d(16, 32, 4)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 32 * 1 * 1
            # nn.AdaptiveAvgPool2d((None, 32)),
            # _layer_init(nn.Linear(32, outchannel)),
            # nn.ReLU(),
        )
        self.outer = nn.Sequential(
            _layer_init(nn.Linear(32, outchannel)),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=(-2, -1))
        return self.outer(x)

class SimpleDecoder(nn.Module):
    def __init__(self, inchannel: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(inchannel, 32)),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #32 * 2 * 2
            _layer_init(nn.ConvTranspose2d(32, 16, 4)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #16 * 10 * 10
            _layer_init(nn.ConvTranspose2d(16, 8, 5)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #8 * 28 * 28
            _layer_init(nn.ConvTranspose2d(8, 4, 5)),
            #4 * 32 * 32
        )
    
    def forward(self, x):
        return self.network(x)

class SkillEncoder(nn.Module):
    def __init__(self, obs_inchannel=4, obs_outchannel=64, act_inchannel=1, device = "cuda") -> None:
        super().__init__()
        self.obs_encoder = SimpleEncoder(inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 128)).to(device)
        self.skill_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, action):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, action), dim = 1)
        skill_embed = self.skill_embed_net(concat)
        return skill_embed, obs_feature

class SkillDecoder(nn.Module):
    def __init__(self, obs_encoded_size=64, skill_embedding_size=64, device = "cuda") -> None:
        super().__init__()
        self.device = device
        # self.obs_encoder = SimpleEncoder(inchannel=obs_inchannel, outchannel=obs_outchannel)
        self.mlp_decoder = MLP(
            channels=[obs_encoded_size + skill_embedding_size, 64, 32, 1], 
            device = self.device).to(device)

    def forward(self, obs_encoded, skill_embedding):
        # obs_encoded = self.obs_encoder(obs)
        concat = torch.cat((obs_encoded, skill_embedding), dim = 1)
        return self.mlp_decoder(concat)

class TaskEncoder(nn.Module):
    def __init__(self, obs_inchannel = 4, obs_outchannel = 64, act_inchannel = 2, device = "cuda") -> None:
        super().__init__()
        self.obs_encoder = SimpleEncoder(inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 256)).to(device)
        self.task_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, prev_action, prev_reward):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, prev_action, prev_reward), dim = 1)
        task_embed = self.task_embed_net(concat)
        return task_embed, obs_feature

class ObservationDecoder(nn.Module):
    def __init__(self, obs_encoded_size=64, task_embedding_size=64, device = "cuda") -> None:
        super().__init__()
        self.obs_decoder = SimpleDecoder(inchannel=obs_encoded_size+task_embedding_size+1).to(device)
    
    def forward(self, obs_encoded, action, task_embedding):
        concat = torch.cat((obs_encoded, action, task_embedding), dim = 1)
        pred_obs = self.obs_decoder(concat)
        return pred_obs

class RewardDecoder(nn.Module):
    def __init__(self, obs_encoded_size=64, task_embedding_size=64, device = "cuda") -> None:
        super().__init__()
        self.device = device
        self.mlp_decoder = MLP(channels=[obs_encoded_size + task_embedding_size + 1, 64, 1], device=self.device).to(device)
    
    def forward(self, obs_encoded, action, task_embedding):
        concat = torch.cat((obs_encoded, action, task_embedding), dim = 1)
        pred_reward = self.mlp_decoder(concat)
        return pred_reward

# Verify

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = wardlord_env_build(frame_size=(16,16))
# obs = env.reset()
# obs = batchify(obs, device)
# obs = obs.to(torch.float)
# obs = obs.permute(0, 3, 2, 1)[0]
# current_obs = obs.view(-1, 4, 16, 16)
# print("Current obs shape: {}".format(current_obs.shape))

# current_action = torch.randn(1)
# current_action = current_action.view(-1, 1)
# print("Current action shape: {}".format(current_action.shape))

# previous_action = torch.randn(1)
# previous_action = previous_action.view(-1, 1)
# print("Previous action shape: {}".format(previous_action.shape))

# previous_reward = torch.randn(1)
# previous_reward = previous_reward.view(-1, 1)
# print("Previous reward shape: {}".format(previous_reward.shape))

# skill_encoder = SkillEncoder(obs_inchannel=4, obs_outchannel=128, act_inchannel=1)
# skill_embedding, obs_skill_encoded = skill_encoder(current_obs, current_action)
# print("Skill Embedding Shape: {}".format(skill_embedding.shape))
# print("Obs SKill Encoded Shape: {}".format(obs_skill_encoded.shape))

# skill_decoder = SkillDecoder(obs_encoded_size=128, skill_embedding_size=128)
# predict_action = skill_decoder(obs_skill_encoded, skill_embedding)
# print("Predict Action Shape: {}".format(predict_action.shape))

# task_encoder = TaskEncoder(obs_inchannel=4, obs_outchannel=128, act_inchannel=2)
# task_embedding, obs_task_encoded = task_encoder(current_obs, previous_action, previous_reward)
# print("Task Embedding Shape: {}".format(task_embedding.shape))
# print("Obs Task Encoded Shape: {}".format(obs_task_encoded.shape))

# obs_decoder = ObservationDecoder(obs_encoded_size=128, task_embedding_size=128)
# predict_obs = obs_decoder(obs_task_encoded, current_action, task_embedding)
# print("Predict Observation Shape: {}".format(predict_obs.shape))

# reward_decoder = RewardDecoder(obs_encoded_size=128, task_embedding_size=128)
# predict_reward = reward_decoder(obs_task_encoded, current_action, task_embedding)
# print("Predict Reward Shape: {}".format(predict_reward.shape))