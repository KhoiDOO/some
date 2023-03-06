import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
from utils.batchify import *
from agents.irg.modules.mlp_backbone import *
from agents.irg.modules.core import BaseLayer, _layer_init

class SkillEncoder(BaseLayer):
    def __init__(self, 
            obs_inchannel:int = 4, obs_outchannel:int = 64, act_inchannel: int = 1, 
            backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index][backbone_scale]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 128)).to(device)
        self.skill_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, action):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, action), dim = 1)
        skill_embed = self.skill_embed_net(concat)
        return skill_embed, obs_feature

class SkillDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, skill_embedding_size:int = 64, device:str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.mlp_decoder = MLP(
            channels=[obs_encoded_size + skill_embedding_size, 64, 32, 1], 
            device = self.device).to(device)

    def forward(self, obs_encoded, skill_embedding):
        # obs_encoded = self.obs_encoder(obs)
        concat = torch.cat((obs_encoded, skill_embedding), dim = 1)
        return self.mlp_decoder(concat)

class TaskEncoder(BaseLayer):
    def __init__(self, obs_inchannel:int = 4, obs_outchannel:int = 64, act_inchannel:int = 2, 
        backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index][backbone_scale]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 256)).to(device)
        self.task_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, prev_action, prev_reward):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, prev_action, prev_reward), dim = 1)
        task_embed = self.task_embed_net(concat)
        return task_embed, obs_feature

class ObservationDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, task_embedding_size:int = 64, 
        backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_decoder = self.backbone_mapping[backbone_index][backbone_scale]["decoder"](inchannel=obs_encoded_size+task_embedding_size+1).to(device)
    
    def forward(self, obs_encoded, action, task_embedding):
        concat = torch.cat((obs_encoded, action, task_embedding), dim = 1)
        pred_obs = self.obs_decoder(concat)
        return pred_obs

class RewardDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, task_embedding_size:int = 64, device:str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.mlp_decoder = MLP(channels=[obs_encoded_size + task_embedding_size + 1, 64, 1], device=self.device).to(device)
    
    def forward(self, obs_encoded, action, task_embedding):
        concat = torch.cat((obs_encoded, action, task_embedding), dim = 1)
        pred_reward = self.mlp_decoder(concat)
        return pred_reward