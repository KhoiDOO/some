import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
from utils.batchify import *
from agents.irg.modules.mlp_backbone import *
from agents.irg.modules.core import *

class SkillEncoder(BaseLayer):
    def __init__(self, 
            obs_inchannel:int = 4, obs_outchannel:int = 64, act_inchannel: int = 1, 
            backbone_index:int = 4, device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 128)).to(device)
        self.skill_embed_net = self._layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
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
        backbone_index:int = 4, device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 256)).to(device)
        self.task_embed_net = self._layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, prev_action, prev_reward):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, prev_action, prev_reward), dim = 1)
        task_embed = self.task_embed_net(concat)
        return task_embed, obs_feature

class ObservationDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, task_embedding_size:int = 64, 
        backbone_index:int = 4, device:str = "cuda") -> None:
        super().__init__()
        self.obs_decoder = self.backbone_mapping[backbone_index]["decoder"](inchannel=obs_encoded_size+task_embedding_size+1).to(device)
    
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