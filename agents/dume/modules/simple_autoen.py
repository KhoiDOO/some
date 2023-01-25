import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any
from envs.warlords.warlord_env import wardlord_env_build
from utils.batchify import *
import matplotlib.pyplot as plt
from mlp_backbone import *

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight) #he normal
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

hparams = {
    "nc" : 4,
    "nfe" : 64,
    "nfd" : 64,
    "nz" : 128
}

class SimpleEncoder(nn.Module):
    def __init__(self, hparams = hparams) -> None:        
        super().__init__()
        self.network = nn.Sequential(
            # input (nc) x 128 x 128
            _layer_init(nn.Conv2d(hparams["nc"], hparams["nfe"], 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfe"]),
            nn.LeakyReLU(True),
            # input (nfe) x 64 x 64
            _layer_init(nn.Conv2d(hparams["nfe"], hparams["nfe"] * 2, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfe"] * 2),
            nn.LeakyReLU(True),
            # input (nfe*2) x 32 x 32
            _layer_init(nn.Conv2d(hparams["nfe"] * 2, hparams["nfe"] * 4, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfe"] * 4),
            nn.LeakyReLU(True),
            # input (nfe*4) x 16 x 16
            _layer_init(nn.Conv2d(hparams["nfe"] * 4, hparams["nfe"] * 8, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfe"] * 8),
            nn.LeakyReLU(True),
            # input (nfe*8) x 8 x 8
            _layer_init(nn.Conv2d(hparams["nfe"] * 8, hparams["nfe"] * 16, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfe"] * 16),
            nn.LeakyReLU(True),
            # input (nfe*16) x 4 x 4
            _layer_init(nn.Conv2d(hparams["nfe"] * 16, hparams["nz"], 4, 1, 0)),
            nn.BatchNorm2d(hparams["nz"]),
            nn.LeakyReLU(True)
            # output (nz) x 1 x 1
        )

    def forward(self, x):
        return self.network(x/255)

class SimpleDecoder(nn.Module):
    def __init__(self, hparams = hparams) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # input (nz) x 1 x 1
            _layer_init(nn.ConvTranspose2d(hparams["nz"], hparams["nfd"] * 16, 4, 1, 0)),
            nn.BatchNorm2d(hparams["nfd"] * 16),
            nn.ReLU(True),
            # input (nfd*16) x 4 x 4
            _layer_init(nn.ConvTranspose2d(hparams["nfd"] * 16, hparams["nfd"] * 8, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfd"] * 8),
            nn.ReLU(True),
            # input (nfd*8) x 8 x 8
            _layer_init(nn.ConvTranspose2d(hparams["nfd"] * 8, hparams["nfd"] * 4, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfd"] * 4),
            nn.ReLU(True),
            # input (nfd*4) x 16 x 16
            _layer_init(nn.ConvTranspose2d(hparams["nfd"] * 4, hparams["nfd"] * 2, 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfd"] * 2),
            nn.ReLU(True),
            # input (nfd*2) x 32 x 32
            _layer_init(nn.ConvTranspose2d(hparams["nfd"] * 2, hparams["nfd"], 4, 2, 1)),
            nn.BatchNorm2d(hparams["nfd"]),
            nn.ReLU(True),
            # input (nfd) x 64 x 64
            _layer_init(nn.ConvTranspose2d(hparams["nfd"], hparams["nc"], 4, 2, 1)),
            nn.Tanh()
            # output (nc) x 128 x 128
        )
    
    def forward(self, x):
        return self.network(x)

class SkillEncoder(nn.Module):
    def __init__(self, obs_inchannel=4, obs_outchannel=512, act_inchannel=1) -> None:
        super().__init__()
        self.obs_encoder = SimpleEncoder(inchannel=obs_inchannel, outchannel=obs_outchannel)
        self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 256))
        self.skill_embed_net = _layer_init(nn.Linear(256, 128))
    
    def forward(self, obs, action):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, action), dim = 1)
        skill_embed = self.skill_embed_net(self.dense(concat))
        return skill_embed

class SkillDecoder(nn.Module):
    def __init__(self, obs_inchannel=4, obs_outchannel=128) -> None:
        super().__init__()
        self.obs_encoder = SimpleEncoder(inchannel=obs_inchannel, outchannel=obs_outchannel)
    def forward(self, obs, skill_embedding):
        
        pass

class TaskEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, obs, prev_action, prev_reward):
        pass

class ObservationDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, obs, action, task_embedding):
        pass

class RewardDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, obs, action, task_embedding):
        pass

# Verify simple encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = wardlord_env_build(frame_size=(128, 128))
obs = env.reset()
obs = batchify(obs, device)
obs = obs.to(torch.float)
obs = obs.permute(0, 3, 2, 1)[0]
obs = obs.view(-1, 4, 128, 128)

simple_en_test = SimpleEncoder()
encoded_obs = simple_en_test(obs)
print(encoded_obs.shape)

# test_action = torch.randn(1)
# test_action = test_action.view(-1, 1)
# test_skill_encoder = SkillEncoder()
# test_skill_embbed = test_skill_encoder(obs, test_action)
# print(test_skill_embbed.shape)