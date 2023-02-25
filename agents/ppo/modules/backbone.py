import os, sys
sys.path.append(os.getcwd())
from agents.ppo.modules.core import *

import numpy as np
import torch.nn as nn

class ActorCriticSiamese(ACSiamese):
    def __init__(self, num_actions: int, stack_size: int):
        super().__init__()

        self.actor = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
            self._layer_init(nn.Linear(512, num_actions), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
            self._layer_init(nn.Linear(512, 1))
        )

class ActorCriticSiameseSmall(ACSiamese):
    def __init__(self, num_actions: int, stack_size: int) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 8, stride=4)),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
            self._layer_init(nn.Linear(512, num_actions), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 8, stride=4)),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
            self._layer_init(nn.Linear(512, 1))
        )

class ActorCriticMultiHead(ACMultiHead):
    def __init__(self, num_actions: int, stack_size: int):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(512, num_actions), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.critic = self._layer_init(nn.Linear(512, 1))
            

class ActorCriticMultiHeadSmall(ACMultiHead):
    def __init__(self, num_actions: int, stack_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(stack_size, 32, 8, stride=4)),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(512, num_actions), std=0.01),
            nn.Softmax(dim=-1)
        )
        self.critic = self._layer_init(nn.Linear(512, 1))