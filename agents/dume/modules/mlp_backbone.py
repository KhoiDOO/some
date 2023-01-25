import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight) #he normal
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class MLP(nn.Module):
    def __init__(self, channels: List[int], 
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
        last_layer_activation: bool = False,
        device = "cuda") -> None:
        super().__init__()
        self.last_layer_activation = last_layer_activation
        self.activation_fn = activation_layer
        self.channels = channels
        self.device = device
        self.layers = []
        for i in range(len(self.channels)-1):
            self.layers.append(_layer_init(nn.Linear(self.channels[i], self.channels[i+1])).to(self.device))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.channels) or self.last_layer_activation:
                x = self.activation_fn(x)
        return x

# class SimpleLinearModel(nn.Module):
#     def __init__(self, in_channel, out_channel) -> None:
#         super().__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.dense = _layer_init(nn.Linear(self.in_channel, self.out_channel))
    
#     def forward(self, x):
#         x = self.dense(x)
#         return x

# class TaskEncoder(nn.Module):
#     def __init__(self, channels: List[int], latent_dim: int, 
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU) -> None:
#         super().__init__()
#         self.channels = channels
#         self.latent_dim = latent_dim
#         self.activation_fn = activation_layer
#         self.mlp = MLP(self.channels)
#         self.dense = _layer_init(nn.Linear(self.channels[-1], self.latent_dim))
    
#     def forward(self, trajectories):
#         task_hidden = self.mlp(trajectories)
#         task_latent_mu = self.dense(task_hidden)
#         task_latent_log_std = self.dense(task_hidden)
#         return task_latent_mu, task_latent_log_std

# class TaskEncoderAE(nn.Module):
#     def __init__(self, channels: List[int], latent_dim: int, 
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU) -> None:
#         super().__init__()
#         self.channels = channels
#         self.latent_dim = latent_dim
#         self.activation_fn = activation_layer
#         self.mlp = MLP(channels=self.channels)
#         self.dense = _layer_init(nn.Linear(self.channels[-1], self.latent_dim))
    
#     def forward(self, trajectories):
#         task_hidden = self.mlp(trajectories)
#         task_latent = self.dense(task_hidden)
#         return task_latent

# class TaskEncoderPrior(nn.Module):
#     def __init__(self, channels: List[int], latent_dim: int,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU) -> None:
#         super().__init__()
#         self.channels = channels
#         self.latent_dim = latent_dim
#         self.activation_layer = activation_layer
#         self.mlp = MLP(channels=self.channels, activation_layer=activation_layer, last_layer_activation=True)
#         self.dense = _layer_init(nn.Linear(self.channels[-1], self.latent_dim))
    
#     def forward(self, states, task_id):
#         task_hidden = self.mlp()

#     @nn.compact
#     def __call__(self, states: jnp.ndarray, task_id:jnp.ndarray) -> jnp.ndarray:
#         inputs = jnp.concatenate([states, task_id], -1)
#         task_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(states)
#         task_latent = nn.Dense(self.latent_dim)(task_hidden)
#         return task_latent

# class PolicyEncoderAE(nn.Module):
#     net_arch: List[int]
#     latent_dim : int
#     activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

#     @nn.compact
#     def __call__(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
#         inputs = jnp.concatenate([states, actions], -1)
#         policy_hidden = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(inputs)
#         policy_latent = nn.Dense(self.latent_dim)(policy_hidden)
# anh ơi chắc phần mô hình mình phải refine lại hết bởi vì observation của mình là ảnh chứ ko phải vector như cno
#         # policy_hidden_2 = MLP(net_arch=self.net_arch, activation_fn=self.activation_fn, last_layer_activation=True)(actions[..., :-4])
#         # policy_latent_2 = nn.Dense(self.latent_dim)(policy_hidden_2)
#         return policy_latent

# class PolicyTaskEncoderAE(nn.Module):
#     def __init__(self, net_arch: List[int], task_net_arch: List[int], 
#         policy_net_arch: List[int], latent_dim : int,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU) -> None:
#         super().__init__()
#         self.hidden_value_net = MLP(channels=net_arch, activation_layer=activation_layer, 
#                                     last_layer_activation=True)
#         self.task_hidden_net = MLP(channels=task_net_arch, activation_layer=activation_layer, 
#                                     last_layer_activation=True)
#         self.policy_hidden_net = MLP(channels=policy_net_arch, activation_layer=activation_layer,
#                                     last_layer_activation=True)
#         self.dense = _layer_init(nn.Linear(task_net_arch[-1], latent_dim))
    
#     def forward(self, trajectories):
#         hidden_value = self.hidden_value_net(trajectories)
#         task_hidden = self.task_hidden_net(hidden_value)
#         policy_hidden = self.policy_hidden_net(hidden_value)

#         task_latent= self.dense(task_hidden)
#         policy_latent = self.dense(policy_hidden)
#         return task_latent, policy_latent

