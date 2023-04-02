import os, sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset

from spds.torchtensorlist import TorchTensorList


class Buffer:
    def __init__(self) -> None:
        self.rb_actions = None
        self.rb_obs = None
        self.rb_logprobs = None
        self.rb_rewards = None
        self.rb_values = None
        self.rb_terms = None
    
    def clear(self):
        raise NotImplementedError
    

class RolloutBuffer(Buffer):
    def __init__(self) -> None:
        super().__init__()
        self.rb_actions = []
        self.rb_obs = []
        self.rb_logprobs = []
        self.rb_rewards = []
        self.rb_values = []
        self.rb_terms = []
    
    def clear(self):
        del self.rb_actions[:]
        del self.rb_obs[:]
        del self.rb_logprobs[:]
        del self.rb_rewards[:]
        del self.rb_values[:]
        del self.rb_terms[:]


class PPORolloutBuffer(Buffer, Dataset):
    def __init__(self, capacity:int = 5,
                 device:torch.device = None) -> None:
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = "cpu"

        self.rb_actions = TorchTensorList(device=self.device)
        self.rb_obs = TorchTensorList(device=self.device)
        self.rb_logprobs = TorchTensorList(device=self.device)
        self.rb_rewards = TorchTensorList(device=self.device)
        self.rb_values = TorchTensorList(device=self.device)
        self.rb_terms = []

        self.count = 0
        self.capacity = capacity    

    def __len__(self):
        return len(self.rb_obs)
    
    def __getitem__(self, idx):
        return (self.rb_obs[idx], 
                self.rb_actions[idx],
                self.rb_logprobs[idx],
                self.rb_rewards[idx],
                self.rb_values[idx],
                self.rb_terms[idx])
    
    def append(self, 
               obs: torch.Tensor, 
               act: torch.Tensor, 
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool):
        
        if self.count < self.capacity:
            self.count += 1
            self.rb_obs.append(obs)
            self.rb_actions.append(act)
            self.rb_logprobs.append(log_probs)
            self.rb_rewards.append(rew)
            self.rb_values.append(obs_val)
            self.rb_terms.append(term)
        else:
            self.rb_obs.pop()
            self.rb_actions.pop()
            self.rb_logprobs.pop()
            self.rb_rewards.pop()
            self.rb_values.pop()
            self.rb_terms.pop()

            self.rb_obs.append(obs)
            self.rb_actions.append(act)
            self.rb_logprobs.append(log_probs)
            self.rb_rewards.append(rew)
            self.rb_values.append(obs_val)
            self.rb_terms.append(term)
    
    def clear(self):
        del self.rb_actions[:]
        del self.rb_obs[:]
        del self.rb_logprobs[:]
        del self.rb_rewards[:]
        del self.rb_values[:]
        del self.rb_terms[:]