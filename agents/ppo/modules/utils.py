from datetime import datetime
import torch
import random
import numpy as np
import pandas as pd
from copy import copy, deepcopy


def mask_checkout(rewards):
    print(rewards)
    mask = torch.zeros_like(rewards)
    nz_idx = torch.nonzero(rewards != 0).squeeze()
    mask[:nz_idx[0] + 1] = 1
    for i in range(1, len(nz_idx)):
        if rewards[nz_idx[i]] == -1:
            mask[nz_idx[i - 1] + 1:nz_idx[i] + 1] = 1
    return mask.bool()


def select_obs(self, obs: torch.Tensor,
               act: torch.Tensor,
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool,
               masks: torch.Tensor):

    sel_obs = obs[masks]
    sel_act = act[masks]
    sel_probs = log_probs[masks]
    sel_rew = rew[masks]
    sel_obs_val = obs_val[masks]
    sel_term = term[masks]
    return sel_obs, sel_act, sel_probs, sel_rew, sel_obs_val, sel_term