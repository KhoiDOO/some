from datetime import datetime
import torch
import random
import numpy as np
import pandas as pd
from copy import copy, deepcopy


def mask_checkout(rewards):
    tmp_mask = rewards
    mask = tmp_mask.bool()
    return None


def select_obs(self, obs: torch.Tensor,
               act: torch.Tensor,
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool,
               masks):

    sel_obs = obs[masks]
    sel_act = act[masks]
    sel_probs = log_probs[masks]
    sel_rew = rew[masks]
    sel_obs_val = obs_val[masks]
    sel_term = term[masks]
    return sel_obs, sel_act, sel_probs, sel_rew, sel_obs_val, sel_term