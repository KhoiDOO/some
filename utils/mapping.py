from envs.warlords.warlord_env import *
from envs.pong.pong_env import *
from torch import optim
from agents.ppo.agent import PPO
from agents.dume.agent import DUME

env_mapping = {
    "warlords" : wardlord_env_build,
    "pong" : pong_env_build
}

env_parobs_mapping = {
    "warlords" : wardlord_coordinate_obs,
    "pong" : pong_coordinate_obs
}

env_parobs_merge_mapping = {
    "warlords" : wardlord_partial_obs_merge,
    "pong" : pong_partial_obs_merge
}

log_mapping = {
    "warlords" : {
        "ep": [],
        "step": [],
        "first_0": [],
        "second_0": [],
        "third_0": [],
        "fourth_0": []
    },
    "pong" : {
        "ep" : [],
        "step" : [],
        "first_0" : [],
        "second_0" : []
    }
}

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

agent_mapping = {
    "ppo" : PPO,
    "dume" : DUME
}

