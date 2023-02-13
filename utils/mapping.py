from envs.warlords import warlord_env
from envs.pong import pong_env
from torch import optim
from agents.ppo.agent import PPO
from agents.dume.agent import DUME

env_mapping = {
    "warlords" : warlord_env.wardlord_env_build,
    "pong" : pong_env.pong_env_build
}

env_parobs_mapping = {
    "warlords" : warlord_env.wardlord_coordinate_obs,
    "pong" : pong_env.pong_coordinate_obs
}

env_parobs_merge_mapping = {
    "warlords" : warlord_env.wardlord_partial_obs_merge,
    "pong" : pong_env.pong_partial_obs_merge
}

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

agent_mapping = {
    "ppo" : PPO,
    "dume" : DUME
}