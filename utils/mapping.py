from envs.warlords.warlord_env import wardlord_env_build
from torch import optim
from agents.ppo.agent import PPO
from agents.dume.agent import DUME

env_mapping = {
    "warlords" : wardlord_env_build
}

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

agent_mapping = {
    "ppo" : PPO,
    "dume" : DUME
}