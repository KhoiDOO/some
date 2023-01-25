from envs.warlords.warlord_env import wardlord_env_build
import numpy as np
import pandas
import argparse
import torch
import os 
import json
from utils.batchify import *
from utils.mapping import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="warlords", choices=["overcook", "warlords"],
        help="Environment used in training and testing")
    parser.add_argument("--agent", type=str, default="ppo",
        help="Number of episodes in rendering")
    parser.add_argument("--ep", type=str, default=5,
        help="Number of episodes in rendering")
    args = parser.parse_args()

    datetime_choosed = "2023-01-08 19:54:05.036171"

    setting_file_pth = os.getcwd() + f"/models/{args.agent}/log/{datetime_choosed}.json"
    # result_file_pth = os.getcwd() + f"/models/{args.agent}/log/{datetime_choosed}.csv"
    model_pth = os.getcwd() + f"/models/{args.agent}/model_save/{datetime_choosed}.pth"

    config = json.load(open(setting_file_pth))
    env_cfg = config["env_meta"]

    env = env_mapping[args.env](stack_size = env_cfg["stack_size"], frame_size = tuple(env_cfg["frame_size"]),
                        max_cycles = env_cfg["max_cycles"], render_mode = "human",
                        parralel = env_cfg["parralel"], color_reduc=env_cfg["color_reduc"])

    agent = torch.load(model_pth)
    agent.eval()

    with torch.no_grad():
        for episode in range(args.ep):
            obs = batchify_obs(env.reset(seed=None), device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]