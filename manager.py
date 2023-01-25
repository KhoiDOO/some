import os, sys
import json
from utils.mapping import *
from utils.fol_struc import run_folder_verify
from utils.batchify import *
import argparse
from datetime import datetime
import torch
from tqdm import trange
import cv2 as cv
import pandas as pd

from envs.warlords.warlord_env import wardlord_coordinate_obs, wardlord_partial_obs_merge

def train(args: argparse.PARSER):
    args_dict = vars(args)
    
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verify
    run_folder_verify(current_time)

    # setting path
    save_train_set_path = os.getcwd() + f"/run/train/{current_time}/settings/settings.json"
    save_valid_set_path = os.getcwd() + f"/run/val/{current_time}/settings/settings.json"

    # aggent logging save_dir
    log_agent_dir = os.getcwd() + f"/run/train/{current_time}/log"
    model_agent_dir = os.getcwd() + f"/run/train/{current_time}/weights"

    # main experiment logging
    main_log_dir = os.getcwd() + f"/run/train/{current_time}/log/main"

    with open(save_train_set_path, "w") as outfile:
        json.dump(args_dict, outfile)
    with open(save_valid_set_path, "w") as outfile:
        json.dump(args_dict, outfile)
    
    # Extract
    # print(args_dict)
    env_name = args_dict["env"]
    env_meta = args_dict["env_meta"]
    episodes = args_dict["ep"]
    gamma = args_dict["gamma"]
    p_size = args_dict["view"]

    agent_algo = args_dict["agent"]
    epoches = args_dict["epoches"]
    batch_size = args_dict["bs"]
    actor_lr = args_dict["actor_lr"]
    critic_lr = args_dict["critic_lr"]
    optimizer = args_dict["opt"]

    dume_in_use = args_dict["dume"]
    dume_epoches = args_dict["dume_epoches"]
    dume_batch_size = args_dict["dume_bs"]
    dume_lr = args_dict["dume_lr"]
    dume_optimizer = args_dict["dume_opt"]

    # Create Env
    default_env_meta_path = os.getcwd() + f"/envs/{env_name}/metadata/{env_meta}.json"

    env_metadata = json.load(open(default_env_meta_path)) # get metadata for env
    output_env = env_mapping[env_name](stack_size = env_metadata["stack_size"], frame_size = tuple(env_metadata["frame_size"]),
                        max_cycles = env_metadata["max_cycles"], render_mode = env_metadata["render_mode"],
                        parralel = env_metadata["parralel"], color_reduc=env_metadata["color_reduc"])
    
    # Agents Setting

    agent_names = output_env.possible_agents

    main_algo_agents = {name : agent_mapping[agent_algo](
        stack_size = env_metadata["stack_size"], 
        action_dim = output_env.action_space(output_env.possible_agents[0]).n, 
        lr_actor = actor_lr, 
        lr_critic = critic_lr, 
        gamma = gamma, 
        K_epochs = epoches, 
        eps_clip = 0.2, 
        device = device, 
        optimizer = optimizer, 
        batch_size = batch_size,
        agent_name = name
    ) for name in agent_names}

    env_dume_def = {
        "max_cycles" : env_metadata["max_cycles"],
        "num_agents" : len(agent_names),
        "stack_size" : env_metadata["stack_size"],
        "single_frame_size" : (int(env_metadata["frame_size"][0]/2), 
                                int(env_metadata["frame_size"][1]/2))
    }

    # print(env_dume_def)

    if dume_in_use:
        dume_agents = {name : agent_mapping["dume"](
            batch_size = dume_batch_size, 
            lr = dume_lr, 
            gamma = gamma,
            optimizer = dume_optimizer, 
            agent_name = name,
            epoches = dume_epoches,
            env_dict = env_dume_def, 
            device = device
        ) for name in agent_names}
    
    # Log dict

    main_log = {
        "step" : [],
        "first_0" : [],
        "second_0" : [],
        "third_0" : [],
        "fourth_0" : []
    }
    
    # Training
    for ep in trange(episodes):

        with torch.no_grad():

            next_obs = output_env.reset(seed=None)

            # temporary buffer
            curr_act_buffer = {agent : torch.zeros(1, 1) for agent in agent_names}
            prev_act_buffer = {agent : torch.zeros(1, 1) for agent in agent_names}
            curr_rew_buffer = {agent : torch.zeros(1, 1) for agent in agent_names}
            prev_rew_buffer = {agent : torch.zeros(1, 1) for agent in agent_names}

            for step in range(env_metadata["max_cycles"]):

                curr_obs = batchify_obs(next_obs, device)[0].view(
                    -1, 
                    env_metadata["stack_size"],
                    env_metadata["frame_size"][0], 
                    env_metadata["frame_size"][1]
                )                

                # print(curr_obs.shape)

                agent_curr_obs = wardlord_coordinate_obs(curr_obs, p_size=p_size)

                # for agent in agent_curr_obs:
                #     print(type(agent_curr_obs[agent]))
                #     agent_obs = agent_curr_obs[agent][0].permute(-1, 1, 0).numpy()
                #     cv.imshow(agent, agent_obs)
                #     cv.waitKey(0)                  

                for agent in agent_names:
                    base_agent_curr_obs = agent_curr_obs[agent] # get obs specific to agent

                    obs_merges = {
                        agent : base_agent_curr_obs
                    } # Create obs dict with restricted view

                    for others in agent_names:
                        if others != agent:
                            # Add other agent information to their dume memory
                            dume_agents[others].add_memory(
                                agent_curr_obs[others], 
                                curr_act_buffer[others], 
                                curr_rew_buffer[others])

                            # print(agent_curr_obs[others].shape)
                            # break

                            # Get predicted information by dume
                            pred_obs, \
                            pred_act, \
                            pred_rew, \
                            skill_embedding, \
                            task_embedding = dume_agents[others](
                                curr_obs = agent_curr_obs[others].to(device=device, dtype=torch.float),
                                curr_act = curr_act_buffer[others].to(device=device, dtype=torch.float),
                                prev_act = prev_act_buffer[others].to(device=device, dtype=torch.float),
                                prev_rew = prev_rew_buffer[others].to(device=device, dtype=torch.float)
                            )

                            # add others' predicted obs to obs_dict
                            obs_merges[others] = pred_obs
                    
                    # Merge Observation
                    base_agent_merge_obs = wardlord_partial_obs_merge(
                        obs_merges = obs_merges,
                        frame_size = tuple(env_metadata["frame_size"]),
                        stack_size = env_metadata["stack_size"],
                        p_size = p_size
                    )                    

                    # Make action
                    action = main_algo_agents[agent].select_action(
                        base_agent_merge_obs.to(device=device, dtype=torch.float)
                    )

                    # Update buffer
                    curr_act_buffer[agent] = torch.Tensor([[action]])
                
                # Extract action from buffer
                
                actions = {
                    agent : int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                }

                next_obs, rewards, terms, truncs, infos = output_env.step(actions) # Update Environment

                # Logging
                main_log["step"].append(step)
                main_log["first_0"].append(rewards["first_0"])
                main_log["second_0"].append(rewards["second_0"])
                main_log["third_0"].append(rewards["third_0"])
                main_log["fourth_0"].append(rewards["fourth_0"])

                # Update Main Algo Memory
                for agent in rewards:
                    main_algo_agents[agent].insert_buffer(rewards[agent], terms[agent])

                # Reupdate buffer
                prev_act_buffer = curr_act_buffer
                prev_rew_buffer = curr_rew_buffer
                curr_rew_buffer = {
                    agent : torch.Tensor([[rewards[agent]]]) for agent in rewards
                }

                # print(curr_rew_buffer)
                # return

                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

        # Update Main Policy and DUME 
        for agent in agent_names:
            main_algo_agents[agent].update()
            main_algo_agents[agent].export_log(rdir = log_agent_dir, ep = ep) # Save main algo log
            main_algo_agents[agent].model_export(rdir = model_agent_dir) # Save main algo model
            print("\n")
            dume_agents[agent].update()
            dume_agents[agent].export_log(rdir = log_agent_dir, ep = ep) # Save dume log
            dume_agents[agent].model_export(rdir = model_agent_dir) # Save dume model
            print("\n")
    
        # Save main log
        main_log_path = main_log_dir + f"/{ep}.csv"
        main_log_df = pd.DataFrame(main_log)
        main_log_df.to_csv(main_log_path)

def valid():
    pass
