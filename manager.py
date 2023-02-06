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

class Training:
    def __init__(self, args: argparse.PARSER) -> None:
        self.args = args

        # setup
        self.setup()

    def train(self):
        if self.args.train_type == "parallel":
            self.parallel()
        elif self.args.train_type == "single":
            self.single()
        elif self.args.train_type == "dumeonly":
            self.dume_only(agent_name=self.args.agent_choose)
        elif self.args.train_type == "algoonly":
            self.algo_only()  
        elif self.args.train_type == "experiment":
            self.experiment()          
    
    def setup(self):
        args_dict = vars(self.args)
    
        self.current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        # device setup
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_device = "cpu"

        # verify
        run_folder_verify(self.current_time)

        # setting path
        self.save_train_set_path = os.getcwd() + f"/run/train/{self.current_time}/settings/settings.json"
        self.save_valid_set_path = os.getcwd() + f"/run/val/{self.current_time}/settings/settings.json"

        # aggent logging save_dir
        self.log_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/log"
        self.model_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/weights"

        # main experiment logging
        self.main_log_dir = os.getcwd() + f"/run/train/{self.current_time}/log/main"

        with open(self.save_train_set_path, "w") as outfile:
            json.dump(args_dict, outfile)
        with open(self.save_valid_set_path, "w") as outfile:
            json.dump(args_dict, outfile)
        
        # Logging

        self.main_log = {
            "ep" : [],
            "step" : [],
            "first_0" : [],
            "second_0" : [],
            "third_0" : [],
            "fourth_0" : []
        }

        self.env_name = args_dict["env"]
        self.stack_size = args_dict["stack_size"]
        self.frame_size = args_dict["frame_size"]
        self.parrallel = args_dict["parrallel"]
        self.color_reduc = args_dict["color_reduc"]
        self.render_mode = args_dict["render_mode"]
        self.max_cycles = args_dict["max_cycles"]

        self.episodes = args_dict["ep"]
        self.gamma = args_dict["gamma"]
        self.p_size = args_dict["view"]

        self.agent_algo = args_dict["agent"]
        self.epoches = args_dict["epoches"]
        self.batch_size = args_dict["bs"]
        self.actor_lr = args_dict["actor_lr"]
        self.critic_lr = args_dict["critic_lr"]
        self.optimizer = args_dict["opt"]

        self.dume_in_use = args_dict["dume"]
        self.dume_epoches = args_dict["dume_epoches"]
        self.dume_batch_size = args_dict["dume_bs"]
        self.dume_lr = args_dict["dume_lr"]
        self.dume_optimizer = args_dict["dume_opt"]

        self.output_env = env_mapping[self.env_name](stack_size = self.stack_size, frame_size = tuple(self.frame_size),
                        max_cycles = self.max_cycles, render_mode = self.render_mode,
                        parralel = self.parrallel, color_reduc=self.color_reduc)

        self.agent_names = self.output_env.possible_agents
        
        self.main_algo_agents = {name : agent_mapping[self.agent_algo](
            stack_size = self.stack_size, 
            action_dim = self.output_env.action_space(self.output_env.possible_agents[0]).n, 
            lr_actor = self.actor_lr, 
            lr_critic = self.critic_lr, 
            gamma = self.gamma, 
            K_epochs = self.epoches, 
            eps_clip = 0.2, 
            device = self.train_device, 
            optimizer = self.optimizer, 
            batch_size = self.batch_size,
            agent_name = name
        ) for name in self.agent_names}

        self.env_dume_def = {
            "max_cycles" : self.max_cycles,
            "num_agents" : len(self.agent_names),
            "stack_size" : self.stack_size,
            "single_frame_size" : (int(self.frame_size[0]/2), 
                                    int(self.frame_size[1]/2))
        }

        if self.dume_in_use:
            self.dume_agents = {name : agent_mapping["dume"](
                batch_size = self.dume_batch_size, 
                lr = self.dume_lr, 
                gamma = self.gamma,
                optimizer = self.dume_optimizer, 
                agent_name = name,
                epoches = self.dume_epoches,
                env_dict = self.env_dume_def, 
                train_device = self.train_device,
                buffer_device = self.buffer_device
            ) for name in self.agent_names}

    def single(self):
        print("This feature is not available")

    def experiment(self):
        
        # Setup
        self.script_filename = self.args.script
        self.script_path = os.getcwd() + f"/script/{self.script_filename}.json"

        script_dict = json.load(open(self.script_path))

        algo_date = script_dict["algo"]

        dume_weight_paths = {
            agent : os.getcwd() + f"/run/train/{script_dict[agent]}/weights/dume_{agent}.pt" for agent in self.agent_names
        }

        algo_weight_paths = {
            agent : os.getcwd() + f"/run/train/{algo_date}/weights/ppo_{agent}.pt" for agent in self.agent_names
        }

        # Load weight

        for agent in self.agent_names:
            self.dume_agents[agent].brain.load_state_dict(torch.load(dume_weight_paths[agent]))
            self.main_algo_agents[agent].policy_old.load_state_dict(torch.load(algo_weight_paths[agent]))

        # Conduct experiment

        for ep in trange(self.episodes):

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                curr_act_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1, 
                        self.stack_size,
                        self.frame_size[0], 
                        self.frame_size[1]
                    )

                    agent_curr_obs = wardlord_coordinate_obs(curr_obs, p_size=self.p_size)

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent] # get obs specific to agent

                        # self.dume_agents[others].add_memory(
                        #             agent_curr_obs[others], 
                        #             curr_act_buffer[others], 
                        #             curr_rew_buffer[others])

                        obs_merges = {
                            agent : base_agent_curr_obs
                        } # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                pred_obs, \
                                pred_act, \
                                pred_rew, \
                                skill_embedding, \
                                task_embedding = self.dume_agents[others](
                                    curr_obs = agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act = curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act = prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew = prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = pred_obs
                        
                        # Merge Observation
                        base_agent_merge_obs = wardlord_partial_obs_merge(
                            obs_merges = obs_merges,
                            frame_size = tuple(self.frame_size),
                            stack_size = self.stack_size,
                            p_size = self.p_size
                        )                    

                        # Make action
                        action = self.main_algo_agents[agent].select_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])
                    
                    # Extract action from buffer
                    
                    actions = {
                        agent : int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, truncs, infos = self.output_env.step(actions) # Update Environment

                    rewards = {
                        agent : step if agent in terms else 0 for agent in self.agent_names
                    }

                    self.main_log["ep"].append(ep)
                    self.main_log["step"].append(step)
                    self.main_log["first_0"].append(rewards["first_0"])
                    self.main_log["second_0"].append(rewards["second_0"])
                    self.main_log["third_0"].append(rewards["third_0"])
                    self.main_log["fourth_0"].append(rewards["fourth_0"])
                
                    # for agent in rewards:
                    #     self.main_algo_agents[agent].insert_buffer(rewards[agent], True if agent in terms else False)

                    if len(list(terms.keys())) == 0:
                        break
                
            # for agent in self.agent_names:
            #     self.main_algo_agents[agent].update()
            #     self.main_algo_agents[agent].export_log(rdir = self.log_agent_dir, ep = ep) # Save main algo log
            #     self.main_algo_agents[agent].model_export(rdir = self.model_agent_dir) # Save main algo model
            #     print("\n")
            #     self.dume_agents[agent].update()
            #     self.dume_agents[agent].export_log(rdir = self.log_agent_dir, ep = ep) # Save dume log
            #     self.dume_agents[agent].model_export(rdir = self.model_agent_dir) # Save dume model
            #     print("\n")
        
        main_log_path = self.main_log_dir + f"/main_log.parquet"
        main_log_df = pd.DataFrame(self.main_log)
        main_log_df.to_parquet(main_log_path)
    
    def dume_only(self, agent_name = "first_0"):

        # Create Agent
        dume_agent = self.dume_agents[agent_name]

        # Buffer Memory
        
        for ep in trange(self.episodes):

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                obs_lst, act_lst, rew_lst = [], [], []

                for step in range(self.max_cycles):

                    actions = {a : self.output_env.action_space(a).sample() for a in self.output_env.possible_agents}

                    next_obs, rewards, terms, truns, info = self.output_env.step(actions)

                    rewards = {
                        agent : step if agent in terms else 0 for agent in self.agent_names
                    }

                    action = torch.tensor([actions[agent_name]])

                    act_lst.append(action)

                    reward = torch.tensor([rewards[agent_name]])

                    rew_lst.append(reward)

                    if len(list(terms.keys())) == 0:
                        break

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1, 
                        self.stack_size,
                        self.frame_size[0], 
                        self.frame_size[1]
                    ) 

                    agent_curr_obs = wardlord_coordinate_obs(curr_obs, p_size=self.p_size)

                    obs_used = agent_curr_obs[agent_name][0]

                    obs_lst.append(obs_used)

                obs_stack = torch.stack(obs_lst)
                act_stack = torch.stack(act_lst)
                rew_stack = torch.stack(rew_lst)

                dume_agent.add_memory(obs = obs_stack, acts = act_stack, rews = rew_stack)
        
        # Dume training
        dume_agent.update()
        dume_agent.export_log(rdir = self.log_agent_dir, ep = ep)
        dume_agent.model_export(rdir = self.model_agent_dir)

    def algo_only(self):

        for ep in range(self.episodes):

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                for step in trange(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1, 
                            self.stack_size,
                            self.frame_size[0], 
                            self.frame_size[1]
                        )
                    
                    actions = {
                        agent : self.main_algo_agents[agent].select_action(curr_obs) for agent in self.agent_names
                    }

                    next_obs, rewards, terms, truncs, infos = self.output_env.step(actions) # Update Environment

                    rewards = {
                        agent : step if agent in terms else 0 for agent in self.agent_names
                    }

                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], True if agent in terms else False)

                    if len(list(terms.keys())) == 0:
                        break

            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir = self.log_agent_dir, ep = ep) # Save main algo log
                self.main_algo_agents[agent].model_export(rdir = self.model_agent_dir) # Save main algo model

    def parallel(self): #Currently wrong

        print("Currently this function is not available")

        return

        # Training
        for ep in trange(self.episodes):

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                # temporary buffer
                curr_act_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent : torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.env_metadata["max_cycles"]):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1, 
                        self.env_metadata["stack_size"],
                        self.env_metadata["frame_size"][0], 
                        self.env_metadata["frame_size"][1]
                    )                

                    # print(curr_obs.shape)

                    agent_curr_obs = wardlord_coordinate_obs(curr_obs, p_size=self.p_size)

                    # for agent in agent_curr_obs:
                    #     print(type(agent_curr_obs[agent]))
                    #     agent_obs = agent_curr_obs[agent][0].permute(-1, 1, 0).numpy()
                    #     cv.imshow(agent, agent_obs)
                    #     cv.waitKey(0)                  

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent] # get obs specific to agent

                        obs_merges = {
                            agent : base_agent_curr_obs
                        } # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                # Add other agent information to their dume memory
                                self.dume_agents[others].add_memory(
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
                                task_embedding = self.dume_agents[others](
                                    curr_obs = agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act = curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act = prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew = prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = pred_obs
                        
                        # Merge Observation
                        base_agent_merge_obs = wardlord_partial_obs_merge(
                            obs_merges = obs_merges,
                            frame_size = tuple(self.env_metadata["frame_size"]),
                            stack_size = self.env_metadata["stack_size"],
                            p_size = self.p_size
                        )                    

                        # Make action
                        action = self.main_algo_agents[agent].select_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])
                    
                    # Extract action from buffer
                    
                    actions = {
                        agent : int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, truncs, infos = self.output_env.step(actions) # Update Environment

                    # Logging
                    self.main_log["step"].append(step)
                    self.main_log["first_0"].append(rewards["first_0"])
                    self.main_log["second_0"].append(rewards["second_0"])
                    self.main_log["third_0"].append(rewards["third_0"])
                    self.main_log["fourth_0"].append(rewards["fourth_0"])

                    # Update Main Algo Memory
                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], terms[agent])

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
            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir = self.log_agent_dir, ep = ep) # Save main algo log
                self.main_algo_agents[agent].model_export(rdir = self.model_agent_dir) # Save main algo model
                print("\n")
                self.dume_agents[agent].update()
                self.dume_agents[agent].export_log(rdir = self.log_agent_dir, ep = ep) # Save dume log
                self.dume_agents[agent].model_export(rdir = self.model_agent_dir) # Save dume model
                print("\n")
        
            # Save main log
            main_log_path = self.main_log_dir + f"/{ep}.csv"
            main_log_df = pd.DataFrame(self.main_log)
            main_log_df.to_csv(main_log_path)       
