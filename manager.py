import os
import json
from utils.mapping import *
from utils.fol_struc import run_folder_verify
from utils.batchify import *
import argparse
from datetime import datetime
import torch
import random
import numpy as np
from tqdm import trange
import pandas as pd
from copy import copy, deepcopy


class Training:
    def __init__(self, args: argparse.PARSER) -> None:
        
        # setup
        self.args = args

        self.current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        # seed setup
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # autotune setup
        torch.backends.cudnn.benchmark = True

        # device setup
        if torch.cuda.is_available():
            self.train_device = torch.device(type = "cuda", index = self.args.device_index)
            if self.args.buffer_device == "cuda":
                self.buffer_device = torch.device(type = "cuda", index = self.args.device_index)
            else:
                self.buffer_device = self.args.buffer_device
        else:
            self.train_device = "cpu"
            self.buffer_device = "cpu"
        
        # verify
        run_folder_verify(self.current_time)

        # setting path
        self.save_train_set_path = os.getcwd() + f"/run/train/{self.current_time}/settings/settings.json"

        # agent logging save_dir
        self.log_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/log"
        self.model_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/weights"

        # main experiment logging
        self.main_log_dir = os.getcwd() + f"/run/train/{self.current_time}/log/main"

        # Script
        self.script_filename = self.args.script
        self.script_path = os.getcwd() + f"/script/{self.script_filename}.json"

        with open(self.save_train_set_path, "w") as outfile:
            json.dump(vars(self.args), outfile)

        # Environment initialization
        self.output_env = env_mapping[self.args.env](stack_size=self.args.stack_size, 
                                                     frame_size=tuple(self.args.frame_size),
                                                     max_cycles=self.args.max_cycles, 
                                                     render_mode=self.args.render_mode,
                                                     parralel=self.args.parallel, 
                                                     color_reduc=self.args.color_reduction)

        # Agent names initialization
        self.agent_names = self.output_env.possible_agents
        self.args.action_dim = self.output_env.action_space(self.output_env.possible_agents[0]).n

        # Actor Critic Initialization
        self.main_algo_agents = {name: agent_mapping[self.args.agent](
            args = self.args,
            agent_name = name
        ) for name in self.agent_names}

        # Environment Params Dictionary for IRG
        self.env_irg_def = {
            "max_cycles": self.args.max_cycles,
            "num_agents": len(self.agent_names),
            "stack_size": self.args.stack_size,
            "single_frame_size": (int(self.args.frame_size[0] / 2),
                                  int(self.args.frame_size[1] / 2))
        }

        # IRG initialization
        # if self.args.irg_in_use:
        #     self.irg_agents = {name: agent_mapping["irg"](
        #         batch_size = self.args.irg_batch_size,
        #         lr = self.args.irg_lr,
        #         gamma = self.args.gamma,
        #         optimizer = self.args.irg_optimizer,
        #         agent_name = name,
        #         epochs = self.irg_epochs,
        #         env_dict = self.env_irg_def,
        #         train_device = self.train_device,
        #         buffer_device = self.buffer_device,
        #         merge_loss = self.irg_merge_loss, 
        #         save_path = self.model_agent_dir,
        #         backbone_scale = self.irg_backbone,
        #         round_scale = self.irg_round_scale
        #     ) for name in self.agent_names}

    def train(self):
        if self.args.train_type == "pong-algo-only":
            self.pong_algo_only(max_reward=self.args.max_reward)
        elif self.args.train_type == "pong-irg-only":
            self.pong_irg_only(agent_name=self.args.agent_choose)
        elif self.args.train_type == "pong-irg-algo":
            self.pong_irg_algo(max_reward=self.args.max_reward)
    
    def main_log_init(self):
        base_dict = {
            "step" : []
        }

        for agent in self.agent_names:
            base_dict[agent] = []
        
        return base_dict
    
    def pong_irg_algo(self, max_reward = None):
        if not self.args.script:
            raise Exception("script arg cannot be None in this mode")
        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")
        
        script_dict = json.load(open(self.script_path))

        irg_weight_paths = {
            agent: os.getcwd() + f"/run/train/{script_dict[agent]}/weights/irg_{agent}.pt" for agent in
            self.agent_names
        }

        for agent in self.agent_names:
            if self.irg_in_use:
                self.irg_agents[agent].brain.load_state_dict(
                    torch.load(irg_weight_paths[agent], map_location=self.train_device)
                )
        
        for ep in trange(self.episodes):

            reward_log = self.main_log_init()

            win_log = self.main_log_init()

            reward_win = {
                agent : 0 for agent in self.agent_names
            }

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                curr_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.max_cycles):
                    
                    try:
                        curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1,
                            self.stack_size,
                            self.frame_size[0],
                            self.frame_size[1]
                        )
                    except:
                        break

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent]  # get obs specific to agent

                        obs_merges = {
                            agent: base_agent_curr_obs
                        }  # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                predict_obs, _ = self.irg_agents[others](
                                    curr_obs=agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act=curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act=prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew=prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = predict_obs

                        # Merge Observation
                        base_agent_merge_obs = env_parobs_merge_mapping[self.env_name](
                            obs_merges=obs_merges,
                            frame_size=tuple(self.frame_size),
                            stack_size=self.stack_size,
                            p_size=self.p_size
                        )

                        # Make action
                        action = self.main_algo_agents[agent].make_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])

                    # Extract action from buffer

                    actions = {
                        agent: int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, _, _ = self.output_env.step(actions)  # Update Environment

                    # Update termination
                    terms = {
                        agent : True if rewards[agent] == 1 or rewards[agent] == -1 else False for agent in self.agent_names
                    }

                    # Update win
                    for agent_name in self.agent_names:
                        if rewards[agent_name] == -1:
                            reward_win[agent_name] += 1
                    
                    # Inverse Reward
                    if self.inverse_reward:
                        for agent in self.agent_names:
                            rewards[agent] = 0 - rewards[agent]

                    # Fix Reward
                    if self.fix_reward:                        
                        for agent in self.agent_names:
                            if rewards[agent] == 0:
                                rewards[agent] = max_reward/(step + 1)
                            elif rewards[agent] == -1:
                                rewards[agent] = -10
                            else:
                                rewards[agent] = 10
                    
                    reward_log["ep"].append(ep)
                    reward_log["step"].append(step)
                    for agent in self.agent_names:
                        reward_log[agent].append(rewards[agent])

                    # Re-update buffer
                    prev_act_buffer = curr_act_buffer
                    prev_rew_buffer = curr_rew_buffer
                    curr_rew_buffer = {
                        agent: torch.Tensor([[rewards[agent]]]) for agent in rewards
                    }          

                    # Update buffer for algo actor critic
                    """
                        For sample-efficient, the agent should selectively choose the sample
                    for the learning process:
                    - At each sampling round, winning agent (scores = 0) will do not insert 
                    the observation into buffers.
                    """
                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], terms[agent])
                
                # Update no. win in episode
                win_log["ep"].append(ep)
                win_log["step"].append(step)
                for agent in self.agent_names:
                    win_log[agent].append(reward_win[agent])
                    
            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)
                self.main_algo_agents[agent].model_export(rdir=self.model_agent_dir)

            reward_log_path = self.main_log_dir + f"/{ep}_reward_log.parquet"
            reward_log_df = pd.DataFrame(reward_log)
            reward_log_df.to_parquet(reward_log_path)

            win_log_path = self.main_log_dir + f"/{ep}_win_log.parquet"
            win_log_df = pd.DataFrame(win_log)
            win_log_df.to_parquet(win_log_path)

    def pong_irg_only(self, agent_name):
        if self.env_name != "pong":
            raise Exception(f"Env must be pong but found {self.env_name} instead")
        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")

        # Create Agent
        irg_agent = self.irg_agents[agent_name]

        # Buffer Memory

        for _ in trange(self.episodes):

            with torch.no_grad():

                self.output_env.reset(seed=None)

                obs_lst, act_lst, rew_lst = [], [], []

                for step in range(self.max_cycles):

                    actions = {a: self.output_env.action_space(a).sample() for a in self.output_env.possible_agents}

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    action = torch.tensor([actions[agent_name]])

                    act_lst.append(action)

                    reward = torch.tensor([rewards[agent_name]])

                    rew_lst.append(reward)

                    try:
                        curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1,
                            self.stack_size,
                            self.frame_size[0],
                            self.frame_size[1]
                        )
                    except:
                        break

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    obs_used = agent_curr_obs[agent_name][0]

                    obs_lst.append(obs_used)

                obs_stack = torch.stack(obs_lst)
                act_stack = torch.stack(act_lst)
                rew_stack = torch.stack(rew_lst)

                irg_agent.add_memory(obs=obs_stack.to(device=self.buffer_device), 
                                        acts=act_stack.to(device=self.buffer_device), 
                                        rews=rew_stack.to(device=self.buffer_device))

        # irg training
        irg_agent.update()
        irg_agent.export_log(rdir=self.log_agent_dir, ep="all")
        
    def pong_algo_only(self, max_reward = 100):

        # Train and logging in parallel
        if self.args.env != "pong":
            raise Exception(f"Env must be pong but found {self.args.env} instead")
        
        next_obs = self.output_env.reset(seed=None)
        
        reward_log = self.main_log_init()

        win_log = self.main_log_init()

        reward_win = {
            agent : 0 for agent in self.agent_names
        }
        
        train_count = 1
        
        for step in trange(self.args.total_steps):
            
            try:
                curr_obs = batchify_obs(next_obs, self.buffer_device)[0]
            except:
                reward_log_path = self.main_log_dir + f"/{train_count}_reward_log.parquet"
                reward_log_df = pd.DataFrame(reward_log)
                reward_log_df.to_parquet(reward_log_path)

                win_log_path = self.main_log_dir + f"/{train_count}_win_log.parquet"
                win_log_df = pd.DataFrame(win_log)
                win_log_df.to_parquet(win_log_path)
                
                reward_log = self.main_log_init()

                win_log = self.main_log_init()

                reward_win = {
                    agent : 0 for agent in self.agent_names
                }
                
                next_obs = self.output_env.reset(seed=None)
                continue

            # Make action
            actions, actions_buffer, log_probs_buffer, obs_values_buffer = {}, {}, {}, {}

            with torch.no_grad():
                for agent in self.agent_names:
                    curr_act, curr_logprob, curr_obs_val = self.main_algo_agents[
                        agent
                        ].make_action(curr_obs.view(
                            -1,
                            self.args.stack_size,
                            self.args.frame_size[0],
                            self.args.frame_size[1]
                        ))                    

                    actions[agent] = curr_act.item()
                    actions_buffer[agent] = curr_act
                    log_probs_buffer[agent] = curr_logprob
                    obs_values_buffer[agent] = curr_obs_val

            next_obs, rewards, terms, _, _ = self.output_env.step(actions)  # Update Environment
            
            if len(list(terms.keys())) == 0:
                terms = {agent : False for agent in self.agent_names}

            # Update win                    
            for agent_name in self.agent_names:
                reward_win[agent_name] += rewards[agent_name] 

            # Inverse reward
            if self.args.inverse_reward:
                for agent in self.agent_names:
                    rewards[agent] = 0 - rewards[agent]

            # Fix reward
            if self.args.fix_reward:                        
                for agent in self.agent_names:
                    if rewards[agent] == 0:
                        rewards[agent] = max_reward/(step + 1)
                    elif rewards[agent] == -1:
                        rewards[agent] = -10
                    else:
                        rewards[agent] = 10
            
            reward_log["step"].append(step)
            for agent in self.agent_names:
                reward_log[agent].append(rewards[agent])
            
            if step == train_count * self.args.step:
                for agent in self.agent_names:
                    self.main_algo_agents[agent].update()
                    
                    if self.args.lr_decay:
                        self.main_algo_agents[agent].update_lr(step, self.args.total_steps)
                    if self.args.clip_decay:
                        self.main_algo_agents[agent].update_clip(step, self.args.total_steps)
                
                train_count += 1

            """
                For sample-efficient, the agent should selectively choose the sample
            for the learning process:
            - At each sampling round, winning agent (scores = 0) will do not insert 
            the observation into buffers.
            """
            for agent in self.agent_names:
                    self.main_algo_agents[agent].insert_buffer(obs = curr_obs, 
                                                                act = actions_buffer[agent], 
                                                                log_probs = log_probs_buffer[agent],
                                                                rew = rewards[agent],
                                                                obs_val = obs_values_buffer[agent],
                                                                term = terms[agent])
        
            # Update no. win in episode
            win_log["step"].append(step)
            for agent in self.agent_names:
                win_log[agent].append(reward_win[agent])