import os, sys
sys.path.append(os.getcwd())
import argparse
from agents.ppo.modules.buffer import RolloutBuffer
from agents.ppo.modules.backbone import *
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import trange
from random import uniform, choice
from torch import optim
import pandas as pd

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

backbone_mapping = {
    "siamese" : ActorCriticSiamese,
    "siamese-small" : ActorCriticSiameseSmall,
    "siamese-nano" : ActorCriticSiameseNano,
    "multi-head" : ActorCriticMultiHead,
    "multi-head-small" : ActorCriticMultiHeadSmall
}

def divide(data: list, chunk_size):
    lst = []
    for i in range(0, len(data), chunk_size):
        lst.append(data[i:i + chunk_size]) 
    return lst

def batch_split(data: list, chunk_size):
    batch_data = divide(data, chunk_size)

    prev_sub_data = None
    for idx, sub_data in enumerate(batch_data):
        if len(sub_data) > 1:
            batch_data[idx] = torch.squeeze(torch.stack(sub_data, dim=0))
        else: 
            batch_data[idx] = torch.squeeze(torch.stack(prev_sub_data, dim=0))
        prev_sub_data = sub_data

    return batch_data

class PPO:
    def __init__(self, stack_size:int = 4, action_dim:int = 6, lr_actor:float = 0.05, 
                lr_critic:float = 0.05, gamma:float = 0.99, K_epochs:int = 2, eps_clip:float = 0.2, 
                device:str = "cuda", optimizer:str = "Adam", batch_size:int = 16, 
                agent_name: str = None, backbone:str = "siamese", target_kl = 0.01,
                debug_mode = None):
        """Constructor of PPO

        Args:
            stack_size (int, optional): size of frames stack. Defaults to 4.
            action_dim (int, optional): number of possible action. Defaults to 6.
            lr_actor (float, optional): actor learning rate. Defaults to 0.0003.
            lr_critic (float, optional): critic learning rate. Defaults to 0.001.
            gamma (float, optional): discount factor. Defaults to 0.99.
            K_epochs (int, optional): number of self training epoch. Defaults to 2.
            eps_clip (float, optional): _description_. Defaults to 0.2.
            device (str, optional): type of device used for training. Defaults to "cpu".
            optimizer (str, optional): type of optimizer. Defaults to "Adam".
            batch_size (int, optional): Batch size for each training step. Defaults to 16.
            backbone (str, optional): Type of backbone using. Defaults to "siamese"
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.target_kl = target_kl
        self.K_epochs = K_epochs
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.agent_name = agent_name
        self.debug_mode = debug_mode
        
        self.buffer = RolloutBuffer()

        self.policy = backbone_mapping[backbone](stack_size = self.stack_size, num_actions = action_dim).to(device)

        self.actor_opt = opt_mapping[optimizer](self.policy.actor.parameters(), lr = lr_actor)
        self.critic_opt = opt_mapping[optimizer](self.policy.critic.parameters(), lr = lr_critic)

        self.policy_old = backbone_mapping[backbone](stack_size = self.stack_size, num_actions = action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.device = device

    def log_init(self):
        return {
            "epoch" : [],
            "actor_loss" : [],
            "critic_loss" : [],
            "kl" : []
        }
    
    def select_action(self, obs: torch.Tensor):
        """choose action from the policy
        this function automatically save observation, action predict and value of policy to the buffer
        the insert_buffer function has to be call after this function to avoid mismatch in buffer.

        Args:
            obs (torch.Tensor): observation of the current agent which has size of [None, stack_size, height, width]

        Returns:
            int: action
        """
        with torch.no_grad():
            action, action_logprob, obs_val = self.policy.act(obs.to(self.device, dtype=torch.float))
            
        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.obs_values.append(obs_val)

        return action.item()
    
    def make_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action, action_logprob, obs_val = self.policy.act(obs.to(self.device, dtype=torch.float))
        return action.item()

    def insert_buffer(self, single_reward: int, single_is_terminals: bool):
        """Insert reward and terminal information to the buffer

        Args:
            single_reward (int): reward gain from each step of env
            single_is_terminals (bool): is terminated or not
        """
        self.buffer.rewards.append(single_reward)
        self.buffer.is_terminals.append(single_is_terminals)
    
    def buffer_sample(self, sample_count = 125, obs_size = (64, 64)):
        """Sampling a random buffer for testing train function

        Args:
            sample_count (int, optional): Number of sample which will be added to the buffer. Defaults to 100.
            obs_size (tuple, optional): frame_size of the env. Defaults to (64, 64).
        """
        for _ in range(sample_count):
            random_obs = torch.randn((self.stack_size, obs_size[0], obs_size[1])).view(-1, 4, 64, 64)
            random_act = self.select_action(random_obs)

            self.buffer.rewards.append(uniform(-1, 1))
            self.buffer.is_terminals.append(choice([True, False]))
        
        old_obs = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_obs_values = torch.squeeze(torch.stack(self.buffer.obs_values, dim=0)).detach().to(self.device)

        print("old_obs: {}".format(old_obs.shape))
        print("old_actions: {}".format(old_actions.shape))
        print("old_logprobs: {}".format(old_logprobs.shape))
        print("old_obs_values: {}".format(old_obs_values.shape))
        print("old_rewards: {}".format(len(self.buffer.rewards)))
        print("old_is_terminals: {}".format(len(self.buffer.is_terminals)))
        
    def update(self):
        """Self updating policy of PPO algoirithm
        """
        # Log
        print(f"\nPPO Update for agent {self.agent_name}")

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = [torch.tensor(np.float32(reward)) for reward in rewards]

        # Batch Split
        obs_batch = batch_split(self.buffer.observations, self.batch_size)
        act_batch = batch_split(self.buffer.actions, self.batch_size)
        logprobs_batch = batch_split(self.buffer.logprobs, self.batch_size)
        obs_values_batch = batch_split(self.buffer.obs_values, self.batch_size)
        reward_batch = batch_split(rewards, self.batch_size)

        # Tracking
        self.log = self.log_init()

        # Optimize policy for K epochs
        for e in trange(self.K_epochs):

            # Loop through batch
            for idx in range(len(obs_batch)):

                # cal advantage
                advantages = reward_batch[idx].to(self.device) - obs_values_batch[idx].to(self.device)

                # Evaluation
                action_probs = self.policy.actor(obs_batch[idx].to(self.device)/255)
                dist = Categorical(logits=action_probs)

                logprobs = dist.log_prob(act_batch[idx].to(self.device))
                dist_entropy = dist.entropy()
                obs_values = self.policy.critic(obs_batch[idx].to(self.device)/255)

                # match obs_values tensor dimensions with rewards tensor
                obs_values = torch.squeeze(obs_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - logprobs_batch[idx].to(self.device))

                # Finding Surrogate Loss   
                obj = ratios * advantages
                obj_clip = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                actor_loss = -torch.min(obj, obj_clip).mean() - 0.01 * dist_entropy.mean()

                critic_loss = 0.5 * nn.MSELoss()(obs_values, reward_batch[idx].to(self.device))

                # Approx KL
                # approx_kl = (logprobs_batch[idx].to(self.device) - logprobs).mean()

                # Logging
                self.logging(
                    epoch=e, 
                    actor_loss = actor_loss.item(), 
                    critic_loss = critic_loss.item(), 
                    kl = approx_kl.item()
                )
                        
                # take gradient step
                # if approx_kl <= 1.5 * self.target_kl:
                    # self.actor_opt.zero_grad()
                    # actor_loss.backward()
                    # self.actor_opt.step()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                self.policy_old.load_state_dict(self.policy.state_dict())
            
        if self.debug_mode == None:
            pass
        elif self.debug_mode == 0:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")
        elif self.debug_mode == 1:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")

            if str(self.policy.state_dict()) == str(self.policy_old.state_dict()):
                print("Policy is updated")
            if str(self.policy.actor.state_dict()) == str(self.policy_old.actor.state_dict()):
                print("Actor is updated")
            if str(self.policy.critic.state_dict()) == str(self.policy_old.critic.state_dict()):
                print("Critic is updated")
        elif self.debug_mode == 2:
            print(f"Total step: {len(self.buffer.observations)}")
            print("Actor Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["actor_loss"]), max(self.log["actor_loss"]), sum(self.log["actor_loss"])/len(self.log["actor_loss"])
            ))
            print("Critic Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["critic_loss"]), max(self.log["critic_loss"]), sum(self.log["critic_loss"])/len(self.log["critic_loss"])
            ))
            print("KL Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["kl"]), max(self.log["kl"]), sum(self.log["kl"])/len(self.log["kl"])
            ))


        # clear buffer
        self.buffer.clear()

    def logging(self, epoch = None, actor_loss = None, critic_loss = None, kl = None):
        self.log["epoch"].append(epoch)
        self.log["actor_loss"].append(actor_loss)
        self.log["critic_loss"].append(critic_loss)
        self.log["kl"].append(kl)

    def export_log(self, rdir: str, ep: int, extension: str = ".parquet"):
        """Export log to file

        Args:
            rdir (str): folder for saving
            ep (int): current episode
            extension (str, optional): save file extension. Defaults to ".parquet".
        """
        sub_dir = rdir + "/ppo"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)    
        agent_sub_dir = sub_dir + f"/{self.agent_name}"
        if not os.path.exists(agent_sub_dir):
            os.mkdir(agent_sub_dir)

        filepath = agent_sub_dir + f"/{ep}{extension}"
        export_df = pd.DataFrame(self.log)

        if extension == ".parquet":
            export_df.to_parquet(filepath)
        elif extension == ".csv":
            export_df.to_csv(filepath)
        elif extension == ".pickle":
            export_df.to_pickle(filepath)
    
    def model_export(self, rdir: str):
        """Export model to file

        Args:
            dir (str): folder for saving model weights
        """
        sub_dir = rdir + "/ppo"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)    
        agent_sub_dir = sub_dir + f"/{self.agent_name}"
        if not os.path.exists(agent_sub_dir):
            os.mkdir(agent_sub_dir)
        
        filename = f"ppo_{self.agent_name}"
        filpath = agent_sub_dir + f"/{filename}.pt"
        torch.save(self.policy.state_dict(), filpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--backbone", type=str, choices=[
        "siamese", "siamese-small", "siamese-nano", "multi-head", "multi-head-small"
        ],
        help="Backbone", default="siamese-nano")
    parser.add_argument("--device", type=str, choices=[
        "cuda", "cpu"
        ],
        help="Device", default="cpu")
    parser.add_argument("--epoch", type=int,
        help="epochs", default=50)
    
    args = parser.parse_args()

    ppo = PPO(backbone=args.backbone, device=args.device, K_epochs=args.epoch)
    ppo.buffer_sample(sample_count = 124)
    # ppo.debug()
    ppo.update()
    # ppo.export_log(rdir = os.getcwd() + "/run", ep = 1)
    # ppo.model_export(rdir = os.getcwd() + "/run")