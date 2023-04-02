import os, sys
sys.path.append(os.getcwd())
import argparse

# Buffer
from agents.ppo.modules.buffer import RolloutBuffer
from agents.ppo.modules.backbone import *

# Main
import numpy as np
import torch
from torch import optim
import pandas as pd

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

backbone_mapping = {
    "multi-head-small" : ActorCriticMultiHeadSmall
}

class PPO:
    def __init__(self, args: argparse = None,
                 agent_name: str = None):

        self.args = args

        self.stack_size = args.stack_size
        self.action_dim = args.action_dim
        self.agent_name = agent_name
        self.gamma = args.gamma
        self.gae = args.gae
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.clip_coef = args.clip_coef
        self.clip_decay = args.clip_decay
        self.clip_low = args.clip_low
        self.clip_diff = self.clip_coef - self.clip_low
        self.K_epochs = args.epochs
        self.batch_size = args.bs
        self.debug = args.debug
        self.lr = args.lr 
        self.lr_decay = args.lr_decay
        self.lr_low = args.lr_low
        self.lr_diff = self.lr - self.lr_low
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.device_index)
        self.buffer = RolloutBuffer()
        self.policy = backbone_mapping[args.backbone](stack_size = self.stack_size, num_actions = self.action_dim).to(self.device)
        self.optimizer = opt_mapping[self.args.opt](self.policy.parameters(), lr=self.lr, eps=1e-5)
            
    def log_init(self):
        return {
            "epoch" : [],
            "actor_loss" : [],
            "critic_loss" : []
        }
    
    def make_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action, action_logprob, obs_val = self.policy.act(obs.to(self.device, dtype=torch.float))
        return action, action_logprob, obs_val

    def insert_buffer(self, obs: torch.Tensor, 
               act: torch.Tensor, 
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool):
        
        self.buffer.rb_obs.append(obs)
        self.buffer.rb_actions.append(act)
        self.buffer.rb_logprobs.append(log_probs)
        self.buffer.rb_values.append(obs_val)
        self.buffer.rb_rewards.append(rew)
        self.buffer.rb_terms.append(term)
        
    def update(self):
        # Log
        self._show_info()
        
        # to torch
        rb_obs = torch.stack(self.buffer.rb_obs).to(self.device)
        rb_rewards = torch.FloatTensor(self.buffer.rb_rewards).to(self.device)
        rb_terms = torch.FloatTensor(self.buffer.rb_terms).to(self.device)
        rb_actions = torch.FloatTensor(self.buffer.rb_actions).to(self.device)
        rb_logprobs = torch.FloatTensor(self.buffer.rb_logprobs).to(self.device)
        rb_values = torch.FloatTensor(self.buffer.rb_values).to(self.device)

        # Adv calculating
        end_step = rb_rewards.shape[0] - 1
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(self.device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + self.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + self.gamma * self.gae * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values
        
        # Tracking
        self.log = self.log_init()

        # Optimize policy for K epochs
        b_index = np.arange(len(self.buffer.rb_obs))
        clip_fracs = []
        for e in range(self.K_epochs):

            # Loop through batch
            np.random.shuffle(b_index)
            for start in range(0, len(self.buffer.rb_obs), self.batch_size):
                end = start + self.batch_size
                batch_index = b_index[start:end]

                # Evaluation
                _, newlogprob, entropy, value = self.policy.get_action_and_value(
                    rb_obs[batch_index], rb_actions.long()[batch_index]
                )

                logratio = newlogprob - rb_logprobs[batch_index]
                ratio = logratio.exp()

                # Approx_kl
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]
                
                # normalize advantaegs
                advantages = rb_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
                
                # Policy loss
                pg_loss1 = -rb_advantages[batch_index] * ratio
                pg_loss2 = -rb_advantages[batch_index] * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - rb_returns[batch_index]) ** 2
                v_clipped = rb_values[batch_index] + torch.clamp(
                    value - rb_values[batch_index],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - rb_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Total Loss
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Logging
                self.logging(epoch=e, actor_loss=pg_loss, critic_loss=v_loss)
                        
                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        if self.debug:
            self._debug(actor_loss=pg_loss, critic_loss=v_loss)

        # clear buffer
        self.buffer.clear()
    
    def _debug(self):
        if self.debug_mode:
            round = 0
            win_round = 0
            for r in self.buffer.rb_rewards:
                round += 1 if r == 1 or r == -1 else 0
                win_round += 1 if r == 1 else 0
            if round == 0:
                print(f"Total step: {len(self.buffer.rb_obs)} - Avg step: 0")
            else:
                print(f"Total step: {len(self.buffer.rb_obs)} - Avg step: {len(self.buffer.rb_obs)/round} - Win/Total: {win_round}/{round}")
            print("Actor Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["actor_loss"]), max(self.log["actor_loss"]), sum(self.log["actor_loss"])/len(self.log["actor_loss"]))
            )
            print("Critic Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["critic_loss"]), max(self.log["critic_loss"]), sum(self.log["critic_loss"])/len(self.log["critic_loss"]))
            )
            if self.lr_decay or self.clip_decay:
                print(f"Current LR: {self.lr} --- Current Clip: {self.clip_coef}")

    def update_lr(self, current_step, max_time_step):
        self.lr = (1-(current_step/max_time_step))*self.lr_diff + self.lr_low
        new_optim = opt_mapping[self.opt_tr](self.policy.parameters(), lr = self.lr)
        new_optim.load_state_dict(self.optimizer.state_dict())
        self.optimizer = new_optim
    
    def update_clip(self, current_step, max_time_step):
        self.clip_coef = (1-(current_step/max_time_step))*self.clip_diff + self.clip_low

    def logging(self, epoch = None, actor_loss = None, critic_loss = None):
        self.log["epoch"].append(epoch)
        self.log["actor_loss"].append(actor_loss)
        self.log["critic_loss"].append(critic_loss)

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

    def _show_info(self):
        print(f"\nPPO Update for agent {self.agent_name}")  