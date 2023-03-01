import os, sys
import argparse
from datetime import datetime
import torch

from manager import Training

if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--env", type=str, default="warlords", choices=["warlords", "pong", "coop-pong"],
                        help="Environment used in training and testing")
    parser.add_argument("--render_mode", type=str, default=None, choices=["rgb_array", "human"],
                        help="Mode of rendering")
    parser.add_argument("--stack_size", type=int, default=4,
                        help="Number of stacking frames")
    parser.add_argument("--max_cycles", type=int, default=124,
                        help="Number of step in one episode")
    parser.add_argument("--frame_size", type=list, default=(64, 64),
                        help="Width and height of frame")
    parser.add_argument("--parallel", type=bool, default=True,
                        help="Process the environment in multi cpu core")
    parser.add_argument("--color_reduction", type=bool, default=True,
                        help="Reduce color to grayscale")
    parser.add_argument("--ep", type=int, default=2,
                        help="Total Episodes")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--view", type=float, default=1,
                        help="Area scale of partial observation varies in range of (0, 2)]")

    # Training
    parser.add_argument("--train_type", type=str, default="train-irg-only",
                        choices=["train-irg-only", "train-parallel", "train-algo-only", "experiment-dual",
                                 "experiment-algo", "pong-algo-only", "pong-irg-only", "pong-irg-algo"],
                        help="Type of training")
    parser.add_argument("--agent_choose", type=str, default="first_0",
                        choices=["first_0", "second_0", "third_0", "fourth_0", "paddle_0", "paddle_1"],
                        help="Agent chose for training, only available for irg or algo irg-only mode")
    parser.add_argument("--script", type=str,
                        help="Script includes weight paths to model, only needed in experiment mode, "
                             "detail in /script folder, create your_setting.json same as sample.json "
                             "for conducting custom experiment")
    parser.add_argument("--fix_reward", type=bool, default=False,
                        help="Make reward by step")
    parser.add_argument("--max_reward", type=int, default=100,
                        help="Max reward only use for pong-algo-only mode")
    parser.add_argument("--inverse_reward", type=bool, default=False,
                        help="change the sign of reward")
    parser.add_argument("--buffer_device", type=str, default="cpu",
                        help="Device used for memory replay")
    parser.add_argument("--device_index", type=int,
                        help="CUDA index used for training")

    # Agent
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo"],
                        help="Deep policy model architecture")
    parser.add_argument("--backbone", type=str, default="siamese", choices=[
        "siamese", "siamese-small", "multi-head", "multi-head-small"],
                        help="PPO Backbone")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--bs", type=int, default=20,
                        help="Batch size")
    parser.add_argument("--actor_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--critic_lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer")
    parser.add_argument("--debug_mode", type=int, default=None,
                        help="Debug mode")

    # irg
    parser.add_argument("--irg", type=bool, default=True,
                        help="Partial Observation Deep Policy")
    parser.add_argument("--irg_epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--irg_bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--irg_merge_loss", type=bool, default=True,
                        help="Take the gradient in the total loss instead of backwarding each loss separately")
    parser.add_argument("--irg_lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--irg_opt", type=str, default="Adam",
                        help="Optimizer for irg")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print(f"Environment: {args.env}")
    print(f"stack_size: {args.stack_size}")
    print(f"frame_size: {args.frame_size}")
    print(f"parallel: {args.parallel}")
    print(f"color_reduc: {args.color_reduction}")
    print(f"render_mode: {args.render_mode}")
    print(f"max_cycles: {args.max_cycles}")
    print(f"Total number of episodes {args.ep}")
    print(f"Discount Factor: {args.gamma}")
    print(f"Partial Observation Area Scale: {args.view}")

    print(f"Type of training experiment: {args.train_type}")
    print(f"Agent chose in irg-only mode: {args.agent_choose}")
    print(f"Script used in experiment mode: {args.script}")
    print(f"Fix reward function status: {args.fix_reward}")
    print(f"Buffer device: {args.buffer_device}")
    print(f"Cuda Index: {args.device_index}")

    print(f"Agent: {args.agent}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Actor Learning rate: {args.actor_lr}")
    print(f"Critic Learning rate: {args.critic_lr}")
    print(f"Discount factor: {args.gamma}")
    print(f"Total Episodes: {args.ep}")
    print(f"Optimizer: {args.opt}")

    print(f"irg: {args.irg}")
    print(f"irg Epochs: {args.irg_epochs}")
    print(f"irg Batch size: {args.irg_bs}")
    print(f"irg Learning rate: {args.irg_lr}")
    print(f"irg Optimizer: {args.irg_opt}")
    print("=" * 80)

    if torch.cuda.device_count() == 0 or not torch.cuda.is_available():
        print()
        print("="*10, "CUDA INFO", "="*10)
        print(f"Cuda is not available on this machine")
        print("="*10, "CUDA INFO", "="*10)
        print()
    elif args.device_index > torch.cuda.device_count():
        raise Exception(f"The device chose is higher than the number of available cuda device.\
            There are {torch.cuda.device_count()} but {args.device_index} chose instead")
    else:
        print()
        print("="*10, "CUDA INFO", "="*10)
        print(f"Total number of cuda: {torch.cuda.device_count()}")
        print(f"CUDA current index: {args.device_index}")
        print(f"CUDA device name: {torch.cuda.get_device_name(args.device_index)}")
        print(f"CUDA device address: {torch.cuda.device(args.device_index)}")
        print("="*10, "CUDA INFO", "="*10)
        print()

    train = Training(args=args)
    train.train()