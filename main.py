import os, sys
import argparse
from datetime import datetime

from manager import train

if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--env", type=str, default="warlords", choices=["warlords"],
        help="Environment used in training and testing")
    parser.add_argument("--env_meta", type=str, default="v1",
        help="Environment metadata version")
    parser.add_argument("--ep", type=int, default=2,
        help="Total Episodes")
    parser.add_argument("--gamma", type=float, default=0.99, 
        help="Discount factor")
    parser.add_argument("--view", type=float, default=1, 
        help="Area scale of partial observation varies in range of (0, 2)]")
    
    # Agent
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo"],
        help="Deep policy model architecture")
    parser.add_argument("--epoches", type=int, default=3,
        help="Number of epoch for training")
    parser.add_argument("--bs", type=int, default=20, 
        help="Batch size")
    parser.add_argument("--actor_lr", type=float, default=0.001, 
        help="learning rate")
    parser.add_argument("--critic_lr", type=float, default=0.0005, 
        help="learning rate")
    parser.add_argument("--opt", type=str, default="Adam",
        help="Otimizer")

    # Dume
    parser.add_argument("--dume", type=bool, 
        default=True, 
        # choices=["ppo"],
        help="Partial Observation Deep Policy")
    parser.add_argument("--dume_epoches", type=int, default=3,
        help="Number of epoch for training")
    parser.add_argument("--dume_bs", type=int, default=20, 
        help="Batch size")
    parser.add_argument("--dume_lr", type=float, default=0.005, 
        help="learning rate")
    parser.add_argument("--dume_opt", type=str, default="Adam",
        help="Otimizer for DUME")
    args = parser.parse_args()

    default_env_meta_path = os.getcwd() + f"/envs/{args.env}/metadata/{args.env_meta}.json"

    print("=" * 80)
    print("Summary of training process:")
    print(f"Environment: {args.env}")
    print(f"Environment Metadata path: {default_env_meta_path}")
    print(f"Total number of episodes {args.ep}")
    print(f"Discount Factor: {args.gamma}")
    print(f"Partial Observation Area Scale: {args.view}")

    print(f"Agent: {args.agent}")
    print(f"Epochs: {args.epoches}")
    print(f"Batch size: {args.bs}")
    print(f"Actor Learning rate: {args.actor_lr}")
    print(f"Critic Learning rate: {args.critic_lr}")
    print(f"Discount factor: {args.gamma}")
    print(f"Total Episodes: {args.ep}")
    print(f"Otimizer: {args.opt}")

    print(f"Dume: {args.dume}")
    print(f"Dume Epochs: {args.dume_epoches}")
    print(f"Dume Batch size: {args.dume_bs}")
    print(f"Dume Learning rate: {args.dume_lr}")
    print(f"Dume Otimizer: {args.dume_opt}")
    print("=" * 80)

    train(args)