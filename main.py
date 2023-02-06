import os, sys
import argparse
from datetime import datetime

from manager import Training

if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--env", type=str, default="warlords", choices=["warlords"],
        help="Environment used in training and testing")

    parser.add_argument("--render_mode", type=str, default=None, choices=["rgb_array", "human"],
        help="Mode of rendering")
    parser.add_argument("--stack_size", type=int, default=4,
        help="Number of stacking frames")
    parser.add_argument("--max_cycles", type=int, default=124,
        help="Number of step in one episode")
    parser.add_argument("--frame_size", type=list, default=(64, 64),
        help="Width and height of frame")
    parser.add_argument("--parrallel", type=bool, default=True,
        help="Process the environment in multi cpu core")
    parser.add_argument("--color_reduc", type=bool, default=True,
        help="Reduce color to grayscale")

    parser.add_argument("--ep", type=int, default=2,
        help="Total Episodes")
    parser.add_argument("--gamma", type=float, default=0.99, 
        help="Discount factor")
    parser.add_argument("--view", type=float, default=1, 
        help="Area scale of partial observation varies in range of (0, 2)]")
    
    # Training
    parser.add_argument("--train_type", type=str, default="dumeonly", 
        choices=["dumeonly", "parallel", "algoonly", "experiment"],
        help="Type of training")       
    parser.add_argument("--agent_choose", type=str, default="first_0", 
        choices=["first_0", "second_0", "third_0", "fourth_0"],
        help="Agent choosed for training, only availabel for dume or algo dumeonly mode")  
    parser.add_argument("--script", type=str, default="sample", 
        help="Script includes weight paths to model, only needed in experiment mode") 
    parser.add_argument("--fix_reward", type=bool, default=False, 
        help="Make reward by step") 
     
    
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

    print("=" * 80)
    print("Summary of training process:")
    print(f"Environment: {args.env}")
    print(f"stack_size: {args.stack_size}")
    print(f"frame_size: {args.frame_size}")
    print(f"parrallel: {args.parrallel}")
    print(f"color_reduc: {args.color_reduc}")
    print(f"render_mode: {args.render_mode}")
    print(f"max_cycles: {args.max_cycles}")

    print(f"Total number of episodes {args.ep}")
    print(f"Discount Factor: {args.gamma}")
    print(f"Partial Observation Area Scale: {args.view}")
    print(f"Type of training experiment: {args.train_type}")

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

    train = Training(args=args)
    train.train()