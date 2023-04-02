import os, sys
import argparse
from datetime import datetime
import torch
from beautifultable import BeautifulTable

from manager import Training

if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    # Check Mode
    parser.add_argument("--check", action='store_true',
                        help="ARGS Check")
    parser.add_argument("--cli", action='store_true',
                        help="Show full CLI")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    
    # Environment
    parser.add_argument("--env", type=str, default="pong", choices=["pong"],
                        help="Environment used in training and testing")
    parser.add_argument("--render_mode", type=str, default=None, choices=["rgb_array", "human"],
                        help="Mode of rendering")
    parser.add_argument("--stack_size", type=int, default=4,
                        help="Number of stacking frames")
    parser.add_argument("--max_cycles", type=int, default=20000,
                        help="Number of step in one episode")
    parser.add_argument("--frame_size", type=list, default=[84, 84],
                        help="Width and height of frame")
    parser.add_argument("--parallel", action='store_true',
                        help="Process the environment in multi cpu core")
    parser.add_argument("--color_reduction", action='store_true',
                        help="Reduce color to grayscale")
    parser.add_argument("--total_steps", type=int, default=1000,
                        help="Total Steps")
    parser.add_argument("--step", type=int, default=128,
                        help="step for training")
    parser.add_argument("--view", type=float, default=1,
                        help="Area scale of partial observation varies in range of (0, 2)]")

    # Training
    parser.add_argument("--train_type", type=str, default="pong-algo-only",
                        choices=["pong-algo-only", "pong-irg-only", "pong-irg-algo"],
                        help="Type of training")
    parser.add_argument("--agent_choose", type=str, default="first_0",
                        choices=["first_0", "second_0", "third_0", "fourth_0", "paddle_0", "paddle_1"],
                        help="Agent chose for training, only available for irg or algo irg-only mode")
    parser.add_argument("--script", type=str,
                        help="Script includes weight paths to model, only needed in experiment mode, "
                             "detail in /script folder, create your_setting.json same as sample.json "
                             "for conducting custom experiment")
    parser.add_argument("--fix_reward", action='store_true',
                        help="Make reward by step")
    parser.add_argument("--max_reward", type=int, default=100,
                        help="Max reward only use for pong-algo-only mode")
    parser.add_argument("--inverse_reward", action='store_true',
                        help="change the sign of reward")
    parser.add_argument("--buffer_device", type=str, default="cpu",
                        help="Device used for memory replay")
    parser.add_argument("--device_index", 
                        type=int,
                        help="CUDA index or indices used for single training")

    # Agent
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo"],
                        help="Deep policy model architecture")
    parser.add_argument("--backbone", type=str, default="multi-head-small", choices=["multi-head-small"],
                        help="PPO Backbone")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--bs", type=int, default=20,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00025,
                        help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae", type=float, default=0.95,
                        help="lambda factor")
    parser.add_argument("--ent_coef", type=float, default=0.1,
                        help="Entropy clip value")
    parser.add_argument("--vf_coef", type=float, default=0.1,
                        help="Entropy clip value")
    parser.add_argument("--clip_coef", type=float, default=0.1,
                        help="Entropy clip value")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer")
    parser.add_argument("--debug", action = "store_true",
                        help="Debug mode")
    parser.add_argument("--lr_decay", action='store_true',
                        help="Learning Rate Scheduler")
    parser.add_argument("--lr_low", type=float, default=float(1e-12),
                        help="Lowest learning rate achieved")
    parser.add_argument("--clip_decay", action='store_true',
                        help="Clip Decaying Scheduler")
    parser.add_argument("--clip_low", type=float, default=0.05,
                        help="Lowest clip range")

    # irg
    parser.add_argument("--irg", action='store_true',
                        help="Partial Observation Deep Policy")
    parser.add_argument("--irg_backbone", type=str, default="small", 
                        choices=["small", "normal"],
                        help="Backbone used in training")
    parser.add_argument("--irg_epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--irg_bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--irg_merge_loss", action='store_true',
                        help="Take the gradient in the total loss instead of backwarding each loss separately")
    parser.add_argument("--irg_lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--irg_opt", type=str, default="Adam",
                        help="Optimizer for irg")
    parser.add_argument("--irg_round_scale", type=int, default=2,
                        help="Number of number after comma in decimal")
    args = parser.parse_args()

    table = BeautifulTable(maxwidth=140, detect_numerics = False)
    table.rows.append([args.env, "train_type", args.train_type, "agent", args.agent, "IRG", str(args.irg)])
    table.rows.append([args.stack_size, "agent_choose", args.agent_choose, "backbone", args.backbone, "irg_epochs", args.irg_epochs])
    table.rows.append([args.frame_size, "script", args.script, "epochs", args.epochs, "irg_bs", args.irg_bs])
    table.rows.append([str(args.parallel), "fix_reward", str(args.fix_reward), "bs", args.bs, "irg_lr", args.irg_lr])
    table.rows.append([str(args.color_reduction), "buffer_device", args.buffer_device, "lr", args.lr, "irg_opt", args.irg_opt])
    table.rows.append([args.render_mode, "device_index", args.device_index, "lr_low", args.lr_low, "irg_merge_loss", str(args.irg_merge_loss)])
    table.rows.append([args.max_cycles, "", "", "opt", args.opt, "irg_backbone", args.irg_backbone])
    table.rows.append([args.total_steps, "", "", "gamma", args.gamma, "irg_round_scale", args.irg_round_scale])
    table.rows.append([args.view, "", "", "gae", args.gae, "", ""])
    table.rows.append([args.seed, "", "", "clip_coef", args.clip_coef, "", ""])
    table.rows.append(["", "", "", "vf_coef", args.vf_coef, "", ""])
    table.rows.append(["", "", "", "lr_decay", args.lr_decay, "", ""])
    table.rows.append(["", "", "", "debug", str(args.debug), "", ""])
    table.rows.append(["", "", "", "clip_decay", args.clip_decay, "", ""])
    table.rows.append(["", "", "", "clip_low", args.clip_low, "", ""])
    table.rows.header = ["env", "stack_size", "frame_size", "parallel", "color_reduc", "render_mode", "max_cycles", "ep", "view", "seed", "", "", "", "", ""]
    table.columns.header = ["ENV INFO", "", "TRAIN INFO", "", "AGENT INFO", "", "IRG INFO"]
    print(table)

    if not torch.cuda.is_available():
        print()
        print("="*10, "CUDA INFO", "="*10)
        print(f"Cuda is not available on this machine")
        print("="*10, "CUDA INFO", "="*10)
        print()
    elif not args.device_index == None:
        if args.device_index > torch.cuda.device_count():
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
    else:
        print("="*10, "CUDA INFO", "="*10)
        print("CUDA not in use")
        print("="*10, "CUDA INFO", "="*10)

    
    if args.check or args.cli:
        if args.cli:
            print()
            print("="*10, "CLI", "="*10)
            print("python / torchrun main.py", end=" ")
            for arg in vars(args):
                if getattr(args, arg) == True or getattr(args, arg) == False:
                    print(f"--{arg}", end=" ")
                else:
                    print(f"--{arg} {getattr(args, arg)}", end=" ")
            print()
            print("="*10, "CLI", "="*10)
            print()
    else:      
        train = Training(args=args)
        train.train()