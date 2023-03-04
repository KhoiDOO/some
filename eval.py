import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
import argparse
from glob import glob

def extract_all():
    raise NotImplementedError

def extract(current_time:str = None):
    run_dir = os.getcwd() + f"/run"

    train_dir = run_dir + "/train"
    val_dir = run_dir + "/val"

    current_time_dir = train_dir + f"/{current_time}"

    log_dir = current_time_dir + "/log"

    sub_dirs = os.listdir(log_dir)

    if "main" in sub_dirs:
        pass
    if "irg" in sub_dirs:
        pass
    if "ppo" in sub_dirs:
        ppo_dir = log_dir + "/ppo"

        ppo_sub_dirs = glob(ppo_dir + "/*")

        for agent_dir in ppo_sub_dirs:
            logs = glob(agent_dir + "/*.parquet")
            log_dfs = [pd.read_parquet(x) for  x in logs]
            df = pd.concat(log_dfs)

            cols = df.columns
            for col in cols:
                if col != "epoch":
                    plt.figure()
                    sns.lineplot(x = "epoch", y = col, data = df)
                    agent = agent_dir.split("/")[-1]
                    fig_path = val_dir + f"/ppo_{agent}_{col}"
                    plt.savefig(fig_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--all", type=bool, help="gen for all")
    parser.add_argument("--y", type=str, help="year")
    parser.add_argument("--m", type=str, help="month")
    parser.add_argument("--d", type=str, help="day")
    parser.add_argument("--h", type=str, help="hour")
    parser.add_argument("--mn", type=str, help="minute")
    parser.add_argument("--s", type=str, help="second")
    parser.add_argument("--cus_name", type=str, help="Customied name of sub dir")

    args = parser.parse_args()

    if args.all == True:
        extract_all()
    else:
        time_chose = f"{args.m}-{args.d}-{args.y}-{args.h}-{args.mn}-{args.s}"

        extract(current_time=time_chose)

