import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

if __name__ == "__main__":
    run_dir = os.getcwd() + "/run"
    
    if not os.path.exists(run_dir):
        exit()
    else:
        train_dir = run_dir + "/train"
        if not os.path.exists(train_dir):
            exit()
    
    sub_res_dirs = os.listdir(train_dir)
    
    for sub_dir in sub_res_dirs:
        curr_dir = os.path.join(train_dir, sub_dir)
        
        sub_main_dir = curr_dir + "/log/main"
        
        if not os.path.exists(sub_main_dir) or len(os.listdir(sub_main_dir)) == 0:
            continue
            
        rew_log_paths = glob(sub_main_dir + "/*_reward_log.parquet")
        
        rew_log_paths = sorted(rew_log_paths)
        
        rew_log_df = pd.concat([pd.read_parquet(x) for x in rew_log_paths])
        
        # plot two
        plt.figure()
        sns.lineplot(data = rew_log_df, x = "step", y = "first_0")
        sns.lineplot(data = rew_log_df, x = "step", y = "second_0")
        filename = curr_dir + "/reward_mv"
        plt.title("Reward Over Training Episode")
        plt.savefig(filename)
        plt.close()
        
        # plot first
        plt.figure()
        sns.lineplot(data = rew_log_df, x = "step", y = "first_0")
        filename = curr_dir + "/first_reward_mv"
        plt.title("Reward of First Agent Over Training Episode")
        plt.savefig(filename)
        plt.close()
        
        # plot second
        plt.figure()
        sns.lineplot(data = rew_log_df, x = "step", y = "second_0")
        filename = curr_dir + "/second_reward_mv"
        plt.title("Reward of Second Agent Over Training Episode")
        plt.savefig(filename)
        plt.close()
        
        # plot reward total
        rew_sum_dict = {
            "episode" : [],
            "first_0" : [],
            "second_0" : []
        }
        
        for idx, rew_log_path in enumerate(rew_log_paths):
            base_df = pd.read_parquet(rew_log_path)
            
            sum_base_df = base_df.sum()
            
            rew_sum_dict["episode"].append(idx)
            rew_sum_dict["first_0"].append(sum_base_df["first_0"])
            rew_sum_dict["second_0"].append(sum_base_df["second_0"])
        
        rew_sum_df = pd.DataFrame(rew_sum_dict)
        plt.figure()
        
        sns.lineplot(data = rew_sum_df, x = "episode", y = "first_0")
        sns.lineplot(data = rew_sum_df, x = "episode", y = "second_0")
        
        filename = curr_dir + "/reward_sum"
        plt.title("Sum of Reward over Training Procedure")
        plt.savefig(filename)
        plt.close()
        
        # plot avg step
        
        rew_avg_round_dict = {
            "episode" : [],
            "avg_step" : [],
        }
        
        for idx, rew_log_path in enumerate(rew_log_paths):
            base_df = pd.read_parquet(rew_log_path)
            
            count_base_df = pd.DataFrame(base_df.drop(["step"], axis = 1).value_counts(), columns=["count"])
            
            count_base_df.reset_index(inplace=True)   
            
            try:
                first_win = count_base_df[count_base_df["first_0"] == 1].iloc[0, -1]
            except:
                first_win = 0
            try:
                second_win = count_base_df[count_base_df["second_0"] == 1].iloc[0, -1]
            except:
                second_win = 0
            
            sum_round = first_win + second_win
            
            rew_avg_round_dict["episode"].append(idx)
            rew_avg_round_dict["avg_step"].append(base_df.shape[0]/sum_round)
        
        rew_avg_df = pd.DataFrame(rew_avg_round_dict)
        plt.figure()
        
        sns.lineplot(data = rew_avg_df, x = "episode", y = "avg_step", label="avg_step")
        
        filename = curr_dir + "/avg_step"
        plt.title("Average round performed in each episode")
        plt.savefig(filename)
        plt.close()