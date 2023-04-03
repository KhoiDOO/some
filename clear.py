import os, sys
import shutil

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
            shutil.rmtree(curr_dir, ignore_errors=False, onerror=None)