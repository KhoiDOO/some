import os, sys

def run_folder_verify(time):
    print("===verifying result folder===")
    print(f"current folder: {os.getcwd()}")

    run_dir = os.getcwd() + "/run"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    run_train_dir = run_dir + "/train"
    run_valid_dir = run_dir + "/val"

    for x in [run_train_dir, run_valid_dir]:
        if not os.path.exists(x):
            os.mkdir(x)
    
    time_dirs = [run_train_dir + f"/{time}", run_valid_dir + f"/{time}"]

    for x in time_dirs:
        if not os.path.exists(x):
            os.mkdir(x)
    
    sub_time_dirs = ["/log", "/settings", "/weights"]

    for time_dir in time_dirs:
        for sub_time_dir in sub_time_dirs:
            dir_mk = time_dir + sub_time_dir
            if not os.path.exists(dir_mk):
                os.mkdir(dir_mk)
            
            if sub_time_dir == "/log":
                main_log_dir = dir_mk + "/main"
                if not os.path.exists(main_log_dir):
                    os.mkdir(main_log_dir)
    
    print("=====Done=====\n")
