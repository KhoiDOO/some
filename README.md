# MT-MA-RL - Multi-task Multi-agent Reinforcement Learning

# Clone
```
git clone https://github.com/DeepAI-Comm/mt-ma-rl.git
```

# Local env setup
## Env creation
```
git clone https://github.com/DeepAI-Comm/mt-ma-rl.git
cd path/to/folder/mt-ma-rl
pythonx.x>=3.9 -m venv .env
```
## Activate Env
### Linux
```
source .env/bin/activate
```
### Window
```
.env/Script/activate
```
### Packages Setup
```
python -m pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
AutoRom
```

# Google Colab setup
```
cd content/path/to/folder/mt-ma-rl
!python -m pip install --upgrade pip
!pip install wheel
!pip install -r requirements.txt
!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
!AutoRom
```

# Training helpdesk
```
cd content/path/to/folder/mt-ma-rl
python main.py -h
```

# Training 
```
cd content/path/to/folder/mt-ma-rl
python main.py --env "warlords" --env_meta "v1" --ep 1000 --gamma 0.99 --view 1 --agent ppo --epoches 50 --bs 4 --actor_lr 0.001  --critic_lr 0.0005 --opt Adam --dume True --dume_epoches 50 --dume_bs 4 --dume_lr 0.005 --dume_opt Adam
```

# Validation - Not implemented
```
```

# Rendering - Not implemented
```
```

# Citation
```
@article{<articlename>,
  Title = {},
  Author = {},
  journal={},
  year={}
}
```