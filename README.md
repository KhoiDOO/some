# SOME - Shared Online Multi-agent Knowledge Exchange

# Clone
```
git clone https://github.com/DeepAI-Comm/some.git
```

# Local env setup
## Env creation
```
git clone https://github.com/DeepAI-Comm/some.git
cd path/to/folder/some
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
cd content/path/to/folder/some
!pip install pettingzoo supersuit autorom multi_agent_ale_py tinyscaler pyarrow
!git clone https://github.com/DeepAI-Comm/rom.git
%cd path/to/some/rom
!python -m atari_py.import_roms .
!AutoROM --accept-license
```

# Training helpdesk
```
cd content/path/to/folder/some
python main.py -h
```

# Training 
```
cd content/path/to/folder/some
python main.py --env "warlords" --render_mode rgb_array --stack_size 4 --max_cycles 124 --parrallel True --color_reduc True --ep 20623 --gamma 0.99 --view 1 --train_type experiment --agent_choose "first_0" --script sample --agent ppo --epoches 50 --bs 4 --actor_lr 0.001 --critic_lr 0.0005 --opt Adam --dume True --dume_epoches 50 --dume_bs 4 --dume_lr 0.005 --dume_opt Adam
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