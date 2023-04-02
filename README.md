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

# Training
## Training helpdesk
```
cd content/path/to/folder/some
python main.py --help -h
```
### Training Parameter Check
```
python main.py --check
```

### Training CLI Check
```
python main.py --cli
```

## Single GPU Training 
```
cd content/path/to/folder/some

python main.py --env warlords --render_mode None --stack_size 4 --max_cycles 124 --frame_size (64, 64) --parallel True --color_reduction True --ep 2 --gamma 0.99 --view 1 --train_type train-irg-only --agent_choose first_0 --script None --fix_reward False --max_reward 100 --inverse_reward False --buffer_device cpu --device_index None --dist_ws 1 --dist_rank -1 --dist_url env:// --dist_be nccl --agent ppo --backbone siamese --epochs 1 --bs 20 --actor_lr 0.001 --critic_lr 0.0005 --eps_clip 0.2 --opt Adam --debug_mode None --exp_mem False --dist_buff False --dist_cap 5 --dist_learn False --dist_opt False --lr_decay False --lr_decay_mode 0 --lr_low 1e-12 --irg True --irg_backbone small --irg_epochs 1 --irg_bs 32 --irg_merge_loss True --irg_lr 0.005 --irg_opt Adam --irg_round_scale 2
```

## Distributed Training
```
cd content/path/to/folder/some

torchrun main.py --env warlords --render_mode None --stack_size 4 --max_cycles 124 --frame_size (64, 64) --parallel True --color_reduction True --ep 2 --gamma 0.99 --view 1 --train_type train-irg-only --agent_choose first_0 --script None --fix_reward False --max_reward 100 --inverse_reward False --buffer_device cpu --device_index None --dist_ws 1 --dist_rank -1 --dist_url env:// --dist_be nccl --agent ppo --backbone siamese --epochs 1 --bs 20 --actor_lr 0.001 --critic_lr 0.0005 --eps_clip 0.2 --opt Adam --debug_mode None --exp_mem False --dist_buff False --dist_cap 5 --dist_learn False --dist_opt False --lr_decay False --lr_decay_mode 0 --lr_low 1e-12 --irg True --irg_backbone small --irg_epochs 1 --irg_bs 32 --irg_merge_loss True --irg_lr 0.005 --irg_opt Adam --irg_round_scale 2
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