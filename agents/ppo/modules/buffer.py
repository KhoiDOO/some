import torch

class Buffer:
    def __init__(self) -> None:
        self.actions = None
        self.observations = None
        self.logprobs = None
        self.rewards = None
        self.obs_values = None
        self.is_terminals = None
    
    def clear(self):
        raise NotImplementedError

class RolloutBuffer(Buffer):
    def __init__(self) -> None:
        super().__init__()
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.obs_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions
        del self.observations
        del self.logprobs
        del self.rewards
        del self.obs_values
        del self.is_terminals

class TorchRolloutBuffer(Buffer):
    def __init__(self, 
                 device:torch.device = None,
                 ele_shapes = {
                    'act' : (1),
                    "obs" : (4, 64, 64),
                    "logprobs" : (),
                    "rewards" : (1),
                    "obs_values" : (1) 
                 }) -> None:
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = "cpu"
        
        if not ele_shapes:
            raise Exception("Arg ele_shapes cannot be None")
        elif not isinstance(ele_shapes, dict):
            raise Exception(f"Arg ele_shapes type must be dict but found {type(ele_shapes)} instead")

        self.actions = TorchTensorList(ele_shape=ele_shapes["act"], device=self.device)
        self.observations = TorchTensorList(ele_shape=ele_shapes["obs"], device=self.device)
        self.logprobs = TorchTensorList(ele_shape=ele_shapes["logprobs"], device=self.device)
        self.rewards = TorchTensorList(ele_shape=ele_shapes["rewards"], device=self.device)
        self.obs_values = TorchTensorList(ele_shape=ele_shapes["obs_values"], device=self.device)
        self.is_terminals = []
    
    def clear(self):
        return super().clear()

class TorchTensorList:
    def __init__(self, ele_shape:tuple = (), device:torch.device = None) -> None:
        if ele_shape == None:
            raise Exception("Arg ele_shape must not be None")
        elif not isinstance(ele_shape, tuple):
            raise Exception(f"Arg ele_shape type must be Tuple but found {type(ele_shape)} instead")
        if device:
            self.device = device
        else:
            self.device = "cpu"

        self.count = 0
        self.arr = torch.randn(ele_shape)[None, :].to(device = self.device)
    
    def __len__(self):
        return self.count
    
    def __setitem__(self, index):
        raise NotImplementedError
    
        if not 0 <= index < self.count:
            raise Exception("Torch List index out of range")
        else:
            pass
    
    def __getitem__(self):
        raise NotImplementedError
    
    def append(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError