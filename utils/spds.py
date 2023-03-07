import torch

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
    
    def size(self):
        return self.arr.shape
        
    @staticmethod
    def _check_index(self, index):
        if index is None:
            raise Exception("index cannot be None")
        elif not isinstance(index, int):
            raise Exception(f"index must be an integer but found {type(index)} instead")
        elif not 0 <= index < self.count:
            return IndexError('index is out of bounds !')
    
    @staticmethod
    def _check_input(input):
        if input is None:
            raise Exception("node cannot be None")
        elif not isinstance(input, torch.Tensor):
            raise Exception(f"node must be an integer but found {type(input)} instead")
    
    def __setitem__(self, index: int , x: torch.Tensor):
        raise NotImplementedError    

        self._check_index(index=index)
        self._check_input(input=x)
    
    def __getitem__(self, index: int):
        self._check_index(index=index)

        return self.arr[index]
    
    def append(self, x:torch.Tensor):
        if self.count == 0:
            self.arr = x[None, :]
            self.count += 1
        else:
            self.arr = torch.cat((self.arr, x[None, :]))
        
    def insert(self):
        raise NotImplementedError

if __name__ == "__main__":

    test_list = TorchTensorList(ele_shape=(1, 4, 32, 32))

    print(test_list.size())

    # for idx in range(9):
    #     x = torch.randn((idx+1, 4, 32, 32))        
    #     test_list.append(x)
    
    # print(test_list.size())