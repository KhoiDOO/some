import torch
import os
import torch.distributed as dist

class DistributeManager:
    def __init__(self, args) -> None:
        self._check_device_index(args.device_index)
        print()
        print("="*10, "CUDA DISTRIBUTION INFO", "="*10)
        print('| distributed init (rank {}): {}, gpu {}'.format(
            args.dist_rank, args.dist_url, args.device_index), flush=True)
        dist.init_process_group(backend=args.dist_be, init_method=args.dist_url,
                                            world_size=args.dist_ws, rank=args.dist_rank)
        dist.barrier()
        self._setup_for_distributed(args.dist_rank == 0)
        print("="*10, "CUDA DISTRIBUTION INFO", "="*10)
        print()
    
    def _set_os_env_var(self, args):
        os.environ["RANK"] = args.dist_rank
    
    def _check_device_index(self, device_index):
        if device_index == None:
            raise Exception("Arg --device-index cannot be None")


    def _setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    def is_dist_avail_and_initialized(self):
        if not self._is_available():
            return False
        if not self._is_initialized():
            return False
        return True
    
    def _is_available(self):
        return dist.is_available()
    
    def _is_initialized(self):
        return dist.is_initialized()

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def _is_main_process(self):
        return self.get_rank() == 0

    def _save_on_master(self, *args, **kwargs):
        if self._is_main_process():
            torch.save(*args, **kwargs)

    def _init_distributed_mode(args):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()

            os.environ['RANK'] = str(args.rank)
            os.environ['LOCAL_RANK'] = str(args.gpu)
            os.environ['WORLD_SIZE'] = str(args.world_size)
        else:
            print('Not using distributed mode')
            return