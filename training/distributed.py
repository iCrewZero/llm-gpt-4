import torch.distributed as dist

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()
