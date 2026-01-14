import torch

class PageAllocator:
    def __init__(self, num_pages:int):
        self.free = list(range(num_pages))
    def alloc(self)->int:
        if not self.free: raise MemoryError("out of free pages")
        return self.free.pop()
    def free_page(self,p:int):
        self.free.append(p)

class KVState:
    __slots__=("pages","seq_len")
    def __init__(self):
        self.pages=[]
        self.seq_len=0
    def reset(self):
        self.pages.clear()
        self.seq_len=0

def append_token_state(state:KVState, allocator:PageAllocator, page_size:int):
    if state.seq_len % page_size==0:
        pid=allocator.alloc()
        state.pages.append(pid)
    state.seq_len+=1

def truncate_state(state:KVState, allocator:PageAllocator, new_len:int, page_size:int):
    old_pages=len(state.pages)
    new_pages=(new_len+page_size-1)//page_size
    for pid in state.pages[new_pages:]:
        allocator.free_page(pid)
    state.pages[:]=state.pages[:new_pages]
    state.seq_len=new_len

def gather_kv(K_pool:torch.Tensor,V_pool:torch.Tensor,state:KVState,upto_len:int,page_size:int):
    needed_pages=(upto_len+page_size-1)//page_size
    ks,vs=[],[]
    for pid in state.pages[:needed_pages]:
        ks.append(K_pool[pid].float())
        vs.append(V_pool[pid])
    K=torch.cat(ks,dim=2)
    V=torch.cat(vs,dim=2)
    if K.size(1)>upto_len:
        K=K[:,:upto_len,:];V=V[:,:upto_len,:]
    return K,V

def write_kv_token(K_pool:torch.Tensor,V_pool:torch.Tensor,state:KVState,k_tensor:torch.Tensor,v_tensor:torch.Tensor,page_size:int):
    token_idx=state.seq_len-1
    page_idx=token_idx//page_size
    offset=token_idx%page_size
    pid=state.pages[page_idx]
    K_pool[pid,offset]=k_tensor
    V_pool[pid,offset]=v_tensor
