import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    """
    AllGather that supports backward propagation.
    """
    @staticmethod
    def forward(ctx, tensor):
        # 1. Prepare output buffer
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        tensor = tensor.contiguous()
        # 2. Allocate output tensors
        # Note: We assume all tensors have the same shape as local tensor
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        # 3. All gather
        dist.all_gather(gathered_tensors, tensor)
        
        # 4. Concatenate
        # Return as a list first to match torch.distributed.all_gather's conceptual output
        # But for autograd, we usually return a single tensor (concatenated) 
        # or we have to handle list output in backward.
        # Let's return the concatenated tensor as it's easier to use.
        
        # However, to support restoring the list structure if needed,
        # we can just return the concatenated tensor.
        # But wait, we need to know where the local gradient goes.
        
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [world_size * N, D]
        # This is the gradient of Loss wrt Global Features on the current Rank.
        # But Global Features are shared across all Ranks.
        # We need to sum the gradients from all Ranks for each chunk.
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        grad_output = grad_output.contiguous()

        # Split gradients back to chunks: list of [N, D]
        chunks = list(grad_output.chunk(world_size, dim=0))
        
        # Prepare output buffer for the reduced gradient: [N, D]
        grad_input = torch.zeros_like(chunks[rank])
        
        # reduce_scatter: Sums chunks[i] from all ranks and puts it in rank i's output
        # Input: chunks (list of tensors)
        # Output: grad_input (tensor)
        dist.reduce_scatter(grad_input, chunks, op=dist.ReduceOp.SUM)
        
        # Return the accumulated gradient for the local tensor
        return grad_input

def all_gather_with_grad(tensor):
    """
    Performs all_gather on a tensor and enables gradient propagation.
    Input: [N, D]
    Output: [world_size * N, D]
    """
    if dist.is_available() and dist.is_initialized():
        return AllGather.apply(tensor)
    else:
        return tensor
