"""Own allocations made by NeMo's manually captured conditional graph bodies."""
import contextlib
import torch


@contextlib.contextmanager
def decoder_graph_pool(model):
    """Keep child-stream temporaries alive across replay and allocator churn.

    Torch 2.11's CUDAGraph pool filter matches its root capture ID. NeMo's
    cudaStreamBeginCaptureToGraph child streams have separate IDs, so their
    temporary allocations otherwise escape that private pool. The enclosing
    thread-local MemPool catches those allocations; the root keeps its own pool.
    """
    comp = model.decoding.decoding.decoding_computer
    existed = '_full_graph_compile' in comp.__dict__
    original = comp._full_graph_compile
    comp.reset_cuda_graphs_state()
    pools = []

    def compile_graph():
        pool = torch.cuda.MemPool()
        # NeMo replaces the previous full_graph inside original(). Its pool
        # destructor calls emptyCache, which cannot run while ANY allocation
        # pool is recording. Hold the old pool until the new capture ends.
        pools.append(pool)
        with torch.cuda.use_mem_pool(pool, device=comp.state.device):
            original()
        comp.full_graph._conditional_memory_pool = pool
        pools[:] = [pool]

    comp._full_graph_compile = compile_graph
    try:
        yield
    finally:
        torch.cuda.synchronize()
        comp.reset_cuda_graphs_state()
        pools.clear()
        if existed:
            comp._full_graph_compile = original
        else:
            del comp._full_graph_compile
