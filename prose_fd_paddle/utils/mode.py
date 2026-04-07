import os
import socket

import paddle


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.world_size = int(paddle.distributed.get_world_size())
    if params.world_size > 1:
        params.global_rank = int(os.environ["RANK"])
        params.local_rank = int(os.environ["LOCAL_RANK"])
        params.n_gpu_per_node = int(os.environ.get("NGPU", params.world_size))
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
    else:
        params.local_rank = 0
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    if params.multi_gpu:
        PREFIX = "%i - " % params.global_rank
        print(PREFIX + "Number of nodes: %i" % params.n_nodes)
        print(PREFIX + "Node ID        : %i" % params.node_id)
        print(PREFIX + "Local rank     : %i" % params.local_rank)
        print(PREFIX + "Global rank    : %i" % params.global_rank)
        print(PREFIX + "World size     : %i" % params.world_size)
        print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
        print(PREFIX + "Master         : %s" % str(params.is_master))
        print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
        print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
        print(PREFIX + "Hostname       : %s" % socket.gethostname())
    if not params.cpu:
        paddle.cuda.set_device(params.local_rank)
    if params.multi_gpu:
        print("Initializing PyTorch distributed ...")
        paddle.distributed.init_parallel_env()
