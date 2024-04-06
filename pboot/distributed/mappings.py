import torch
import torch.distributed as dist


def _copy_tensor_attributes(src_tensor, tgt_tensor):
    tgt_tensor = tgt_tensor.to(
        dtype=src_tensor.dtype,
        device=src_tensor.device,
    )
    return tgt_tensor


def all_gather(tensor, group, dim=-1):
    world_size = dist.get_world_size(group=group)

    tensor_list = [
        torch.empty_like(tensor) for _ in range(world_size)
    ]
    dist.all_gather(
        tensor_list,
        tensor,
        group=group,
    )

    return torch.cat(tensor_list, dim=dim)


def all_reduce(tensor, group, op="sum"):
    if op == "avg":
        tensor = tensor / dist.get_world_size(group)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

    return tensor


def reduce_scatter(tensor, group, dim=-1):
    world_size = dist.get_world_size(group)

    tensor = tensor / world_size

    new_shape = tuple([
        tensor.shape[i] if i != dim else tensor.shape[i] // world_size
        for i in range(len(tensor.shape))
    ])
    output = _copy_tensor_attributes(tensor, torch.empty(new_shape))

    dist.reduce_scatter_tensor(output, tensor, op=dist.ReduceOp.SUM, group=group)

    return output


def broadcast(tensor, src, group):
    dist.broadcast(tensor, src, group=group)
    return tensor


def send(tensor, dst, group, async_op=False):
    if not async_op:
        dist.send(tensor, dst, group=group)
        return
    else:
        work = dist.isend(tensor, dst, group=group)
        return work


def recv(tensor, src, group, async_op=False):
    if not async_op:
        dist.recv(tensor, src, group=group)
        return
    else:
        work = dist.isend(tensor, src, group=group)
        return work


def async_send_and_recv(send_tensor, recv_src, send_dst, group):
    # Build send and recv
    recv_tensor = torch.empty_like(send_tensor)
    recv = dist.P2POp(dist.irecv, recv_tensor, peer=recv_src, group=group)
    send = dist.P2POp(dist.isend, send_tensor, peer=send_dst, group=group)

    work = dist.batch_isend_irecv([recv, send])

    for req in work:
        req.wait()

    return recv_tensor
