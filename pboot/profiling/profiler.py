import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity


def profile_function(
    func: callable,
    inputs: any,
    warmup_steps: int = 2,
    active_steps: int = 10,
    wait_steps: int = 1,
    profile_row_limit: int = 5,
    print_profile: bool = True,
    trace_save_dir: Optional[str] = None,
    trace_save_name: Optional[str] = None,
):
    # Silence kineto warnings
    os.environ["KINETO_LOG_LEVEL"] = "5"

    num_steps = wait_steps + warmup_steps + active_steps
    schedule = torch.profiler.schedule(
        wait=wait_steps,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )

    if trace_save_dir is not None:
        save_path = trace_save_dir
        if not os.path.isdir(save_path):
            os.path.makedirs(save_path, exist_ok=False)

        if trace_save_name is not None:
            save_path += f"/{trace_save_name}"
        else:
            save_path += f"/{func.__name__}"

        trace_handler = torch.profiler.tensorboard_trace_handler(save_path)
    else:
        trace_handler = None

    with torch.profiler.profile(
        schedule=schedule,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_steps):
            func(inputs)
            prof.step()

        if print_profile is True:
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                sort_variables = [
                    "self_cpu_time_total",
                    "self_cuda_time_total",
                    "self_cpu_memory_usage",
                    "self_cuda_memory_usage",
                ]
                key_avgs = prof.key_averages(group_by_input_shape=True)

                for sort_var in sort_variables:
                    print(key_avgs.table(sort_by=sort_var, row_limit=profile_row_limit))
