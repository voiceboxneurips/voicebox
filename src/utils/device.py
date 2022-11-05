import torch
import torch.nn as nn

from typing import OrderedDict, Dict, Any, TypeVar, Union

################################################################################
# Utilities for single/multi-GPU training
################################################################################


class DataParallelWrapper(nn.DataParallel):
    """Extend DataParallel class to allow full method/attribute access"""
    def __getattr__(self, name):

        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False):
        """Avoid `module` prefix in saved weights"""
        return self.module.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars
        )

    def load_state_dict(self,
                        state_dict: OrderedDict[str, torch.Tensor],
                        strict: bool = True):
        """Avoid `module` prefix in saved weights"""
        self.module.load_state_dict(state_dict, strict)


def get_cuda_device_ids():
    """Fetch all available CUDA devices"""
    return list(range(torch.cuda.device_count()))


def wrap_module_multi_gpu(m: nn.Module, device_ids: list):
    """Implement data parallelism for arbitrary Module objects."""

    if len(device_ids) < 1:
        return m

    elif isinstance(m, DataParallelWrapper):
        return m

    else:
        return DataParallelWrapper(
            module=m,
            device_ids=device_ids
        )


def unwrap_module_multi_gpu(m: nn.Module, device: Union[str, int, torch.device]):

    if isinstance(m, DataParallelWrapper):
        return m.module.to(device)
    else:
        return m.to(device)


def wrap_attack_multi_gpu(m: nn.Module, device_ids: list):
    """
    Implement data parallelism for attack objects, including stored Pipeline
    and Perturbation instances that may be accessed outside of `forward()`
    """

    if len(device_ids) < 1:
        return m

    if hasattr(m, 'pipeline') and isinstance(m.pipeline, nn.Module):
        m.pipeline = wrap_pipeline_multi_gpu(m.pipeline, device_ids)

    if hasattr(m, 'perturbation') and isinstance(m.perturbation, nn.Module):
        m.perturbation = wrap_module_multi_gpu(m.perturbation, device_ids)

    # scale batch size to number of devices
    if hasattr(m, 'batch_size'):
        m.batch_size *= len(device_ids)

    return m


def unwrap_attack_multi_gpu(m: nn.Module, device: Union[str, int, torch.device]):
    """

    """
    if hasattr(m, 'pipeline') and isinstance(m.pipeline, DataParallelWrapper):
        m.pipeline = unwrap_module_multi_gpu(m.pipeline, device)

    if hasattr(m, 'perturbation') and isinstance(m.perturbation, DataParallelWrapper):
        m.perturbation = unwrap_module_multi_gpu(m.perturbation, device)

    # scale batch size to number of devices
    if hasattr(m, 'batch_size'):
        m.batch_size = m.batch_size // len(get_cuda_device_ids())

    return m


def wrap_pipeline_multi_gpu(m: nn.Module, device_ids: list):
    """
    Implement data parallelism for Pipeline objects, including all intermediate
    stages that may be accessed outside of `forward()`
    """

    if len(device_ids) < 1:
        return m

    return wrap_module_multi_gpu(m, device_ids)


def unwrap_pipeline_multi_gpu(m: nn.Module, device: Union[str, int, torch.device]):
    """
    """
    return unwrap_module_multi_gpu(m, device)
