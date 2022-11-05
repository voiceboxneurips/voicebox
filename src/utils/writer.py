import torch
import torch.nn as nn
from copy import deepcopy
import contextlib
import math
import time
import logging
import sys
import json

from typing import Union, Dict, Any

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.constants import RUNS_DIR
from src.utils.filesystem import ensure_dir
from src.utils.plotting import *

################################################################################
# Logging utility with optional TensorBoard support
################################################################################


class Writer:
    """
    Handles file, console, and TensorBoard logging
    """
    def __init__(self,
                 root_dir: Union[str, Path] = RUNS_DIR,
                 name: str = None,
                 use_tb: bool = False,
                 log_iter: int = 100,
                 use_timestamp: bool = True,
                 log_images: bool = False,
                 **kwargs
                 ):
        """
        Configure logging.

        :param root_dir: root logging directory
        :param name: descriptive name for run
        :param use_tb: if True, use TensorBoard
        :param log_iter: iterations between logging
        """

        # generate run-specific name and create directory
        run_name = f'{name}'
        if use_timestamp:
            run_name += f'_{time.strftime("%m-%d-%H_%M_%S")}'
        self.run_dir = Path(root_dir) / run_name
        ensure_dir(self.run_dir)

        # prepare checkpoint directory
        self.checkpoint_dir = self.run_dir / 'checkpoints'

        # log to TensorBoard
        self.use_tb = use_tb
        self.log_iter = log_iter
        self.writer = SummaryWriter(
            log_dir=str(self.run_dir),
            flush_secs=20,
        ) if self.use_tb else None

        # log to console and file 'log.txt'
        self.logger = logging.getLogger(run_name)

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(
            logging.StreamHandler(sys.stdout)
        )
        self.logger.addHandler(
            logging.FileHandler(self.run_dir / 'log.txt')
        )

        # to avoid segmentation faults, it may be necessary to skip image
        # logging
        self.log_images = log_images

        # self.logger.info(f'Logging to {self.run_dir}')

        # disable Matplotlib logging
        logging.getLogger('matplotlib.font_manager').disabled = True

    def get_run_dir(self):
        return str(self.run_dir)

    def log_info(self, info: str):
        """
        Log generic statements
        """
        self.logger.info(info)

    def _dict_to_str(self, d: dict):
        """Recursively cast dictionary entries to strings"""

        d_out = {}

        for k, v in d.items():
            if isinstance(v, dict):
                d_out[k] = self._dict_to_str(v)
            elif not isinstance(v, (float, int, bool)):
                d_out[k] = str(v)
            else:
                d_out[k] = v
        return d_out

    def log_config(self,
                   config: Union[dict, str],
                   tag: str = "config",
                   path: Union[str, Path] = None):
        """Save config file for run, given dictionary"""

        path = path if path is not None else self.run_dir / f'{tag}.conf'

        with open(path, "w") as out_config:
            self.logger.info(f'Saving config to {path}')

            if isinstance(config, dict):
                config = self._dict_to_str(config)
                json.dump(config, out_config, indent=4)
            else:
                out_config.write(config)

    def log_scalar(self, x: torch.Tensor, tag: str, global_step: int = 0):
        """
        Log scalar
        """

        # only log at specified iterations
        if not self.log_iter or global_step % self.log_iter:
            return

        # log scalar to file and console
        self.logger.info(f'iter {global_step}\t{tag}: {x}')

        # if TensorBoard is enabled
        if self.use_tb:
            self.writer.add_scalar(f'{tag}', x, global_step=global_step)
            self.writer.flush()

    def log_logits(self,
                   x: torch.Tensor,
                   target: int = None,
                   tag: str = None,
                   global_step: int = 0):
        """
        Log class scores (logits)
        """

        # only log at specified iterations
        if not self.log_iter or global_step % self.log_iter:
            return

        # log plot to TensorBoard
        if self.use_tb and self.log_images:
            self.writer.add_image(
                f"{tag}", plot_logits(x, target),
                global_step=global_step
            )

            self.writer.flush()

    def log_audio(self,
                  x: torch.Tensor,
                  tag: str,
                  global_step: int = 0,
                  sample_rate: int = 16000,
                  scale: Union[int, float] = 1.0):
        """
        Given a single audio waveform, log a normalized recording, waveform
        plot, and spectrogram plot to TensorBoard
        """

        # only log at specified iterations and if TensorBoard is enabled
        if not self.log_iter or global_step % self.log_iter or not self.use_tb:
            return

        # normalize and log audio recording
        normalized = (scale / torch.max(
            torch.abs(x) + 1e-12, dim=-1, keepdim=True
        )[0]) * x * 0.95

        self.writer.add_audio(f"{tag}-audio",
                              normalized,
                              sample_rate=sample_rate,
                              global_step=global_step)

        if self.log_images:

            # log waveform
            self.writer.add_image(f"{tag}-waveform",
                                  plot_waveform(x, scale),
                                  global_step=global_step)

            # log spectrogram
            self.writer.add_image(f"{tag}-spectrogram",
                                  plot_spectrogram(x),
                                  global_step=global_step)

        # flush
        self.writer.flush()

    def log_norm(self,
                 x: torch.Tensor,
                 tag: str,
                 global_step: int = 0):
        """
        Plot norm of input tensor
        """

        # only log at specified iterations and if TensorBoard is enabled
        if not self.log_iter or global_step % self.log_iter or not self.use_tb:
            return

        # log norms
        norm_2 = torch.norm(x, p=2)
        norm_inf = torch.norm(x, p=float('inf'))
        self.writer.add_scalar(f'{tag}/norm-2', norm_2, global_step=global_step)
        self.writer.add_scalar(f'{tag}/norm-inf', norm_inf, global_step=global_step)

        self.writer.flush()

    def log_image(self, image: torch.Tensor, tag: str, global_step: int = 0):
        """
        Log image plot
        """

        if not self.log_images:
            return

        # only log at specified iterations and if TensorBoard is enabled
        if not self.log_iter or global_step % self.log_iter or not self.use_tb:
            return

        self.writer.add_image(
            tag,
            image,
            global_step
        )

        self.writer.flush()

    def log_filter(self,
                   amplitudes: torch.Tensor,
                   tag: str,
                   global_step: int = 0):
        """
        Plot filter controls
        """

        # only log at specified iterations and if TensorBoard is enabled
        if not self.log_iter or global_step % self.log_iter or not self.use_tb:
            return

        if self.log_images:

            self.writer.add_image(
                f'filter_controls/{tag}',
                plot_filter(amplitudes),
                global_step
            )

            self.writer.flush()

    @staticmethod
    def bytes_to_gb(size_bytes: int):
        """
        Code from: https://stackoverflow.com/a/14822210
        """
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def log_cuda_memory(self, device: int = 0):

        total_memory = self.bytes_to_gb(
            torch.cuda.get_device_properties(device).total_memory
        )
        reserved_memory = self.bytes_to_gb(
            torch.cuda.memory_reserved(device)
        )
        allocated_memory = self.bytes_to_gb(
            torch.cuda.memory_allocated(device)
        )
        free_memory = self.bytes_to_gb(
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
                device
            )
        )

        self.logger.info(f'\nMemory management:\n'
                         f'------------------\n'
                         f'Total: {total_memory}\n'
                         f'Reserved: {reserved_memory}\n'
                         f'Allocated: {allocated_memory}\n'
                         f'Free: {free_memory}\n')

    def checkpoint(self,
                   checkpoint: Union[nn.Module, Dict[str, Any]],
                   tag: str,
                   global_step: int = None
                   ):
        """
        Given nn.Module object or state dictionary, save to disk
        """
        ensure_dir(self.checkpoint_dir)

        if global_step is not None:
            filename = f'{tag}_{global_step}.pt'
        else:
            filename = f'{tag}.pt'

        if isinstance(checkpoint, nn.Module):
            checkpoint = checkpoint.state_dict()

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename
        )

    @contextlib.contextmanager
    def force_logging(self):
        """Force Writer to log by temporarily overriding logging interval"""
        log_iter = self.log_iter
        self.log_iter = 1
        yield
        self.log_iter = log_iter
