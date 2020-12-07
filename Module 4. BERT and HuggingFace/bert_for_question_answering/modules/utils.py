import os
import csv
import torch
import pandas as pd
from typing import Optional, Union
import logging
import datetime


logger = logging.getLogger(__name__)


def set_hardware_acceleration(default: Optional[str] = None) -> torch.device:
    """
    Helper function to set your device. If you don't specify a device argument, it will default to GPU if one is
    available, else it will use a CPU.
    :param default: can be one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu. Default: None
    :return: device: the torch.device to be used for training.
    """
    if default is not None:
        device = torch.device(default)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                f"There are {torch.cuda.device_count()} GPUs available. Using the {torch.cuda.get_device_name()} GPU."
            )
        else:
            device = torch.device("cpu")
            logger.info("No GPUs available, using CPU instead.")
    return device


def format_time(seconds: Union[int, float]) -> str:
    """
    Takes a time in seconds and returns a string in the hh:mm:ss format. E.g. format_time(123) will return '0:02:03'.
    :param seconds: the number of seconds to format in human readable format
    :return: formatted_time: a str in the hh:mm:ss format.
    """
    formatted_time = str(datetime.timedelta(seconds=round(seconds)))
    return formatted_time


def gpu_memory_usage() -> Optional[pd.DataFrame]:
    """
    This function uses Nvidia's SMI tool to check the current GPU memory usage. Reported values are in "MiB".
    1 MiB = 2^20 bytes = 1,048,576 bytes.
    :return: df: a pd.DataFrame with two columns: "memory.total [MiB]" and "memory.used [MiB]". If no GPU is available,
             it returns None and raises a warning.
    """
    if torch.cuda.is_available():
        results = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv')
        reader = csv.reader(results, delimiter=",")
        df = pd.DataFrame(reader)
        df.columns = df.iloc[0]
        df = df[1:]
        return df
    else:
        logger.warning("You called gpu_memory_usage but no GPU is available.")
