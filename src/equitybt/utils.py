import functools
import numpy as np
import pandas as pd
import psutil
import time


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        nonlocal total
        nonlocal total_cpu

        start = time.time()
        start_cpu = time.process_time()

        result = func(*args, **kwargs)

        duration = time.time() - start
        duration_cpu = time.process_time() - start_cpu

        total += duration
        total_cpu += duration_cpu

        print(f"Execution time for {func.__name__}: {duration:.2f}, Total: {total:.2f}")

        return result

    total = 0
    total_cpu = 0
    return wrapper

def cpu_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        proc = psutil.Process()
        proc.cpu_percent(None)
        result = func(*args, **kwargs)
        print(f"CPU usage for {func.__name__}: {proc.cpu_percent(None):.2f}%")
        return result
    return wrapper
    
def with_array_inputs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = tuple(convert_to_numpy_array(arg) for arg in args)
        kwargs = {key: convert_to_numpy_array(value) for key, value in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

def convert_to_numpy_array(value):
    """
    Coerce array-like inputs to a float ndarray
    """
    if isinstance(value, (pd.Series, pd.Index, pd.DataFrame)):
        return value.to_numpy(dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    return value
