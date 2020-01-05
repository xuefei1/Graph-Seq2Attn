import numpy as np
import gc
import torch
import sys
import os
import psutil


# By @ smth
def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def write_line_to_file(s, f_path="progress.txt", new_file=False, verbose=False):
    code = "w" if new_file else "a"
    if verbose: print(s)
    with open(f_path, code, encoding='utf-8') as f:
        f.write(s)
        f.write("\n")


class UnbufferedStdOut:

    def __init__(self, stream, filename=None):
        self.stream = stream
        self.filename = filename

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if self.filename is not None:
            write_line_to_file(str(data), self.filename)

# sys.stdout=UnbufferedStdOut(sys.stdout)