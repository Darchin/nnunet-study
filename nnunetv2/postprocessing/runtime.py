"""Case-based CPU work, persistent mask storage, and visible progress."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import multiprocessing
from numbers import Integral
import sys
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from nnunetv2.configuration import default_num_postprocessing_processes


def annotation_digest(segmentation):
    # Reader/writers may represent the same labels with different numeric dtypes.
    values = np.ascontiguousarray(segmentation, dtype='<i4')
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def atomic_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def store_masks(directory, masks, spacing):
    directory = Path(directory)
    atomic_array(directory / 'masks.npy', np.packbits(masks.reshape(-1), bitorder='little'))
    atomic_json(directory / 'geometry.json', {'shape': list(masks.shape), 'spacing': list(map(float, spacing))})


@dataclass
class CachedCaseLoader:
    directory: str

    def __call__(self, identifier):
        directory = Path(self.directory) / identifier
        geometry = json.loads((directory / 'geometry.json').read_text(encoding='utf-8'))
        packed = np.load(directory / 'masks.npy', mmap_mode='r', allow_pickle=False)
        masks = np.unpackbits(packed, count=int(np.prod(geometry['shape'])), bitorder='little').reshape(
            geometry['shape']).view(bool)
        reference = np.load(directory / 'reference.npy', mmap_mode='r', allow_pickle=False)
        return masks, reference, geometry['spacing']


class Progress:
    def __init__(self, log=None, enabled=True):
        self.log, self.enabled = log, enabled

    def __call__(self, message):
        if self.log:
            self.log(message)
        else:
            with tqdm.external_write_mode():
                print(f'{datetime.now()}: {message}')

    @contextmanager
    def phase(self, description):
        self(description)
        yield


def trainer_progress(trainer):
    def log(message):
        with tqdm.external_write_mode():
            trainer.print_to_log_file(message)

    return Progress(log)


def limit_worker_threads():
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    # Keep the limiter alive in this process.
    global _thread_limiter
    if '_thread_limiter' not in globals():
        _thread_limiter = threadpool_limits(1)
    try:
        import SimpleITK
        SimpleITK.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except ImportError:
        pass


def _execute(function, args):
    limit_worker_threads()
    start = time.monotonic()
    value = function(*args)
    return TaskResult(value, time.monotonic() - start, peak_memory())


def peak_memory():
    if sys.platform != 'win32':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, 'peak_wset', info.rss)
    except ImportError:
        return None


@dataclass
class TaskResult:
    value: object
    seconds: float
    peak_bytes: int | None


class CaseExecutor:
    """One long-lived spawn pool; arrays stay local to each case task."""
    def __init__(self, pool=None, num_processes=None, progress=None):
        self.num_processes = default_num_postprocessing_processes if num_processes is None else num_processes
        if isinstance(self.num_processes, bool) or not isinstance(self.num_processes, Integral) or self.num_processes < 1:
            raise ValueError('Post-processing process count must be positive.')
        self.pool, self.owned = pool, False
        self.progress = progress or Progress()
        self.statistics = {'processes': self.num_processes, 'batches': 0, 'case_tasks': 0,
                           'worker_seconds': 0., 'peak_worker_bytes': None}

    def __enter__(self):
        if self.pool is None and self.num_processes > 1:
            self.pool = multiprocessing.get_context('spawn').Pool(self.num_processes)
            self.owned = True
        return self

    def __exit__(self, kind, value, traceback):
        if self.owned:
            self.pool.terminate() if kind else self.pool.close()
            self.pool.join()

    def map(self, function, arguments, description, consume=None):
        arguments = list(arguments)
        self.statistics['batches'] += 1

        def unwrap(result):
            self.statistics['case_tasks'] += 1
            if isinstance(result, TaskResult):
                self.statistics['worker_seconds'] += result.seconds
                if result.peak_bytes is not None:
                    self.statistics['peak_worker_bytes'] = max(
                        self.statistics['peak_worker_bytes'] or 0, result.peak_bytes)
                return result.value
            return result
        with self.progress.phase(description):
            with tqdm(total=len(arguments), desc=description, disable=not self.progress.enabled) as bar:
                if self.pool is None:
                    results = []
                    for index, args in enumerate(arguments):
                        start = time.monotonic()
                        result = unwrap(TaskResult(function(*args), time.monotonic() - start, peak_memory()))
                        consume(index, result) if consume else results.append(result)
                        bar.update()
                    return results
                workers = list(self.pool._pool)
                # Bound queued tasks as nnU-Net does for export; retain deterministic result order.
                pending, results, next_index = {}, [None] * len(arguments) if consume is None else None, 0
                ready, next_result = {}, 0
                while next_index < len(arguments) or pending or ready:
                    if not all(worker.is_alive() for worker in workers):
                        raise RuntimeError('A post-processing worker died. Reduce nnUNet_n_post_proc if RAM is exhausted.')
                    while next_index < len(arguments) and len(pending) + len(ready) < self.num_processes * 2:
                        pending[next_index] = self.pool.apply_async(_execute, (function, arguments[next_index]))
                        next_index += 1
                    completed = [index for index, job in pending.items() if job.ready()]
                    for index in completed:
                        value = unwrap(pending.pop(index).get())
                        if consume:
                            ready[index] = value
                        else:
                            results[index] = value
                        bar.update()
                    while next_result in ready:
                        consume(next_result, ready.pop(next_result))
                        next_result += 1
                    if not completed:
                        time.sleep(.05)
                return results
