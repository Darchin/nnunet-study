"""Persistent CPU batch mappings with privately owned pinned output tensors.

Workers receive the pool once (spawn on Windows, the process context on Linux),
then queue only slot identifiers. A slot is reusable only after its synchronous
copy to a private pinned tensor completes. Returned batches can be retained.
"""

import multiprocessing as mp
import os
import queue
import sys
import threading
import warnings

import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter, producer


class SharedMemoryUnavailable(RuntimeError):
    """The batch pool cannot fit in the available shared-memory storage."""


def batch_bytes(sample):
    if isinstance(sample, torch.Tensor):
        return sample.numel() * sample.element_size()
    if isinstance(sample, (list, tuple)):
        return sum(batch_bytes(t) for t in sample)
    raise TypeError(f"Unsupported batch value {type(sample).__name__}")


def check_shared_memory_capacity(required_bytes):
    if sys.platform == "linux" and os.path.isdir("/dev/shm"):
        stats = os.statvfs("/dev/shm")
        available = stats.f_bavail * stats.f_frsize
        if required_bytes + 64 * 1024**2 > available:
            raise SharedMemoryUnavailable(
                f"Batch pool needs {required_bytes / 1024**2:.0f} MiB plus reserve; "
                f"/dev/shm has {available / 1024**2:.0f} MiB available"
            )


def allocate_pool(sample, slots):
    if isinstance(sample, torch.Tensor):
        return torch.empty((slots, *sample.shape), dtype=sample.dtype).share_memory_()
    return type(sample)(allocate_pool(t, slots) for t in sample)


def copy_to_pool(pool, slot, sample):
    if isinstance(pool, torch.Tensor):
        if pool[slot].shape != sample.shape or pool.dtype != sample.dtype:
            raise ValueError("Shared batches require fixed tensor shapes and dtypes")
        pool[slot].copy_(sample)
    else:
        if type(pool) is not type(sample) or len(pool) != len(sample):
            raise ValueError("Target structure changed between batches")
        for p, t in zip(pool, sample):
            copy_to_pool(p, slot, t)


def pin_from_pool(pool, slot):
    if isinstance(pool, torch.Tensor):
        return pool[slot].pin_memory()
    return type(pool)(pin_from_pool(p, slot) for p in pool)


class SharedBatchLoader(DataLoader):
    def __init__(self, base, slots, context):
        super().__init__([0], base.batch_size, infinite=True)
        self.base = base
        self.transforms = base.transforms
        example = next(base)
        check_shared_memory_capacity(slots * (batch_bytes(example["data"]) + batch_bytes(example["target"])))
        try:
            self.data_pool = allocate_pool(example["data"], slots)
            self.target_pool = allocate_pool(example["target"], slots)
        except (RuntimeError, OSError) as exc:
            raise SharedMemoryUnavailable(f"Cannot allocate shared batch pool: {exc}") from exc
        self.free_slots = context.Queue()
        for slot in range(slots):
            self.free_slots.put(slot)

    def set_thread_id(self, thread_id):
        super().set_thread_id(thread_id)
        self.base.set_thread_id(thread_id)

    def generate_train_batch(self):
        item = next(self.base)
        slot = self.free_slots.get()
        copy_to_pool(self.data_pool, slot, item.pop("data"))
        copy_to_pool(self.target_pool, slot, item.pop("target"))
        item["_shared_batch_slot"] = slot
        return item


def shared_results_loop(in_queue, out_queue, abort_event, workers, pool, gpu, wait_time):
    item = None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        while not abort_event.is_set():
            if not all(worker.is_alive() for worker in workers):
                raise RuntimeError("An augmentation worker exited; inspect its traceback")
            if item is None:
                try:
                    item = in_queue.get(timeout=wait_time)
                except queue.Empty:
                    continue
                slot = item.pop("_shared_batch_slot")
                try:
                    item["data"] = pin_from_pool(pool.data_pool, slot)
                    item["target"] = pin_from_pool(pool.target_pool, slot)
                finally:
                    pool.free_slots.put(slot)
            try:
                out_queue.put(item, timeout=wait_time)
                item = None
            except queue.Full:
                continue
    except Exception:
        abort_event.set()
        raise


class SharedBatchAugmenter(NonDetMultiThreadedAugmenter):
    def __init__(self, data_loader, num_processes, num_cached, start_method=None):
        self.context = mp.get_context(start_method)
        super().__init__(data_loader, None, num_processes, num_cached, pin_memory=True, wait_time=0.002)
        self.base_loader = data_loader
        self.pool_loader = None

    def _start(self):
        if self.initialized:
            return
        # Cover both queues plus one batch per producer and the parent copy.
        slots = self.num_processes + 2 * self.num_cached + 2
        try:
            self.pool_loader = SharedBatchLoader(self.base_loader, slots, self.context)
        except SharedMemoryUnavailable as exc:
            warnings.warn(f"{exc}; using the ordinary augmentation queue", RuntimeWarning, stacklevel=2)
            super()._start()
            return
        self.generator = self.pool_loader
        self._queue = self.context.Queue(self.num_cached)
        self.results_loop_queue = queue.Queue(self.num_cached)
        self.abort_event = self.context.Event()
        self.pause_event = self.context.Event()
        for i in range(self.num_processes):
            process = self.context.Process(
                target=producer,
                args=(self._queue, self.generator, None, i, self.seeds[i], self.abort_event),
                kwargs={"pause_event": self.pause_event, "wait_time": self.wait_time},
                daemon=True,
            )
            self._processes.append(process)
        try:
            for process in self._processes:
                process.start()
        except Exception:
            self._processes = [p for p in self._processes if p.pid is not None]
            self._finish(force=True)
            raise
        self.results_loop_thread = threading.Thread(
            target=shared_results_loop,
            args=(
                self._queue,
                self.results_loop_queue,
                self.abort_event,
                self._processes,
                self.pool_loader,
                torch.cuda.current_device(),
                self.wait_time,
            ),
            daemon=True,
        )
        self.results_loop_thread.start()
        self.initialized = True

    def _finish(self, timeout=10, force=False):
        pool = getattr(self, "pool_loader", None)
        try:
            super()._finish(timeout=timeout, force=force)
        finally:
            if pool is not None:
                pool.free_slots.close()
                pool.free_slots.join_thread()
                self.pool_loader = None
                self.generator = self.base_loader
