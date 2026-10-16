"""Best-effort observation snapshots; no hardware access or timing allowances."""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

import numpy as np


class FrameSnapshots:
    """Save the latest RGB observations without propagating ordinary failures.

    Calls are serial and synchronous. There are no sleeps, retries, or deadline
    adjustments; the caller retains its original control deadline. Failed
    attempts use the same interval as successful attempts to avoid repeated IO
    on every control tick. Each file is replaced atomically, with the manifest
    published last after every image has been encoded.
    """

    def __init__(self, report_path, interval=5):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('Snapshot interval must be finite and positive')
        report_path = Path(report_path).absolute()
        self.directory = report_path.with_name(report_path.stem + '_frames')
        self.manifest = self.directory / 'manifest.json'
        self.last_elapsed_s = None
        self.last_attempt_elapsed_s = None
        self.snapshot_count = 0
        self.snapshot_error = None
        self.image_paths = {}

    def _status(self):
        return {'directory': str(self.directory), 'manifest': str(self.manifest),
                'last_elapsed_s': self.last_elapsed_s,
                'last_attempt_elapsed_s': self.last_attempt_elapsed_s,
                'snapshot_count': self.snapshot_count,
                'snapshot_error': self.snapshot_error,
                'image_paths': dict(self.image_paths)}

    def capture(self, obs, ages, elapsed_s):
        """Return snapshot status; image/metadata/IO failures become an error field."""
        temporary_paths = []
        try:
            elapsed_s = float(elapsed_s)
            if not math.isfinite(elapsed_s) or elapsed_s < 0:
                raise ValueError('Snapshot elapsed_s must be finite and nonnegative')
            if (self.last_attempt_elapsed_s is not None and
                    elapsed_s - self.last_attempt_elapsed_s < self.interval):
                return self._status()
            self.last_attempt_elapsed_s = elapsed_s

            images = obs['images']
            if not isinstance(images, Mapping) or not images:
                raise ValueError('Snapshot needs a nonempty image mapping')
            state = np.asarray(obs['state'], dtype=np.float64)
            if not np.isfinite(state).all():
                raise ValueError('Snapshot state must be finite')
            camera_ages = {name: float(ages[name]) for name in images}
            if any(not math.isfinite(age) or age < 0 for age in camera_ages.values()):
                raise ValueError('Snapshot camera ages must be finite and nonnegative')

            prepared = []
            for name, rgb in images.items():
                if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                    raise ValueError('Snapshot camera name is not a safe filename')
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
                    raise ValueError(f'{name}: expected a nonempty uint8 RGB image')
                prepared.append((name, rgb[..., ::-1].copy()))

            self.directory.mkdir(parents=True, exist_ok=True)
            cv2 = importlib.import_module('cv2')
            token = uuid.uuid4().hex
            pending_images = []
            image_paths = {}
            for name, bgr in prepared:
                destination = self.directory / (name + '.jpg')
                temporary = self.directory / f'.{name}.{token}.tmp.jpg'
                temporary_paths.append(temporary)
                if not cv2.imwrite(str(temporary), bgr):
                    raise OSError(f'JPEG encoding failed for {name}')
                pending_images.append((temporary, destination))
                image_paths[name] = str(destination)

            manifest = {'elapsed_s': elapsed_s, 'state': state.tolist(),
                        'camera_ages_s': camera_ages, 'images': image_paths,
                        'snapshot_index': self.snapshot_count + 1}
            temporary_manifest = self.directory / f'.manifest.{token}.tmp.json'
            temporary_paths.append(temporary_manifest)
            temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + '\n', encoding='utf-8')
            for temporary, destination in pending_images:
                temporary.replace(destination)
            temporary_manifest.replace(self.manifest)
            self.last_elapsed_s = elapsed_s
            self.snapshot_count += 1
            self.snapshot_error = None
            self.image_paths = image_paths
        except Exception as error:
            self.snapshot_error = f'{type(error).__name__}: {error}'
        finally:
            for temporary in temporary_paths:
                with contextlib.suppress(Exception):
                    temporary.unlink(missing_ok=True)
        return self._status()


class AsyncFrameSnapshots:
    """Run the existing snapshot writer outside the caller's control loop.

    There is at most one submitted job, including a running job. Calls while
    that job is busy are dropped; no old observations accumulate in a queue.
    Only this wrapper's caller collects completed status. The writer receives
    detached arrays and never accesses hardware or an external report.
    """

    def __init__(self, report_path, interval=5):
        self._writer = FrameSnapshots(report_path, interval)
        self.interval = self._writer.interval
        self.directory = self._writer.directory
        self.manifest = self._writer.manifest
        self._completed_status = self._writer._status()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pi0-snapshots')
        self._lock = threading.Lock()
        self._future = None
        self._closed = False
        self._last_attempt_elapsed_s = None
        self._last_submitted_elapsed_s = None

    def _poll_locked(self):
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        if future.cancelled():
            return
        try:
            completed = dict(future.result())  # done() makes this nonblocking
            completed['image_paths'] = dict(completed.get('image_paths', {}))
            self._completed_status = completed
        except Exception as error:
            self._completed_status['snapshot_error'] = f'{type(error).__name__}: {error}'

    def _status_locked(self):
        status = dict(self._completed_status)
        status['image_paths'] = dict(status['image_paths'])
        status.update({'pending': self._future is not None, 'closed': self._closed,
                       'last_submitted_elapsed_s': self._last_submitted_elapsed_s})
        return status

    def status(self):
        """Collect an already completed result without waiting or doing IO."""
        with self._lock:
            self._poll_locked()
            return self._status_locked()

    def capture(self, obs, ages, elapsed_s):
        """Schedule a due observation and return the last completed status."""
        with self._lock:
            self._poll_locked()
            if self._closed:
                return self._status_locked()
            try:
                elapsed_s = float(elapsed_s)
                if not math.isfinite(elapsed_s) or elapsed_s < 0:
                    raise ValueError('Snapshot elapsed_s must be finite and nonnegative')
                if (self._last_attempt_elapsed_s is not None and
                        elapsed_s - self._last_attempt_elapsed_s < self.interval):
                    return self._status_locked()
                if self._future is not None:
                    return self._status_locked()
                # Throttle failed copying/submission attempts too. All encoding,
                # validation and file operations remain in the original writer.
                self._last_attempt_elapsed_s = elapsed_s
                images = obs['images']
                if not isinstance(images, Mapping) or not images:
                    raise ValueError('Snapshot needs a nonempty image mapping')
                detached = {'images': {name: np.asarray(rgb).copy() for name, rgb in images.items()},
                            'state': np.asarray(obs['state']).copy()}
                detached_ages = dict(ages)
                self._future = self._executor.submit(
                    self._writer.capture, detached, detached_ages, elapsed_s)
                self._last_submitted_elapsed_s = elapsed_s
            except Exception as error:
                self._completed_status['snapshot_error'] = f'{type(error).__name__}: {error}'
            self._poll_locked()
            return self._status_locked()

    def close(self):
        """Cancel an unstarted job; a running file writer finishes independently."""
        with self._lock:
            self._poll_locked()
            if not self._closed:
                self._closed = True
                if self._future is not None:
                    self._future.cancel()
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception as error:
                    self._completed_status['snapshot_error'] = f'{type(error).__name__}: {error}'
                self._poll_locked()
            return self._status_locked()
