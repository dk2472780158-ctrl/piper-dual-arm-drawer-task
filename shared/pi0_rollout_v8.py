#!/usr/bin/env python3
"""Pi0 drawer client. Dry-run by default; every motion run homes first.

Hardware is delegated to verified_hardware.HardwareSession. In particular this
file never enables motors, bypasses the passive feedback gate, or sends e-stop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import contextlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import numpy as np


ROOT = Path('/home/databall/pi0_drawer_mixed_74_v1')
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.array([6, 13])
LOW = np.tile([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439, 0], 2)
HIGH = np.tile([2.6179, 3.14, 0, 1.745, 1.22, 2.09439, .07], 2)
CAMERAS = {
    'top_head': ('middle', '261822074601', 19011),
    # Canonical training names describe the scene role, not today's USB alias.
    # All eight source batches: hand_left sees drawer; hand_right sees cube.
    'hand_left': ('right', '261622072187', 19013),
    'hand_right': ('left', '261722072569', 19012),
}


def vector(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError(f'{label}: expected 14 finite values')
    return value.copy()


class JointLimitError(ValueError):
    """A rejected policy chunk with exact positions beyond the existing gate."""
    def __init__(self, violations):
        self.violations = violations
        details = '; '.join(
            f"row={v['row']} slot={v['slot']} {v['interface']} J{v['joint']} "
            f"value={v['value_rad']:.6f} limits=[{v['lower_rad']:.6f},{v['upper_rad']:.6f}] "
            f"excess={v['excess_rad']:.6f} rad"
            for v in violations)
        super().__init__('Policy joint target more than 0.05 rad outside hardware limits: ' + details)


def validated_actions(value):
    actions = np.asarray(value, dtype=np.float64)
    if actions.shape != (10, 14) or not np.isfinite(actions).all():
        raise ValueError('Policy must return finite absolute actions with shape (10, 14)')
    invalid = (actions[:, JOINTS] < LOW[JOINTS] - .05) | (actions[:, JOINTS] > HIGH[JOINTS] + .05)
    if np.any(invalid):
        violations = []
        for row, joint_column in np.argwhere(invalid):
            slot = int(JOINTS[joint_column])
            value_rad, lower, upper = float(actions[row, slot]), float(LOW[slot]), float(HIGH[slot])
            violations.append({'row': int(row), 'slot': slot, 'interface': ('can1', 'can0')[slot // 7],
                               'joint': slot % 7 + 1, 'value_rad': value_rad,
                               'lower_rad': lower, 'upper_rad': upper,
                               'excess_rad': max(lower - value_rad, value_rad - upper),
                               'allowed_extrapolation_rad': .05})
        raise JointLimitError(violations)
    actions = actions.copy()
    clamped = int(np.count_nonzero((actions[:, GRIPPERS] < 0) | (actions[:, GRIPPERS] > .07)))
    actions[:, GRIPPERS] = np.clip(actions[:, GRIPPERS], 0, .07)
    actions[:, JOINTS] = np.clip(actions[:, JOINTS], LOW[JOINTS], HIGH[JOINTS])
    return actions, clamped


def limited_target(previous, desired, dt, joint_rate=.12, gripper_rate=.006):
    previous, desired = vector(previous, 'previous command'), vector(desired, 'target')
    if not 0 < dt <= .1:
        raise ValueError('Invalid command interval')
    if np.any(previous < LOW - 1e-3) or np.any(previous > HIGH + 1e-3):
        raise ValueError('Previous command outside hardware limits')
    limits = np.full(14, joint_rate * dt)
    limits[GRIPPERS] = gripper_rate * dt
    return np.clip(previous + np.clip(desired - previous, -limits, limits), LOW, HIGH)


def verify_tracking(measured, commanded, joint_tolerance=.12, gripper_tolerance=.012):
    measured, commanded = vector(measured, 'feedback'), vector(commanded, 'command')
    error = np.abs(measured - commanded)
    if np.max(error[JOINTS]) > joint_tolerance or np.max(error[GRIPPERS]) > gripper_tolerance:
        raise RuntimeError(f'Tracking error: joint={np.max(error[JOINTS]):.4f} rad, '
                           f'gripper={np.max(error[GRIPPERS]):.4f} m')
    return error


def metadata(root):
    info = json.loads((root / 'dataset_train/meta/info.json').read_text())
    fps = float(info['fps'])
    if fps != 30:
        raise ValueError(f'Expected verified training FPS 30, got {fps}')
    prompt = json.loads((root / 'dataset_train/meta/tasks.jsonl').read_text().splitlines()[0])['task']
    return fps, prompt


def home_target(path):
    obj = json.loads(path.read_text())
    target = vector(obj['target'], 'home target')
    if np.any(target < LOW) or np.any(target > HIGH) or not obj.get('source'):
        raise ValueError('Home target needs bounded values and a verified source')
    return target, obj


def usb_parent(path):
    """Return stable USB physical-parent path, excluding USB interface suffix."""
    for part in Path(path).parts:
        if re.fullmatch(r'\d+-\d+(?:\.\d+)*', part):
            candidate = part
    if 'candidate' not in locals():
        raise RuntimeError(f'No USB device parent in {path}')
    return candidate


def verify_camera_mapping():
    """Enumerate RS metadata only; do not start a RealSense capture pipeline."""
    import pyrealsense2 as rs
    devices = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        devices[serial] = usb_parent(device.get_info(rs.camera_info.physical_port))
    report = {}
    for name, (alias, serial, port) in CAMERAS.items():
        node = Path('/dev/camera_' + alias).resolve(strict=True)
        actual_usb = usb_parent((Path('/sys/class/video4linux') / node.name / 'device').resolve(strict=True))
        if devices.get(serial) != actual_usb:
            raise RuntimeError(f'Camera binding mismatch for {name}: serial {serial}, USB {actual_usb}')
        report[name] = {'alias': alias, 'serial': serial, 'video_device': str(node),
                        'usb_device': actual_usb, 'url': f'tcp://127.0.0.1:{port}'}
    return report


def verify_unoccupied_streams():
    # The installed publisher handles one client per stream. Never join a queue
    # behind an existing preview client and later steal its connection.
    result = subprocess.run(['ss', '-tn', 'state', 'established'], capture_output=True,
                            text=True, check=True, timeout=3)
    occupied = []
    for _, (_, _, port) in CAMERAS.items():
        if re.search(rf':{port}(?:\s|$)', result.stdout):
            occupied.append(port)
    if occupied:
        raise RuntimeError(f'Single-client camera streams already in use: {occupied}; '
                           'use a verified shared preview endpoint before rollout')


class CameraStreams:
    """Decode each fanout stream in the background; only expose fresh RGB."""
    live_frames = True
    def __init__(self, max_age=.6):
        self.max_age = max_age
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames = {}
        self.errors = {}
        self.captures = []
        self.threads = []

    def __enter__(self):
        import cv2
        verify_unoccupied_streams()
        for name, (_, _, port) in CAMERAS.items():
            # Open/read timeouts keep camera failure from blocking a hardware
            # monitoring loop indefinitely. Background workers own the handles.
            thread = threading.Thread(target=self._reader, args=(name, port, cv2), daemon=True)
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + 8
        try:
            while True:
                try:
                    self.snapshot()
                    return self
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _reader(self, name, port, cv2):
        capture = None
        try:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000]
            capture = cv2.VideoCapture(f'tcp://127.0.0.1:{port}', cv2.CAP_FFMPEG, params)
            with self.lock:
                self.captures.append(capture)
            if not capture.isOpened():
                raise RuntimeError('Cannot open MPEG-TS stream')
            while not self.stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None or frame.shape != (480, 640, 3):
                    raise RuntimeError('Missing/invalid camera frame')
                rgb = np.ascontiguousarray(frame[:, :, ::-1])
                with self.lock:
                    self.frames[name] = (time.monotonic(), rgb)
        except Exception as error:
            with self.lock:
                self.errors[name] = str(error)
        finally:
            if capture is not None:
                capture.release()

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if self.errors:
                raise RuntimeError(f'Camera reader failed: {self.errors}')
            missing = set(CAMERAS) - set(self.frames)
            if missing:
                raise RuntimeError(f'Waiting for cameras: {sorted(missing)}')
            ages = {name: now - timestamp for name, (timestamp, _) in self.frames.items()}
            if max(ages.values()) > self.max_age:
                raise RuntimeError(f'Stale camera frames: {ages}')
            if max(ages.values()) - min(ages.values()) > .25:
                raise RuntimeError(f'Camera frame timing mismatch: {ages}')
            return {name: frame.copy() for name, (_, frame) in self.frames.items()}, ages

    def __exit__(self, *_):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=1.5)


class Policy:
    """Same msgpack protocol as WebsocketClientPolicy with bounded waits."""
    def __init__(self, host, port, timeout):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.msgpack = msgpack_numpy
        self.timeout = timeout
        self.connection = connect(f'ws://{host}:{port}', compression=None,
                                  max_size=16 * 1024 * 1024, open_timeout=timeout)
        try:
            self.metadata = self.msgpack.unpackb(self.connection.recv(timeout=timeout))
        except BaseException:
            self.connection.close()
            raise

    def infer(self, obs):
        self.connection.send(self.msgpack.packb(obs))
        result = self.connection.recv(timeout=self.timeout)
        if isinstance(result, str):
            raise RuntimeError(f'Policy server error: {result}')
        response = self.msgpack.unpackb(result)
        return response['actions']

    def close(self):
        self.connection.close()


def observation(cameras, state, prompt):
    images, ages = cameras.snapshot()
    return {'images': images, 'state': vector(state, 'observation').astype(np.float32), 'prompt': prompt}, ages


def settle_home(hardware, target, timeout=5):
    deadline = time.monotonic() + timeout
    steady_since = None
    while time.monotonic() < deadline:
        measured = vector(hardware.read_state(require_enabled=True), 'home feedback')
        error = np.abs(measured - target)
        if np.max(error[JOINTS]) <= .035 and np.max(error[GRIPPERS]) <= .004:
            steady_since = time.monotonic() if steady_since is None else steady_since
            if time.monotonic() - steady_since >= .5:
                return measured, error
        else:
            steady_since = None
        time.sleep(.02)
    raise RuntimeError('Home did not stay within tolerance for 0.5 seconds')


def monitored_inference(executor, policy, obs, hardware, cameras, commanded, timeout, deadline,
                        require_enabled=True):
    expires = min(time.monotonic() + timeout, deadline)

    def monitor():
        if time.monotonic() >= expires:
            raise TimeoutError('Inference exceeded timeout or rollout duration')
        measured = vector(hardware.read_state(require_enabled=require_enabled), 'inference feedback')
        if commanded is not None:
            verify_tracking(measured, commanded)
        cameras.snapshot()
        if time.monotonic() >= expires:
            raise TimeoutError('Inference exceeded timeout or rollout duration')

    # Both pre-motion and rollout requests keep the passive feedback/camera
    # watchdog active. Pre-motion never requires enabled motors or tracking.
    monitor()
    future = executor.submit(policy.infer, obs)
    while True:
        monitor()
        try:
            result = future.result(timeout=min(.02, max(.001, expires - time.monotonic())))
        except FutureTimeout:
            if future.done():
                result = future.result()
            else:
                continue
        # A response arriving after the deadline or during a watchdog fault
        # cannot become an accepted chunk.
        monitor()
        return result


def sample_valid_actions(executor, policy, hardware, cameras, commanded, prompt, timeout,
                         deadline, max_samples, report, phase, on_observation=None):
    """Only rejected joint limits permit another fresh, monitored sample."""
    # All attempts share the original inference budget and motion deadline.
    # A rejected response never grants more waiting time.
    deadline = min(time.monotonic() + timeout, deadline)
    require_enabled = phase == 'rollout'
    for attempt in range(1, max_samples + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Inference exceeded timeout or rollout duration')
        measured = vector(hardware.read_state(require_enabled=require_enabled), 'inference observation')
        if commanded is not None:
            verify_tracking(measured, commanded)
        obs, ages = observation(cameras, measured, prompt)
        if on_observation is not None:
            on_observation(obs, ages)
        raw_actions = monitored_inference(executor, policy, obs, hardware, cameras, commanded,
                                         timeout, deadline, require_enabled=require_enabled)
        try:
            actions, clamped = validated_actions(raw_actions)
        except JointLimitError as error:
            rejected = {
                'phase': phase, 'chunk_index': report['rollout']['chunks'], 'attempt': attempt,
                'raw_actions': np.asarray(raw_actions, dtype=np.float64).tolist(),
                'observation_state': vector(obs['state'], 'rejected observation').tolist(),
                'camera_ages_s': dict(ages), 'violations': error.violations,
            }
            report['rejected_policy_chunks'].append(rejected)
            report['rejected_policy_chunk'] = rejected
            report['rejected_policy_chunks_total'] += 1
            if attempt == max_samples:
                raise
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError('Inference exceeded timeout or rollout duration')
        return actions, clamped, raw_actions, obs, ages, attempt


class ChunkUnderrun(RuntimeError):
    """The current block ended before a fresh, validated next block was ready."""


def control_watchdog(hardware, cameras, commanded):
    measured = vector(hardware.read_state(require_enabled=True), 'control feedback')
    error = verify_tracking(measured, commanded)
    cameras.snapshot()
    return measured, error


class PolicyPrefetch:
    """Only policy.infer runs in the worker; all gates and writes stay in the controller."""
    def __init__(self, executor, policy, hardware, cameras, prompt, args, report,
                 motion_deadline, on_observation=None):
        self.executor, self.policy = executor, policy
        self.hardware, self.cameras = hardware, cameras
        self.prompt, self.args, self.report = prompt, args, report
        self.motion_deadline, self.on_observation = motion_deadline, on_observation
        self.future = self.ready = None
        self.closed = False

    def begin(self, chunk_index, activation_tick, commanded, initial=False):
        if self.closed or self.future is not None or self.ready is not None:
            raise RuntimeError('Prefetch already occupied or closed')
        self.chunk_index, self.activation_tick, self.initial = chunk_index, activation_tick, initial
        self.started = time.monotonic()
        # The next block's planned activation is not a new inference budget.
        # A rare retry may cross that tick while the controller sends only its
        # last legal position target. All attempts retain this original expiry.
        self.expires = min(self.started + self.args.inference_timeout,
                           self.motion_deadline)
        self.attempt = 0
        self._submit(commanded)

    def _expired(self):
        if time.monotonic() >= self.expires:
            raise TimeoutError('Inference exceeded timeout or rollout duration')

    def _submit(self, commanded):
        self._expired()
        self.observed_at = time.monotonic()
        measured, _ = control_watchdog(self.hardware, self.cameras, commanded)
        self.obs, self.ages = observation(self.cameras, measured, self.prompt)
        if self.on_observation is not None:
            self.on_observation(self.obs, self.ages)
        self._expired()
        self.attempt += 1
        # No hardware/camera access or report mutation is delegated to this worker.
        self.future = self.executor.submit(self.policy.infer, self.obs)

    def _fresh(self, commanded):
        control_watchdog(self.hardware, self.cameras, commanded)
        age = time.monotonic() - self.observed_at
        input_ages = {name: value + age for name, value in self.ages.items()}
        if age < 0 or max(input_ages.values()) > self.args.camera_max_age:
            raise RuntimeError(f'Prefetched observation expired: {input_ages}')
        return age

    def poll(self, commanded):
        if self.closed or self.ready is not None or self.future is None:
            return
        self._expired()
        if not self.future.done():
            return
        future, self.future = self.future, None
        raw = future.result()  # Network/server faults cannot enter the retry branch.
        record = {'phase': 'rollout', 'chunk_index': self.chunk_index, 'attempt': self.attempt,
                  'observation_state': vector(self.obs['state'], 'policy observation').tolist(),
                  'camera_ages_s': dict(self.ages), 'observed_monotonic_s': self.observed_at,
                  'received_monotonic_s': time.monotonic(), 'status': 'received'}
        try:
            record['raw_actions'] = np.asarray(raw, dtype=np.float64).tolist()
        except (ValueError, TypeError):
            record['raw_actions_repr'] = repr(raw)
        self.report.setdefault('policy_samples', []).append(record)
        try:
            actions, clamped = validated_actions(raw)
        except JointLimitError as error:
            record.update({'status': 'rejected_joint_limits', 'violations': error.violations})
            rejected = dict(record)
            self.report['rejected_policy_chunks'].append(rejected)
            self.report['rejected_policy_chunk'] = rejected
            self.report['rejected_policy_chunks_total'] += 1
            if self.attempt >= self.args.max_policy_samples:
                raise
            self._submit(commanded)
            return
        except Exception as error:
            record.update({'status': 'invalid', 'error': f'{type(error).__name__}: {error}'})
            raise
        record['observation_age_on_validation_s'] = self._fresh(commanded)
        self._expired()
        record['status'] = 'validated'
        self.ready = {'actions': actions, 'clamped': clamped, 'raw_actions': raw,
                      'attempt': self.attempt, 'record': record,
                      'latency_s': time.monotonic() - self.started}

    def take(self, commanded):
        if self.ready is None:
            raise ChunkUnderrun(f'Chunk {self.chunk_index} has no validated actions at activation')
        age = self._fresh(commanded)
        if time.monotonic() >= self.motion_deadline:
            raise TimeoutError('Motion duration exhausted before chunk activation')
        result, self.ready = self.ready, None
        result['record'].update({'status': 'accepted', 'observation_age_on_activation_s': age,
                                 'activated_monotonic_s': time.monotonic()})
        return result

    def cancel(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()
            self.future = None
        if self.ready is not None:
            self.ready['record']['status'] = 'discarded_at_stop'
            self.ready = None


def continuous_rollout(hardware, cameras, policy, executor, commanded, prompt, deadline,
                       args, report, on_observation=None):
    """Fixed 30 Hz actions or bounded position holds, without replay or catch-up."""
    interval = 1 / report['fps']
    prefetch_after_rows = int(getattr(args, 'prefetch_after_rows', 3))
    if not 1 <= prefetch_after_rows < 10:
        raise ValueError('prefetch_after_rows must be in 1..9')
    rollout = report['rollout']
    rollout.update({'control_hz': report['fps'], 'tick_clock': 'absolute',
                    'prefetch_after_rows': prefetch_after_rows, 'control_tick_times_s': [], 'underrun': None,
                    'hold_ticks': 0, 'hold_duration_s': 0., 'hold_tick_times_s': [],
                    'chunk_boundary_waits': []})
    started = time.monotonic()
    prefetch = PolicyPrefetch(executor, policy, hardware, cameras, prompt, args,
                              report, deadline, on_observation)
    boundary_wait = None
    active_record = None

    def finish_wait(status):
        nonlocal boundary_wait
        if boundary_wait is not None:
            boundary_wait.update({'status': status, 'attempts': prefetch.attempt,
                                  'duration_s': time.monotonic() - boundary_wait['started_monotonic_s'],
                                  'finished_monotonic_s': time.monotonic()})
            boundary_wait = None

    def accept():
        nonlocal active_record
        chunk = prefetch.take(commanded)
        active_record = chunk['record']
        rollout['chunks'] += 1
        rollout['last_sampling_attempts'] = chunk['attempt']
        rollout['last_inference_latency_s'] = chunk['latency_s']
        report['gripper_values_clamped'] += chunk['clamped']
        raw = np.asarray(chunk['raw_actions'])
        report['joint_values_clamped'] += int(np.count_nonzero(
            (raw[:, JOINTS] < LOW[JOINTS]) | (raw[:, JOINTS] > HIGH[JOINTS])))
        return chunk['actions']

    try:
        prefetch.begin(0, min(started + args.inference_timeout, deadline), commanded, initial=True)
        while prefetch.ready is None:
            if time.monotonic() >= deadline:
                return
            control_watchdog(hardware, cameras, commanded)
            prefetch.poll(commanded)
            if prefetch.ready is None:
                time.sleep(min(.005, max(0, prefetch.expires - time.monotonic())))
        actions = accept()
        epoch = control_started = time.monotonic()
        sent_rows, row_in_chunk = 0, 0
        last_write = None
        while time.monotonic() < deadline:
            tick = epoch + sent_rows * interval
            while time.monotonic() < min(tick, deadline):
                control_watchdog(hardware, cameras, commanded)
                prefetch.poll(commanded)
                time.sleep(min(.005, max(0, min(tick, deadline) - time.monotonic())))
            if time.monotonic() >= deadline:
                break
            prefetch.poll(commanded)
            holding = False
            if row_in_chunk == 10:
                if prefetch.ready is not None:
                    actions = accept()
                    finish_wait('resumed')
                    row_in_chunk = 0
                else:
                    holding = True
                    if boundary_wait is None:
                        boundary_wait = {'chunk_index': prefetch.chunk_index,
                                         'planned_activation_monotonic_s': prefetch.activation_tick,
                                         'started_monotonic_s': time.monotonic(),
                                         'inference_started_monotonic_s': prefetch.started,
                                         'inference_expires_monotonic_s': prefetch.expires,
                                         'status': 'waiting', 'hold_ticks': 0,
                                         'hold_duration_s': 0.}
                        rollout['chunk_boundary_waits'].append(boundary_wait)
            measured, error = control_watchdog(hardware, cameras, commanded)
            now = time.monotonic()
            lateness = now - tick
            if lateness >= interval:
                raise RuntimeError(f'Control tick missed by {lateness:.6f}s; no catch-up writes')
            # Late ticks do not enlarge slew. A shorter actual write interval
            # also gets proportionately less allowance, preventing catch-up speed.
            dt = interval if last_write is None else min(interval, now - last_write)
            if not holding:
                # Live cameras do not refresh the observation behind this plan.
                source_ages = active_record['camera_ages_s']
                observation_age = now - active_record['observed_monotonic_s']
                plan_age = observation_age + max(source_ages.values())
                if observation_age < 0 or plan_age > args.camera_max_age:
                    raise RuntimeError(f'Active policy observation expired: {plan_age:.6f}s')
                rollout['max_active_observation_age_s'] = max(
                    rollout.get('max_active_observation_age_s', 0.), plan_age)
            next_command = commanded.copy() if holding else limited_target(
                commanded, actions[row_in_chunk], dt, args.joint_rate, args.gripper_rate)
            if time.monotonic() >= deadline:
                break
            if time.monotonic() - tick >= interval:
                raise RuntimeError('Control tick expired before write; no catch-up writes')
            hardware.write_state(next_command)
            commanded = next_command
            last_write = time.monotonic()
            rollout['control_tick_times_s'].append(last_write - control_started)
            rollout['max_tick_lateness_s'] = max(rollout.get('max_tick_lateness_s', 0), lateness)
            rollout['last_state'] = measured.tolist()
            rollout['last_command'] = commanded.tolist()
            rollout['max_joint_tracking_error_rad'] = max(
                rollout.get('max_joint_tracking_error_rad', 0), float(np.max(error[JOINTS])))
            if holding:
                # This consumes one real control interval, not a policy row.
                # Moving the action epoch prevents a burst when inference ends.
                epoch += interval
                rollout['hold_ticks'] += 1
                rollout['hold_duration_s'] += interval
                rollout['hold_tick_times_s'].append(last_write - control_started)
                boundary_wait['hold_ticks'] += 1
                boundary_wait['hold_duration_s'] += interval
                continue
            rollout['action_rows_sent'] += 1
            sent_rows += 1
            row_in_chunk += 1
            if row_in_chunk == prefetch_after_rows:
                activation = epoch + (sent_rows + 10 - prefetch_after_rows) * interval
                if activation < deadline:
                    prefetch.begin(rollout['chunks'], activation, commanded)
            if row_in_chunk == 10:
                print(json.dumps({'event': 'rollout_progress', 'elapsed_s': time.monotonic() - started,
                                  'chunks': rollout['chunks'], 'action_rows_sent': rollout['action_rows_sent']},
                                 ensure_ascii=False), flush=True)
    except ChunkUnderrun as error:
        rollout['underrun'] = {'chunk_index': prefetch.chunk_index, 'attempts': prefetch.attempt,
                              'activation_tick_monotonic_s': prefetch.activation_tick,
                              'elapsed_s': time.monotonic() - started, 'reason': str(error)}
        raise
    except BaseException:
        finish_wait('aborted')
        raise
    finally:
        finish_wait('motion_deadline')
        prefetch.cancel()
        rollout['duration_s'] = time.monotonic() - started


def interrupted(*_):
    raise KeyboardInterrupt()


def run(args):
    if str(args.client_root) not in sys.path:
        sys.path.insert(0, str(args.client_root))
    from verified_hardware import HardwareSession
    report = {'mode': 'motion' if args.enable_motion else 'dryrun', 'motion_sent': False,
              'command_pairs': 0, 'policy_command_pairs': 0, 'task_success': None,
              'home': {'required_before_motion': True, 'success': False},
              'rollout': {'chunks': 0, 'action_rows_sent': 0},
              'max_policy_samples': args.max_policy_samples,
              'rejected_policy_chunks_total': 0, 'rejected_policy_chunks': [],
              'model_slot_can_order': ['can1', 'can0'], 'units': ['rad x 6', 'm'] * 2}
    started = time.monotonic()
    hardware = policy = executor = snapshots = None
    baseline_pairs = None
    motion_started = None
    resources = contextlib.ExitStack()
    try:
        fps, prompt = metadata(args.root)
        target, target_source = home_target(args.home_target)
        report.update({'fps': fps, 'prompt': prompt, 'speed_percent': args.speed_percent,
                       'max_duration_s': args.duration, 'home_source': target_source})
        hardware = resources.enter_context(
            HardwareSession(interfaces=('can1', 'can0'), speed_percent=args.speed_percent))
        # Must succeed before any SDK enable, home or rollout writes.
        report['passive_gate'] = hardware.preflight(seconds=2, allow_estop=True)
        report['initial_state'] = vector(hardware.read_state(require_enabled=False), 'initial state').tolist()
        report['camera_mapping'] = verify_camera_mapping()
        cameras = resources.enter_context(CameraStreams(args.camera_max_age))
        policy = Policy(args.host, args.port, args.inference_timeout)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pi0-inference')
        inference_started = time.monotonic()
        # Retries share one fixed pre-motion budget, before any home command.
        actions, clamped, raw_actions, obs, ages, attempt = sample_valid_actions(
            executor, policy, hardware, cameras, None, prompt, args.inference_timeout,
            inference_started + args.inference_timeout, args.max_policy_samples, report, 'pre_motion')
        report['pre_motion_inference'] = {'shape': list(actions.shape), 'finite': True,
                                         'latency_s': time.monotonic() - inference_started,
                                         'attempts': attempt,
                                         'first_action': actions[0].tolist(), 'camera_ages_s': ages}
        report['gripper_values_clamped'] = clamped
        raw_array = np.asarray(raw_actions)
        report['joint_values_clamped'] = int(np.count_nonzero(
            (raw_array[:, JOINTS] < LOW[JOINTS]) | (raw_array[:, JOINTS] > HIGH[JOINTS])))
        if not args.enable_motion:
            report['result'] = 'dryrun_passed_no_commands'
            return report
        # Capture/source validation and bounded inference have finished.
        # Check the passive gate a second time immediately before home.
        report['pre_home_gate'] = hardware.preflight(seconds=2, allow_estop=True)
        motion_started = home_started = time.monotonic()
        deadline = home_started + args.duration
        home_result = hardware.home(target, timeout=min(60, args.duration))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Motion duration exhausted during home')
        settled, error = settle_home(hardware, target, timeout=min(5, remaining))
        # The adapter's final command is the canonical home target. Feedback at
        # a joint limit can include small encoder noise outside the nominal
        # range; keep measured feedback separate from the commanded trajectory.
        commanded = target.copy()
        report['home'].update({'success': True, 'duration_s': time.monotonic() - home_started,
                               'final_state': settled.tolist(), 'max_joint_error_rad': float(np.max(error[JOINTS])),
                               'max_gripper_error_m': float(np.max(error[GRIPPERS]))})
        if home_result is not None:
            report['home']['adapter_report'] = home_result
        baseline_pairs = hardware.command_pairs
        rollout_started = time.monotonic()
        on_observation = None
        if getattr(cameras, 'live_frames', False):
            try:
                from pi0_frame_snapshots import AsyncFrameSnapshots
                snapshots = AsyncFrameSnapshots(args.report, interval=5)
            except Exception as error:
                report['frame_snapshots'] = {'snapshot_error': f'{type(error).__name__}: {error}'}
            else:
                def on_observation(obs, ages):
                    try:
                        report['frame_snapshots'] = snapshots.capture(
                            obs, ages, time.monotonic() - rollout_started)
                    except Exception as error:
                        report.setdefault('frame_snapshots', {})['snapshot_error'] = (
                            f'{type(error).__name__}: {error}')
        continuous_rollout(hardware, cameras, policy, executor, commanded, prompt, deadline,
                           args, report, on_observation)
        report['result'] = 'bounded_rollout_finished_task_outcome_unverified'
    except KeyboardInterrupt:
        report['result'] = 'interrupted'
    except JointLimitError as error:
        report['result'] = 'blocked_or_aborted'
        report['error'] = f'{type(error).__name__}: {error}'
    except ChunkUnderrun as error:
        report['result'] = 'policy_underrun_held'
        report['error'] = f'{type(error).__name__}: {error}'
    except Exception as error:
        report['result'] = 'blocked_or_aborted'
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        # HardwareSession also owns its lock/subscriber cleanup. Never derive a
        # new target from invalid feedback, disable torque, or send normal e-stop.
        if hardware is not None:
            report['command_pairs'] = hardware.command_pairs
            report['motion_sent'] = hardware.command_pairs > 0
            if baseline_pairs is not None:
                report['policy_command_pairs'] = max(0, hardware.command_pairs - baseline_pairs)
            try:
                report['final_hold'] = hardware.hold() if report['motion_sent'] else 'no_commands_sent'
            except Exception as error:
                report['final_hold'] = {'error': str(error), 'last_valid_command_retained': True}
            report['command_pairs'] = hardware.command_pairs
            report['motion_sent'] = hardware.command_pairs > 0
        if snapshots is not None:
            with contextlib.suppress(Exception):
                report['frame_snapshots'] = snapshots.close()
        if policy is not None:
            with contextlib.suppress(Exception):
                policy.close()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        try:
            resources.close()
        except Exception as error:
            report['hardware_cleanup_error'] = str(error)
        if motion_started is not None:
            report['motion_duration_s'] = time.monotonic() - motion_started
        report['elapsed_s'] = time.monotonic() - started
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-motion', action='store_true', help='Home first, verify arrival, then start bounded rollout')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--client-root', type=Path, default=Path('/home/databall/openpi_pi0_drawer/packages/openpi-client/src'))
    parser.add_argument('--home-target', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--duration', type=float, default=120, help='Maximum seconds from homing start through rollout')
    parser.add_argument('--speed-percent', type=int, default=5)
    parser.add_argument('--joint-rate', type=float, default=.12)
    parser.add_argument('--gripper-rate', type=float, default=.006)
    parser.add_argument('--inference-timeout', type=float, default=3)
    parser.add_argument('--max-policy-samples', type=int, default=3,
                        help='Total fresh samples per chunk, including the first (1..3); only joint-limit rejection retries')
    parser.add_argument('--camera-max-age', type=float, default=.6)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            parser.error(f'{name} must be finite')
    if not 1 <= args.port <= 65535:
        parser.error('Port must be 1..65535')
    if not 0 < args.duration <= 120 or not 1 <= args.speed_percent <= 5:
        parser.error('Duration must be (0,120] seconds and speed-percent 1..5')
    if not 0 < args.joint_rate <= .12 or not 0 < args.gripper_rate <= .006:
        parser.error('Maximum joint-rate is .12 rad/s and gripper-rate .006 m/s')
    if not 0 < args.inference_timeout <= 5 or not 0 < args.camera_max_age <= .6:
        parser.error('Inference timeout must be (0,5] s; camera max age (0,.6] s')
    if not 1 <= args.max_policy_samples <= 3:
        parser.error('Max policy samples must be 1..3 total attempts')
    args.home_target = args.home_target or args.root / 'inference/home_target.json'
    args.report = args.report or args.root / 'inference/reports/pi0_full_rollout.json'
    return args


if __name__ == '__main__':
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    result = run(parse_args())
    raise SystemExit(0 if result['result'] in ('dryrun_passed_no_commands',
                     'bounded_rollout_finished_task_outcome_unverified') else 2)
