#!/usr/bin/env python3
"""Raw-feedback-gated dual Piper hardware, in recording order can1 then can0.

Imports and preflight are passive. The Piper SDK is instantiated only by home().
Closing a session neither disables torque nor sends an emergency-stop command.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import hashlib
import math
from pathlib import Path
import socket
import struct
import threading
import time
import types

import numpy as np


JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.array([6, 13])
LOWER = LOW = np.tile([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439, 0], 2)
UPPER = HIGH = np.tile([2.6179, 3.14, 0, 1.745, 1.22, 2.09439, .07], 2)
STATUS_ID = 0x2A1
JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
GRIPPER_ID = 0x2A8
LOW_SPEED_IDS = tuple(range(0x261, 0x267))
FEEDBACK_IDS = (STATUS_ID,) + JOINT_IDS + (GRIPPER_ID,) + LOW_SPEED_IDS
JOINT_SPREAD_MAX_S = .02
JOINT_SAMPLE_MAX_AGE_S = .05
CLOCK_OFFSET_TOLERANCE_S = .002
CLOCK_READ_MAX_SPAN_S = .005
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)  # Linux SO_TIMESTAMPNS_OLD.
LOCK_PATH = Path('/home/databall/pico_piper_teleop/runtime/hardware.lock')
RESET_PATH = Path('/home/databall/VLA/lerobot_piper/reset_piper_pose.py')
POSITION_KEYS = [f'joint_{i}_pos' for i in range(1, 7)] + ['gripper_pos']


class GateError(RuntimeError):
    def __init__(self, code, interface, detail):
        self.code, self.interface, self.detail = code, interface, detail
        super().__init__(f'{interface}: {code}: {detail}')


def vector(value, label='state'):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError(f'{label} must have 14 finite rad/m values')
    return value.copy()


def bounded(value):
    value = vector(value)
    if np.any(value < LOWER) or np.any(value > UPPER):
        bad = np.flatnonzero((value < LOWER) | (value > UPPER)).tolist()
        raise ValueError(f'Command outside SDK nominal bounds at indices {bad}')
    return value


def measured_preload(value):
    """Validate original feedback before explicitly correcting boundary noise."""
    value = vector(value, 'raw measured pose')
    clipped = np.clip(value, LOWER, UPPER)
    difference = np.abs(clipped - value)
    if np.max(difference[JOINTS]) > .01 + 1e-12 or np.max(difference[GRIPPERS]) > .001 + 1e-12:
        raise ValueError('Raw pose outside preload tolerance (joint .01 rad / gripper .001 m)')
    return clipped, (clipped - value)


class RawFeedback:
    """Thread-safe parser; neither this object nor RawMonitor can send frames."""
    def __init__(self, interface, clock=time.monotonic):
        self.interface, self.clock = interface, clock
        self.lock = threading.RLock()
        self.parts = {}
        self.pending_joints = {}
        self.joint_sample = None
        self.receive_clock = 'feed_timestamp'
        self.frames = defaultdict(deque)
        self.alternations = deque()
        self.jumps = defaultdict(deque)
        self.reader_error = None

    def feed(self, can_id, payload, timestamp=None):
        if can_id not in FEEDBACK_IDS:
            return
        if len(payload) != 8:
            with self.lock:
                self.reader_error = f'Invalid DLC {len(payload)} for {can_id:#x}'
            return
        now = self.clock() if timestamp is None else timestamp
        payload = bytes(payload)
        with self.lock:
            history = self.frames[can_id]
            if can_id == STATUS_ID and len(history) >= 2:
                # Motion/teach/trajectory bytes 2:6 can alternate while one
                # healthy controller moves. Only source/control/fault fields
                # participate in the duplicate-status identity gate.
                identity = payload[:2] + payload[6:8]
                older_identity = history[-2][1][:2] + history[-2][1][6:8]
                previous_identity = history[-1][1][:2] + history[-1][1][6:8]
                if identity == older_identity and identity != previous_identity:
                    self.alternations.append(now)
            if can_id in JOINT_IDS and history:
                previous_time, previous_data = history[-1]
                delta = np.abs(np.array(struct.unpack('>ii', payload), dtype=np.float64)
                               - np.array(struct.unpack('>ii', previous_data), dtype=np.float64))
                if 0 <= now - previous_time <= .02 and np.max(delta) * math.pi / 180000 > .03:
                    self.jumps[can_id].append(now)
            history.append((now, payload))
            self.parts[can_id] = (now, payload)
            if can_id in JOINT_IDS:
                # Publish only after receiving every pair. A reader must never
                # observe one pair from the new cycle and two old pairs.
                self.pending_joints[can_id] = (now, payload)
                if len(self.pending_joints) == len(JOINT_IDS):
                    stamps = [self.pending_joints[cid][0] for cid in JOINT_IDS]
                    if max(stamps) - min(stamps) <= JOINT_SPREAD_MAX_S + 1e-12:
                        self.joint_sample = dict(self.pending_joints)
                        self.pending_joints.clear()
                    else:
                        # A delayed/incomplete pair cannot contaminate a
                        # completed group. Await new data for the oldest pair.
                        oldest = min(self.pending_joints, key=lambda cid: self.pending_joints[cid][0])
                        del self.pending_joints[oldest]
            self._prune(now)

    def _prune(self, now):
        for history in self.frames.values():
            while history and history[0][0] < now - .5:
                history.popleft()
        while self.alternations and self.alternations[0] < now - .5:
            self.alternations.popleft()
        for history in self.jumps.values():
            while history and history[0] < now - .5:
                history.popleft()

    def _fail(self, code, detail):
        raise GateError(code, self.interface, detail)

    def snapshot(self, require_enabled=False, allow_estop=True, now=None):
        with self.lock:
            now = self.clock() if now is None else now
            self._prune(now)
            if self.reader_error:
                self._fail('receive_error', self.reader_error)
            rates = {hex(cid): len(self.frames[cid]) / .5 for cid in JOINT_IDS}
            if any(rate > 300 for rate in rates.values()):
                self._fail('joint_rate', f'Joint feedback exceeds 300 Hz: {rates}')
            if len(self.alternations) >= 4:
                self._fail('alternating_status', f'{len(self.alternations)} ABAB control/status/error returns in .5 s')
            for cid, jumps in self.jumps.items():
                if len(jumps) >= 4:
                    self._fail('joint_jump', f'{cid:#x}: {len(jumps)} jumps >.03 rad within .02 s in .5 s')
            missing = [hex(cid) for cid in FEEDBACK_IDS if cid not in self.parts]
            if missing:
                self._fail('missing_feedback', missing)
            ages = {hex(cid): now - self.parts[cid][0] for cid in FEEDBACK_IDS}
            stale = {hex(cid): ages[hex(cid)] for cid in FEEDBACK_IDS
                     if ages[hex(cid)] < 0 or ages[hex(cid)] > (.3 if cid in LOW_SPEED_IDS else .2)}
            if stale:
                self._fail('stale_feedback', stale)
            latest_stamps = [self.parts[cid][0] for cid in JOINT_IDS]
            if self.joint_sample is None:
                self._fail('incoherent_joints', 'No complete joint group within .02 s')
            joint_stamps = [self.joint_sample[cid][0] for cid in JOINT_IDS]
            joint_age = now - min(joint_stamps)
            if joint_age < 0 or joint_age > JOINT_SAMPLE_MAX_AGE_S + 1e-12:
                self._fail('stale_joint_sample', f'Completed joint group age {joint_age:.6f} s exceeds .05 s')
            status = self.parts[STATUS_ID][1]
            arm_status, error_code = status[1], int.from_bytes(status[6:8], 'big')
            if error_code:
                self._fail('arm_error', f'Error {error_code:#x}, status={status.hex()}')
            if arm_status not in (0, 1) or (arm_status == 1 and (require_enabled or not allow_estop)):
                self._fail('arm_status', f'Expected NORMAL{"/ESTOP" if allow_estop and not require_enabled else ""}, got {arm_status}')
            flags = [self.parts[cid][1][5] for cid in LOW_SPEED_IDS]
            if any(flag & 0xBF for flag in flags):
                self._fail('motor_error', f'Low-speed flags={flags}; non-enable bits set')
            enabled = [bool(flag & 0x40) for flag in flags]
            gripper_flags = self.parts[GRIPPER_ID][1][6]
            if gripper_flags & 0x3F:
                self._fail('gripper_error', f'Gripper flags={gripper_flags:#x}')
            gripper_enabled = bool(gripper_flags & 0x40)
            if require_enabled and not all(enabled):
                self._fail('disabled', f'Motor enable flags={enabled}')
            if require_enabled and not gripper_enabled:
                self._fail('gripper_disabled', f'Gripper flags={gripper_flags:#x}')
            latest_q = np.array([n * math.pi / 180000 for cid in JOINT_IDS
                                 for n in struct.unpack('>ii', self.parts[cid][1])]
                         + [struct.unpack('>i', self.parts[GRIPPER_ID][1][:4])[0] / 1e6], dtype=np.float64)
            if not np.isfinite(latest_q).all() or np.max(np.abs(latest_q[:6])) > 3.2 or not -.001 <= latest_q[6] <= .1:
                self._fail('raw_sanity', f'Original raw pose outside ±3.2 rad / [-.001,.1] m: {latest_q.tolist()}')
            q = np.array([n * math.pi / 180000 for cid in JOINT_IDS
                          for n in struct.unpack('>ii', self.joint_sample[cid][1])]
                         + [latest_q[6]], dtype=np.float64)
            if not np.isfinite(q).all() or np.max(np.abs(q[:6])) > 3.2:
                self._fail('raw_sanity', f'Completed raw joint group outside ±3.2 rad: {q.tolist()}')
            # q is original feedback. Never clip or replace it with a command.
            return q, {'interface': self.interface, 'status': arm_status, 'error_code': error_code,
                       'enabled': enabled, 'low_speed_flags': flags, 'ages_s': ages,
                       'gripper_flags': gripper_flags, 'gripper_enabled': gripper_enabled,
                       'joint_rates_hz': rates, 'status_alternations': len(self.alternations),
                       'joint_frame_spread_s': max(joint_stamps) - min(joint_stamps),
                       'latest_joint_frame_spread_s': max(latest_stamps) - min(latest_stamps),
                       'joint_sample_age_s': joint_age, 'receive_clock': self.receive_clock,
                       'joint_jumps': {hex(cid): len(v) for cid, v in self.jumps.items()}}


def verified_clock_offset(reference=None):
    """Bracket realtime with monotonic; reject slow reads or a wall-clock step."""
    before = time.monotonic()
    realtime = time.time()
    after = time.monotonic()
    if not all(math.isfinite(value) for value in (before, realtime, after)):
        raise RuntimeError('Invalid clock calibration values')
    span = after - before
    if span < 0 or span > CLOCK_READ_MAX_SPAN_S:
        raise RuntimeError(f'Clock calibration read span {span:.6f} s exceeds .005 s')
    # The realtime read occurred somewhere inside this monotonic bracket.
    lower, upper = realtime - after, realtime - before
    if reference is None:
        return realtime - (before + span / 2)
    if (not math.isfinite(reference)
            or reference < lower - CLOCK_OFFSET_TOLERANCE_S
            or reference > upper + CLOCK_OFFSET_TOLERANCE_S):
        raise RuntimeError('Realtime/monotonic clock offset changed by more than .002 s')
    return reference


def kernel_monotonic_timestamp(ancillary, realtime_minus_monotonic):
    """Convert the kernel CAN arrival time, never the reader's scheduling time."""
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS and len(data) >= struct.calcsize('@ll'):
            seconds, nanos = struct.unpack('@ll', data[:struct.calcsize('@ll')])
            if seconds < 0 or not 0 <= nanos < 1_000_000_000:
                raise RuntimeError('Invalid kernel CAN timestamp')
            return seconds + nanos / 1e9 - realtime_minus_monotonic
    raise RuntimeError('Missing kernel CAN timestamp')


class RawMonitor(RawFeedback):
    def __init__(self, interface):
        super().__init__(interface)
        self.socket = None
        self.thread = None
        self.stop = threading.Event()
        self.realtime_minus_monotonic = None
        self.receive_clock = 'kernel_so_timestampns'

    def start(self):
        self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        self.realtime_minus_monotonic = verified_clock_offset()
        # Exact standard-data-frame IDs. No TX method exists on this class.
        filters = b''.join(struct.pack('=II', cid, 0xC00007FF) for cid in FEEDBACK_IDS)
        self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
        self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        self.socket.bind((self.interface,))
        self.socket.settimeout(.05)
        self.thread = threading.Thread(target=self._receive, daemon=True, name='raw-' + self.interface)
        self.thread.start()

    def _receive(self):
        try:
            while not self.stop.is_set():
                try:
                    frame, ancillary, flags, _ = self.socket.recvmsg(16, socket.CMSG_SPACE(struct.calcsize('@ll')))
                except socket.timeout:
                    continue
                # Keep the calibrated conversion fixed. Recalibrating after a
                # wall-clock step could assign a false age to queued frames.
                verified_clock_offset(self.realtime_minus_monotonic)
                if len(frame) != 16:
                    raise RuntimeError(f'Unexpected CAN packet length {len(frame)}')
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    raise RuntimeError('Truncated CAN frame or kernel timestamp')
                cid, length, payload = struct.unpack('=IB3x8s', frame)
                received_at = kernel_monotonic_timestamp(ancillary, self.realtime_minus_monotonic)
                self.feed(cid, payload[:length], timestamp=received_at)
        except Exception as error:
            if not self.stop.is_set():
                with self.lock:
                    self.reader_error = repr(error)

    def close(self):
        self.stop.set()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.thread is not None:
            self.thread.join(timeout=.2)


def load_reset_module(path=RESET_PATH):
    """Load audited reset maths using AST; strip all lerobot constructors/imports."""
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and (node.module or '').startswith('lerobot'))
                 and not (isinstance(node, ast.Import) and any(n.name.startswith('lerobot') for n in node.names))
                 and not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'main')
                 and not isinstance(node, ast.If)]
    module = types.ModuleType('audited_reset_pose')
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    module.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module


class _HomeBus:
    def __init__(self, robot, slot):
        self.robot, self.slot = robot, slot

    def read(self):
        self.robot.check_deadline()
        # read_positions calls left and right consecutively; share one whole snapshot.
        fresh = self.robot.session.read_state(require_enabled=True)
        if self.slot == 0:
            self.robot.snapshot = fresh
        q = self.robot.snapshot[self.slot * 7:self.slot * 7 + 7]
        return dict(zip(POSITION_KEYS, q.tolist()))

    def write(self, state):
        self.robot.check_deadline()
        q = np.asarray(state, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Invalid home bus command')
        if self.slot == 0:
            self.robot.pending = q.copy()
        else:
            if self.robot.pending is None:
                raise RuntimeError('Home pair missing first arm')
            paired = np.concatenate((self.robot.pending, q))
            self.robot.pending = None
            self.robot.session.write_state(paired, recovery=True)


class _HomeRobot:
    def __init__(self, session, deadline):
        self.session, self.deadline = session, deadline
        self.snapshot = self.pending = None
        self.left_bus, self.right_bus = _HomeBus(self, 0), _HomeBus(self, 1)

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Home exceeded overall timeout; last valid target retained')


class HardwareSession:
    def __init__(self, interfaces=('can1', 'can0'), speed_percent=5):
        if tuple(interfaces) != ('can1', 'can0'):
            raise ValueError('Verified model slot order must be can1 then can0')
        if not isinstance(speed_percent, int) or not 1 <= speed_percent <= 5:
            raise ValueError('speed_percent must be an integer in 1..5')
        self.interfaces, self.speed_percent = tuple(interfaces), speed_percent
        self.monitors = []
        self.arms = []
        self.command_pairs = 0
        self.last_command = self.last_command_time = None
        self.lock_file = None
        self._active = self._homing = self._homed = False
        self._write_lock = threading.Lock()

    def __enter__(self):
        import fcntl
        self.lock_file = LOCK_PATH.open('a+')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for interface in self.interfaces:
                monitor = RawMonitor(interface)
                self.monitors.append(monitor)
                monitor.start()
            self._active = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self._active = self._homing = False
        errors = []
        for arm in self.arms:
            try:
                callback = getattr(arm, 'DisconnectPort', None)
                if callback:
                    callback()
            except Exception as error:
                errors.append(repr(error))
        self.arms = []
        for monitor in self.monitors:
            try:
                monitor.close()
            except Exception as error:
                errors.append(repr(error))
        self.monitors = []
        if self.lock_file is not None:
            self.lock_file.close()  # closes and unlocks even if SDK close failed
            self.lock_file = None
        if errors:
            raise RuntimeError('Hardware cleanup failed: ' + '; '.join(errors))

    def _snapshots(self, require_enabled=False, allow_estop=True):
        if not self._active or len(self.monitors) != 2:
            raise RuntimeError('HardwareSession is not open')
        # Freeze both complete raw state banks before deriving one atomic np14.
        with self.monitors[0].lock, self.monitors[1].lock:
            now = time.monotonic()
            rows = [monitor.snapshot(require_enabled, allow_estop, now=now) for monitor in self.monitors]
        return np.concatenate([row[0] for row in rows]), [row[1] for row in rows]

    def read_state(self, require_enabled=False):
        return self._snapshots(require_enabled, allow_estop=not require_enabled)[0]

    def preflight(self, seconds=2, allow_estop=True):
        if not math.isfinite(seconds) or seconds < .5:
            raise ValueError('Passive preflight must observe at least .5 seconds')
        started = time.monotonic()
        last = None
        valid_since = None
        while time.monotonic() - started < seconds:
            try:
                _, last = self._snapshots(False, allow_estop)
                if valid_since is None:
                    valid_since = time.monotonic()
            except GateError as error:
                if error.code not in ('missing_feedback', 'stale_feedback') or valid_since is not None:
                    raise
            time.sleep(.01)
        q, last = self._snapshots(False, allow_estop)
        if valid_since is None or time.monotonic() - valid_since < .5:
            raise GateError('insufficient_observation', 'both', 'Need .5 s continuously valid raw feedback')
        return {'ok': True, 'passive_only': True, 'seconds': time.monotonic() - started,
                'raw_state': q.tolist(), 'interfaces': last, 'sdk_initialized': bool(self.arms)}

    def _initialize_sdk(self):
        if self.arms:
            return
        from piper_sdk import C_PiperInterface_V2
        for interface in self.interfaces:
            arm = C_PiperInterface_V2(interface, start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
            self.arms.append(arm)
            arm.ConnectPort(piper_init=False)

    def _send_pair(self, target):
        target = bounded(target)
        if len(self.arms) != 2:
            raise RuntimeError('SDK has not been initialized by home')
        # Count a partially sent pair as motion for exception/hold reporting.
        self.command_pairs += 1
        for slot, arm in enumerate(self.arms):
            row = target[slot * 7:slot * 7 + 7]
            arm.MotionCtrl_2(1, 1, self.speed_percent, 0)
            arm.JointCtrl(*np.rint(row[:6] * 180000 / math.pi).astype(int).tolist())
            arm.GripperCtrl(int(round(row[6] * 1e6)), 1000, 1, 0)
        self.last_command, self.last_command_time = target.copy(), time.monotonic()

    def write_state(self, state, recovery=False):
        target = bounded(state)
        with self._write_lock:
            if recovery and not self._homing:
                raise RuntimeError('Recovery writes are restricted to home')
            if not recovery and not self._homed:
                raise RuntimeError('Successful home required before rollout writes')
            raw, _ = self._snapshots(True, False)
            if recovery:
                measured_preload(raw)  # validate original raw, never replace read_state
            if self.last_command is None:
                raise RuntimeError('Measured preload must precede writes')
            dt = min(.1, max(0, time.monotonic() - self.last_command_time))
            delta = np.abs(target - self.last_command)
            if np.max(delta[JOINTS]) > .25 * dt + .01 + 1e-12:
                raise ValueError('Joint command slew exceeds .25 rad/s + .01 rad tolerance')
            if np.max(delta[GRIPPERS]) > .006 * dt + .001 + 1e-12:
                raise ValueError('Gripper command slew exceeds .006 m/s + .001 m tolerance')
            self._send_pair(target)

    def home(self, target, timeout=60):
        target = bounded(target)
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('Home timeout must be (0, 60] s')
        started, deadline = time.monotonic(), time.monotonic() + timeout
        self._homed = False
        raw, initial_status = self._snapshots(False, False)  # never automatically reset ESTOP
        initial_raw = raw.copy()
        preload, correction = measured_preload(raw)  # abort before SDK/enable for out-of-range raw
        reset = load_reset_module()
        # Preserve the original final-target check as well as SDK nominal bounds.
        reset.validate_pose(target, 'home target')
        if np.asarray(reset.TRAINING_START_POSE).shape != (14,):
            raise RuntimeError('Audited reset source has invalid default pose')
        self._initialize_sdk()
        self._homing = True
        try:
            # Refresh after connecting; never preload a pose cached before SDK setup.
            raw, initial_status = self._snapshots(False, False)
            preload, correction = measured_preload(raw)
            self._send_pair(preload)
            time.sleep(.05)
            enable_deadline = min(deadline, time.monotonic() + 8)
            while True:
                raw, statuses = self._snapshots(False, False)
                measured_preload(raw)
                if np.max(np.abs(raw[JOINTS] - preload[JOINTS])) > .08:
                    raise RuntimeError('Enable excursion >.08 rad; last valid command retained')
                if all(all(status['enabled']) and status['gripper_enabled'] for status in statuses):
                    self._snapshots(True, False)
                    break
                if time.monotonic() >= enable_deadline:
                    raise TimeoutError('Motor enable timeout; last valid command retained')
                for arm, status in zip(self.arms, statuses):
                    if not all(status['enabled']):
                        arm.EnableArm(7)
                self._send_pair(preload)
                time.sleep(.05)
            raw = self.read_state(require_enabled=True)
            start, correction_after_enable = measured_preload(raw)
            robot = _HomeRobot(self, deadline)
            args = types.SimpleNamespace(fps=30., joint_speed=.12, gripper_speed=.006,
                                         tracking_limit=.12, gripper_tracking_limit=.015,
                                         settle_tolerance=.025, settle_timeout=min(10., max(.001, deadline-time.monotonic())))
            reset.stop_requested = False
            reset.move_slowly(robot, start, target, args)
            robot.check_deadline()
            final, statuses = self._snapshots(True, False)
            error = np.abs(final - target)
            if np.max(error[JOINTS]) > .025 or np.max(error[GRIPPERS]) > .015:
                raise RuntimeError('Home final raw feedback outside tolerance')
            self._homed = True
            return {'success': True, 'duration_s': time.monotonic() - started,
                    'source': str(RESET_PATH), 'source_sha256': reset.source_sha256,
                    'initial_state': initial_raw.tolist(), 'preload_correction': correction.tolist(),
                    'start_boundary_correction': correction_after_enable.tolist(),
                    'target': target.tolist(), 'final_state': final.tolist(),
                    'max_joint_error_rad': float(np.max(error[JOINTS])),
                    'max_gripper_error_m': float(np.max(error[GRIPPERS])), 'interfaces': statuses,
                    'command_pairs': self.command_pairs}
        finally:
            self._homing = False

    def hold(self):
        if not self.arms:
            return {'sent': False, 'reason': 'SDK was not initialized', 'last_valid_command_retained': True}
        try:
            with self._write_lock:
                raw, statuses = self._snapshots(True, False)
                pose, correction = measured_preload(raw)
                self._send_pair(pose)
                return {'sent': True, 'reason': 'Fresh normal enabled measured pose',
                        'raw_state': raw.tolist(), 'explicit_boundary_correction': correction.tolist(),
                        'command_pairs': self.command_pairs}
        except Exception as error:
            return {'sent': False, 'reason': str(error), 'last_valid_command_retained': True}
