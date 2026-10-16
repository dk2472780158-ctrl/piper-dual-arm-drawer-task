"""Bounded π0 trial against the arm-role-corrected checkpoint.

Same verified feedback and motion gates as pi0_real_trial_prelift.py, with the
policy always remote and both ends of that link configurable:

  * the checkpoint is not built here; it is served by the A10 and reached through
    an SSH tunnel, so the robot side holds no weights and cannot silently run a
    different one -- every response is checked against
    --expected-remote-checkpoint and its SHA256 is recorded in the report;
  * --remote-url picks the server, which is what makes an A/B run possible: the
    18024 server answers for the armfix checkpoint, the older 18021 server for
    drawer_pi0_20k_resume_20261013.

The motion path resets first and only then starts the policy:

  preflight -> cameras -> one policy sample with motors still passive
    -> preflight -> 复位 hardware.home(home_target.json) -> settle
    -> prelift 0.45 rad off the J2 floor / J3 ceiling -> settle -> rollout

The prelift exists because the demonstration started from J2 = 0.000 and
J3 = 0.000, exactly the J2 floor and the J3 ceiling: the policy's residual at
that pose has no room on the wrong side of either limit and the unchanged joint
gate rejects it before any CAN frame is written.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import cv2
import contextlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pi0_rollout_v8 as control
from verified_hardware import HardwareSession

ARMFIX_CHECKPOINT = ('/home/devuser/pi0_drawer_mixed_74_v1/pi0_checkpoints/'
                     'drawer_pi0_armfix_20k_20261013/checkpoints/020000/pretrained_model')
RESUME_CHECKPOINT = ('/home/devuser/pi0_drawer_mixed_74_v1/pi0_checkpoints/'
                     'drawer_pi0_20k_resume_20261013/checkpoints/020000/pretrained_model')

PRELIFT_J2_SLOTS = (1, 8)      # can1 J2, can0 J2
PRELIFT_J3_SLOTS = (2, 9)      # can1 J3, can0 J3
# The lift off the J2 floor / J3 ceiling that lets the policy's first block pass
# the joint gate (pi0_full_rollout.validated_actions allows .05 rad past a limit).
# Measured over the 13 held-out validation episodes: the worst frame-0 request is
# 0.113 rad below the J2 floor, and 0.30 rad clears the gate for all 13, while the
# old 0.45 was 50% more than needed.  The demonstrations all start at J2 = 0.004,
# so the surplus is pure out-of-distribution offset at the start of every run.
PRELIFT_J2_RAD = PRELIFT_J3_RAD = float(os.environ.get('PI0_PRELIFT_RAD', '0.45'))


def prelift_target(home):
    """Home pose pushed off the J2 floor and the J3 ceiling."""
    lifted = np.asarray(home, dtype=np.float64).copy()
    lifted[list(PRELIFT_J2_SLOTS)] += PRELIFT_J2_RAD
    lifted[list(PRELIFT_J3_SLOTS)] -= PRELIFT_J3_RAD
    return lifted


class RecordingSpeedHardwareSession(HardwareSession):
    """v5: the firmware speed byte matches how the 86 demonstrations were recorded.

    Recording (pico_follower.command_joints) sent MotionCtrl_2(0x01, 0x01, 50, 0x00);
    every rollout until now sent 5, because pi0_real_trial_v3/v4 constructed
    HardwareSession(speed_percent=5) and verified_hardware.py:454 sent that byte.
    Measured capability ratio between the two: 7-10x, which is the whole story of
    "太慢 / 拉不开抽屉 / 很抖" -- the client clamp was never what limited the arm.

    Only that one byte changes.  The attribute is swapped for the duration of the
    inherited call rather than _send_pair being copied, so the driver body and every
    guard around it stay byte-identical to verified_hardware.py.

    Home and prelift are unaffected in planning -- HardwareSession.home drives them
    through reset_piper_pose.move_slowly at joint_speed=.12 / gripper_speed=.006 -- so
    only the byte underneath changes and resetting stays as gentle as it was.

    PI0_FIRMWARE_SPEED=5 restores the old behaviour exactly.
    """

    FIRMWARE_SPEED = int(os.environ.get('PI0_FIRMWARE_SPEED', '50'))

    def _send_pair(self, target):
        previous, self.speed_percent = self.speed_percent, self.FIRMWARE_SPEED
        try:
            super()._send_pair(target)
        finally:
            self.speed_percent = previous



class TraceWriter:
    """Per-tick command/measurement log.  Writes nothing unless a path is set.

    `tick` is one motor write: the target and the state read at that same write,
    so both channels are sampled at one instant at 30 Hz.  `chunk` is one policy
    answer: its 10 rows are the target for the next 10 ticks before the slew
    clamp touches them, which is how the raw desired stream is recovered at 30 Hz
    even though inference only runs at 3 Hz.
    """

    def __init__(self):
        self.path = None
        self.handle = None
        self.ticks = 0
        self.chunks = 0
        self.tick_errors = 0
        self.last_error = None

    def open(self, path=None):
        if path is not None:
            self.path = path
        if self.path is not None and self.handle is None:
            self.handle = open(self.path, 'w')
        return self.handle is not None

    def _write(self, payload):
        if self.handle is None:
            return
        self.handle.write(json.dumps(payload, separators=(',', ':')) + '\n')
        self.handle.flush()

    def tick(self, target, measured):
        if self.handle is None:
            return
        self.ticks += 1
        # Called from write_state on the 30 Hz thread: a serialisation error here would
        # abort a motion run, so it is counted instead of raised, and the count is in
        # the report where zeros are visible.
        try:
            self._write({'t': round(time.monotonic(), 4),
                         'target': [round(float(v), 5) for v in target],
                         'measured': [round(float(v), 5) for v in measured]})
        except Exception as error:
            self.tick_errors += 1
            self.last_error = '%s: %s' % (type(error).__name__, error)

    def chunk(self, actions, tag):
        if self.handle is None:
            return
        self.chunks += 1
        self._write({'event': 'chunk', 'tag': tag, 't': round(time.monotonic(), 4),
                     'rows': [[round(float(v), 5) for v in row] for row in actions]})

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None


TRACE = TraceWriter()


class TracedHardwareSession(RecordingSpeedHardwareSession):
    """v7: v5/v6 plus the 30 Hz tick log.

    The bus is untouched: write_state already takes exactly one snapshot before
    every write, and the tracer only keeps the array that was already being read.
    A tick is logged only when a pair was really sent, so a write rejected by the
    slew guard leaves no row -- its target never reached the motors.
    """

    def __init__(self):
        super().__init__()
        self._trace_measured = None
        self._trace_pending = None

    def _snapshots(self, require_enabled=False, allow_estop=True):
        raw, meta = super()._snapshots(require_enabled, allow_estop)
        self._trace_measured = raw.copy()
        return raw, meta

    def _send_pair(self, target):
        super()._send_pair(target)
        self._trace_pending = self._trace_measured

    def write_state(self, state, recovery=False):
        self._trace_pending = None
        super().write_state(state, recovery)
        if not recovery and self._trace_pending is not None:
            TRACE.tick(self.last_command, self._trace_pending)


# ---- joint low-pass, lifted verbatim in spirit from the towel card ----------
# the towel card's src/lerobot/motors/piper/piper_slave.py keeps
#   filtered = alpha * new + (1 - alpha) * old
# on the six arm joints of each side, and passes the gripper through untouched
# ("never delay the gripper").  The towel inference launcher exposes it as
# PIPER_ACTION_FILTER_ALPHA and turns it off (1.0) when temporal ensembling is
# doing the smoothing.  This trial has no lerobot-side filter at all: whatever
# the server returns is written to the motors row by row, so every bit of
# high-frequency policy noise inside a chunk reaches the joints.  Alpha == 1
# reproduces the unfiltered stream exactly.
ACTION_FILTER_ALPHA = float(os.environ.get('PI0_ACTION_ALPHA', '0.25'))
ACTION_FILTER_GAP_S = float(os.environ.get('PI0_FILTER_GAP_S', '1.0'))
if not 0.0 < ACTION_FILTER_ALPHA <= 1.0:
    raise ValueError('PI0_ACTION_ALPHA must be in (0, 1]')
if ACTION_FILTER_GAP_S <= 0.0:
    raise ValueError('PI0_FILTER_GAP_S must be positive')

# CameraStreams._reader already converts OpenCV BGR to RGB. imencode requires
# BGR, and the server converts decoded BGR back to RGB. This conversion is
# mandatory: PI0_IMAGE_ORDER no longer changes the source/wire contract.
IMAGE_ORDER = 'rgb'
IMAGE_FLIP = True


class RemotePolicy:
    """The A10 policy over the same JSON protocol the 20261013 trial used."""
    compat_error = None

    def __init__(self, url, expected_checkpoint, can1_only=False, hold_target=None):
        self.url = url
        self.expected_checkpoint = expected_checkpoint
        self.can1_only = can1_only
        self.hold_target = hold_target
        self.checkpoint = None
        self.checkpoint_sha256 = None
        self.inference_s = []
        self.server_ensemble = None
        self.max_active_target_delta = 0.
        self.excluded_can0_actions = []
        self._filtered_joints = None
        self._filtered_at = 0.0
        self.filter_stats = {'alpha': ACTION_FILTER_ALPHA,
                             'gap_s': ACTION_FILTER_GAP_S,
                             'joints_only': True,
                             'enabled': ACTION_FILTER_ALPHA < 1.0,
                             'rows_filtered': 0, 'reseeds': 0}

    def _apply_action_filter(self, actions):
        """Low-pass the joints of every returned row, exactly as the towel card's
        piper_slave does (filtered = alpha*new + (1-alpha)*old).  Grippers are
        copied through unchanged: a delayed gripper only shortens a grasp window.

        The state is carried across chunks on purpose -- the seam between the
        tenth row of one chunk and the first row of the next is a step the filter
        would otherwise pass through at full amplitude.  It is re-seeded whenever
        more than PI0_FILTER_GAP_S has passed since the last call, so the passive
        pre-motion peek (taken at home, before the prelift) cannot drag the start
        of the rollout towards the home pose."""
        self.filter_stats['rows_filtered'] += int(len(actions))
        if ACTION_FILTER_ALPHA >= 1.0:
            return actions
        now = time.monotonic()
        seed = self._filtered_joints is None or (now - self._filtered_at) > ACTION_FILTER_GAP_S
        if seed:
            self.filter_stats['reseeds'] += 1
        joints = control.JOINTS
        output = np.array(actions, dtype=np.float64, copy=True)
        carried = None if seed else self._filtered_joints
        for row in output:
            if carried is None:
                carried = row.copy()
            else:
                joints_now = (ACTION_FILTER_ALPHA * row[joints]
                              + (1.0 - ACTION_FILTER_ALPHA) * carried[joints])
                carried = row.copy()      # grippers always come from this row
                carried[joints] = joints_now
            row[:] = carried
        self._filtered_joints = carried
        self._filtered_at = now
        return output

    def infer(self, obs):
        import cv2
        images = {}
        for name, frame in obs['images'].items():
            # Input is RGB; OpenCV JPEG encoding takes BGR.
            wire = np.ascontiguousarray(frame[:, :, ::-1])
            ok, data = cv2.imencode('.jpg', wire, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('Camera encoding failed')
            images[name] = base64.b64encode(data).decode()
        data = json.dumps({'state': obs['state'].tolist(), 'images': images}).encode()
        request = urllib.request.Request(self.url, data, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
        if 'error' in result:
            raise RuntimeError(result['error'])
        ensemble = result.get('ensemble')
        if not isinstance(ensemble, dict) or ensemble.get('enabled') is not False:
            raise RuntimeError('v8 requires a verified --no-ensemble server; received: '
                               + repr(ensemble))
        self.server_ensemble = dict(ensemble)
        if result.get('checkpoint') != self.expected_checkpoint:
            raise RuntimeError(f"Remote checkpoint mismatch: {result.get('checkpoint')}")
        self.checkpoint = result['checkpoint']
        self.checkpoint_sha256 = result.get('checkpoint_sha256')
        self.inference_s.append(result.get('inference_s'))
        actions = np.array(result['actions'], dtype=np.float64)
        if actions.shape != (10, 14) or not np.isfinite(actions).all():
            raise RuntimeError('Invalid remote actions')
        TRACE.chunk(actions, 'raw')
        actions = self._apply_action_filter(actions)
        if self.can1_only:
            # This is a single-arm test. No can0 policy target enters the controller.
            # The active can1 targets still pass the unchanged full validation.
            self.excluded_can0_actions.append(actions[:, 7:].tolist())
            actions[:, 7:] = self.hold_target[7:]
            validated, _ = control.validated_actions(actions)
            delta = float(np.max(np.abs(validated[:, :6] - self.hold_target[:6])))
            self.max_active_target_delta = max(self.max_active_target_delta, delta)
            if delta > .035:
                raise RuntimeError(f'Single-arm bench target exceeds .035 rad home envelope: {delta}')
        TRACE.chunk(actions, 'sent')
        return actions

    def close(self):
        pass


def one_block_trial(hw, cameras, policy, executor, target, prompt, report):
    observed_at = time.monotonic()
    actions, _, _, _, ages, _ = control.sample_valid_actions(
        executor, policy, hw, cameras, target, prompt, 3., observed_at + 3., 1, report,
        'single_arm_post_home')
    if max(ages.values()) + time.monotonic() - observed_at > .6:
        raise RuntimeError('Single-arm policy observation expired')
    interval = 1. / 30
    epoch = time.monotonic()
    commanded = target.copy()
    last_write = None
    report['rollout'].update({'chunks': 1, 'control_hz': 30, 'max_joint_tracking_error_rad': 0.,
                              'max_joint_measured_displacement_rad': 0., 'duration_s': 0.})
    for row in actions:
        tick = epoch + report['rollout']['action_rows_sent'] * interval
        while time.monotonic() < tick:
            control.control_watchdog(hw, cameras, commanded)
            time.sleep(min(.003, max(0, tick - time.monotonic())))
        measured, error = control.control_watchdog(hw, cameras, commanded)
        now = time.monotonic()
        if now - tick >= interval:
            raise RuntimeError('Control tick missed; no catch-up writes')
        if max(ages.values()) + now - observed_at > .6:
            raise RuntimeError('Single-block policy observation expired during execution')
        if np.max(np.abs(measured[:6] - target[:6])) > .04:
            raise RuntimeError('Single-arm measured displacement exceeded .04 rad')
        dt = interval if last_write is None else min(interval, now - last_write)
        command = control.limited_target(commanded, row, dt, .12, .006)
        if time.monotonic() - tick >= interval:
            raise RuntimeError('Control tick expired before write')
        hw.write_state(command)
        commanded = command
        last_write = time.monotonic()
        report['rollout']['action_rows_sent'] += 1
        report['rollout']['max_joint_tracking_error_rad'] = max(
            report['rollout']['max_joint_tracking_error_rad'],
            float(error[control.JOINTS].max()))
        report['rollout']['max_joint_measured_displacement_rad'] = max(
            report['rollout']['max_joint_measured_displacement_rad'],
            float(np.abs(measured[:6] - target[:6]).max()))
        report['rollout']['last_command'] = command.tolist()
    report['rollout']['duration_s'] = time.monotonic() - epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--remote-url', default='http://127.0.0.1:18024/')
    parser.add_argument('--expected-remote-checkpoint', default=ARMFIX_CHECKPOINT,
                        help='The resume-20261013 checkpoint is %s' % RESUME_CHECKPOINT)
    parser.add_argument('--can1-only', action='store_true')
    parser.add_argument('--pre-motion-sample', action='store_true',
                        help='Strict passive pre-motion sample that raises on a gate violation '
                             '(expected to fail from home, where both arms sit on the stops).')
    parser.add_argument('--no-prelift', action='store_true',
                        help='Skip the J2/J3 prelift and roll out straight from home.')
    parser.add_argument('--rollout-seconds', type=float, default=60.,
                        help='Motion deadline for continuous_rollout, seconds. The demonstrations '
                             'run 22-39 s, so anything below ~40 s cannot show a whole task.')
    parser.add_argument('--trajectory', type=Path, default=None,
                        help='JSONL of the measured state each time a chunk is prefetched, so the '
                             'rollout can be compared against the demonstrations afterwards.')
    parser.add_argument('--trace', type=Path, default=None,
                        help='JSONL of every written target with the state read at '
                             'that write, plus one line per policy chunk (raw and '
                             'sent).  Off unless given.')
    parser.add_argument('--frames', type=Path, default=None,
                        help='Directory for one JPEG per camera per frame interval -- '
                             'the frames the policy was handed during the run, not the '
                             'pre-home peek.  Off unless given.')
    parser.add_argument('--frame-interval', type=float,
                        default=float(os.environ.get('PI0_FRAME_INTERVAL', '1.0')),
                        help='Seconds between saved frame sets.')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.frames is not None:
        args.frames.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'motion' if args.enable_motion else 'dryrun', 'task_success': None,
              'model_slot_can_order': ['can1', 'can0'], 'rollout': {'chunks': 0, 'action_rows_sent': 0},
              'rejected_policy_chunks': [], 'rejected_policy_chunks_total': 0,
              'gripper_values_clamped': 0, 'joint_values_clamped': 0,
              'checkpoint_sha256': None, 'remote_url': args.remote_url,
              'checkpoint': args.expected_remote_checkpoint,
              'chunk_execution': 'first 10 rows, replan at 30 Hz', 'policy_command_pairs': 0}
    # Gripper rate limit.  .006 m/s needs ~12 s to cross the 0.07 m stroke, so a short
    # "open" pulse from the policy (10 rows = 0.33 s buys 0.002 m) is averaged away
    # before it can move anything.  PI0_GRIPPER_RATE=off drops the rate clamp; the
    # hardware range clip to [0, .07] inside limited_target still applies.  The joint
    # v3 left the joint rate at .12 rad/s; v4 raises both clamps (see below).
    # v4: both clamps raised to just inside verified_hardware.py's own slew guard.
    #   guard joints  .25*dt + .01  rad -> .55 rad/s at 30 Hz;  v4 asks .40
    #   guard gripper .006*dt + .001 m  -> .036 m/s at 30 Hz;  v4 asks .030
    # v3 asked .12 and .006, i.e. 4.6x and 6x inside the guard, while the 86
    # demonstrations run A.J2 at mean .108 / p99 1.126 rad/s (20.6% of frames already
    # over .12) and move the gripper at mean .007 / p99 .258 m/s -- a full .07 stroke
    # in .15 s against 11.7 s at .006.  A clamp pinned at its ceiling emits a bang-bang
    # command: that is the dead-straight 12 s gripper ramp the 20261015_104646 run
    # opened with, and the .243 reversal rate against the demonstrations' .003.
    _grip = (os.environ.get("PI0_GRIPPER_RATE", ".030").strip().lower() or ".030")
    gripper_rate = float("inf") if _grip in ("off", "none", "inf") else float(_grip)
    params = SimpleNamespace(camera_max_age=float(os.environ.get("PI0_CAMERA_MAX_AGE", ".6")),
                             prefetch_after_rows=int(os.environ.get("PI0_PREFETCH_AFTER_ROWS", "8")),
                             inference_timeout=3., max_policy_samples=1,
                             joint_rate=float(os.environ.get("PI0_JOINT_RATE", ".40")),
                             gripper_rate=gripper_rate)
    if not 0 < params.camera_max_age <= .6:
        raise ValueError('v8 requires PI0_CAMERA_MAX_AGE in (0, .6]')
    if not 1 <= params.prefetch_after_rows < 10:
        raise ValueError('PI0_PREFETCH_AFTER_ROWS must be in 1..9')
    hardware = executor = None
    started = time.monotonic()
    # The report only keeps the final state, which cannot answer "did the task
    # happen" -- the demonstrations lift J2 to 1.8 rad and open both grippers to
    # 0.070, so the run has to be readable as a trajectory.  continuous_rollout
    # already exposes its prefetch hook; this records through it rather than
    # touching the verified control module.
    trajectory = []

    frames_written = {'last': -1e9, 'count': 0, 'errors': 0, 'last_error': None}

    def record_observation(obs, ages):
        now = time.monotonic()
        if args.trajectory:
            trajectory.append({'t': round(now - started, 3),
                               'state': [round(float(v), 4) for v in obs['state']],
                               'camera_ages_s': {name: round(value, 3)
                                                 for name, value in ages.items()}})
        # Diagnostic only, and this hook is called unwrapped on the critical path: a
        # frame write that fails or slows down must never disturb the motion loop.
        if args.frames is not None and now - frames_written['last'] >= args.frame_interval:
            frames_written['last'] = now
            try:
                for name, image in obs['images'].items():
                    path = args.frames / ('t%07.2f_%s.jpg' % (now - started, name))
                    if cv2.imwrite(str(path), image[:, :, ::-1]):
                        frames_written['count'] += 1
            except Exception as error:
                frames_written['errors'] += 1
                frames_written['last_error'] = '%s: %s' % (type(error).__name__, error)
    resources = contextlib.ExitStack()
    try:
        # cubephase fork: PI0_HOME_JSON redirects the start pose.  Unset means
        # byte-identical v8 behaviour (the live pi0_drawer_mixed_74_v1 home).
        _home_json = os.environ.get('PI0_HOME_JSON')
        target, report['home_source'] = control.home_target(
            Path(_home_json).expanduser() if _home_json
            else control.ROOT / 'inference/home_target.json')
        policy = RemotePolicy(args.remote_url, args.expected_remote_checkpoint,
                              args.can1_only, target.copy())
        if args.can1_only:
            report['trial_scope'] = 'one fresh 10-row can1 pi0 block; can0 held at verified home'
        report['compatibility_max_error'] = policy.compat_error
        report['action_filter'] = policy.filter_stats
        report['knobs'] = {
            'trial': str(Path(__file__).resolve()),
            'firmware_speed_percent': RecordingSpeedHardwareSession.FIRMWARE_SPEED,
            'joint_rate_rad_s': params.joint_rate,
            'gripper_rate_m_s': ('off(inf)' if gripper_rate == float('inf') else gripper_rate),
            'gripper_rate_env': os.environ.get('PI0_GRIPPER_RATE'),
            'prelift_rad': PRELIFT_J2_RAD,
            'action_alpha': ACTION_FILTER_ALPHA,
            'filter_gap_s': ACTION_FILTER_GAP_S,
            'camera_max_age_s': params.camera_max_age,
            'image_order': IMAGE_ORDER,
            'image_flip_before_jpeg': IMAGE_FLIP,
            'rollout_seconds': args.rollout_seconds,
            'traced_session': True,
            'frames_dir': str(args.frames) if args.frames is not None else None}
        print(json.dumps({'event': 'knobs', **report['knobs']}), flush=True)
        report['trial_file'] = __file__
        report['fps'], prompt = control.metadata(control.ROOT)
        if TRACE.open(args.trace):
            resources.callback(TRACE.close)
            print(json.dumps({'event': 'trace_open', 'path': str(TRACE.path)}), flush=True)
        # TracedHardwareSession, not RecordingSpeedHardwareSession: the tracer is what
        # records the target/measured pair at every write.  It only adds read-only
        # bookkeeping around the inherited calls -- the driver body, the slew guard and
        # every gate stay byte-identical, and a write the guard rejects leaves no tick
        # line because no pair was sent.
        hardware = resources.enter_context(TracedHardwareSession())
        report['passive_gate'] = hardware.preflight(2, allow_estop=False)
        report['camera_mapping'] = control.verify_camera_mapping()
        cameras = resources.enter_context(control.CameraStreams(.6))
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pi0-inference')
        # At home both arms sit exactly on the J2 floor and the J3 ceiling, where
        # the policy's residual has no room and the gate legitimately rejects it
        # (measured: 7 of 8 held-out episodes from home, worst 0.095 rad against a
        # 0.05 tolerance).  Pass --pre-motion-sample for the strict check that
        # raises; the default only records what the policy would have asked for,
        # because the pose that matters is the prelift one, sampled by the rollout.
        measured_peek = control.vector(hardware.read_state(require_enabled=False), 'pre_motion peek')
        if args.pre_motion_sample:
            sample = control.sample_valid_actions(executor, policy, hardware, cameras, None, prompt,
                                                  3., time.monotonic() + 3., 1, report, 'pre_motion')
            report['pre_motion_first_action'] = sample[0][0].tolist()
        else:
            obs, ages = control.observation(cameras, measured_peek, prompt)
            raw = policy.infer(obs)
            excess = np.maximum(control.LOW[control.JOINTS] - raw[:, control.JOINTS],
                                raw[:, control.JOINTS] - control.HIGH[control.JOINTS])
            report['pre_motion_peek'] = {
                'first_action': raw[0].tolist(),
                'worst_excess_rad': float(excess.max()),
                'would_be_rejected': bool((excess > .05).any()),
                'violating_row_slot': [[int(row), int(control.JOINTS[column])]
                                       for row, column in np.argwhere(excess > .05)],
                'state': measured_peek.tolist(), 'camera_ages_s': dict(ages),
                'checkpoint_sha256': policy.checkpoint_sha256}
            print(json.dumps({'event': 'pre_motion_peek',
                              'worst_excess_rad': report['pre_motion_peek']['worst_excess_rad'],
                              'would_be_rejected': report['pre_motion_peek']['would_be_rejected'],
                              'violating_row_slot': report['pre_motion_peek']['violating_row_slot']}),
                  flush=True)
        report['checkpoint_sha256'] = policy.checkpoint_sha256
        images, _ = cameras.snapshot()
        import cv2
        for name, image in images.items():
            cv2.imwrite(str(args.report.with_name(args.report.stem + '_' + name + '.png')),
                        image[:, :, ::-1])
        if args.enable_motion:
            report['pre_home_gate'] = hardware.preflight(2, allow_estop=False)
            report['home'] = hardware.home(target, timeout=60)
            control.settle_home(hardware, target, timeout=5)
            if args.no_prelift:
                start_pose = target.copy()
                report['prelift'] = {'skipped': True}
            else:
                start_pose = prelift_target(target)
                report['prelift_target'] = start_pose.tolist()
                report['prelift_delta'] = {'j2_slots': list(PRELIFT_J2_SLOTS), 'j2_rad': PRELIFT_J2_RAD,
                                           'j3_slots': list(PRELIFT_J3_SLOTS), 'j3_rad': -PRELIFT_J3_RAD}
                print(json.dumps({'event': 'prelift_start', 'target': start_pose.tolist()}), flush=True)
                report['prelift'] = hardware.home(start_pose, timeout=60)
                prelift_measured, prelift_error = control.settle_home(hardware, start_pose, timeout=5)
                report['prelift_measured'] = prelift_measured.tolist()
                report['prelift_max_joint_error_rad'] = float(np.max(prelift_error[control.JOINTS]))
                print(json.dumps({'event': 'prelift_done',
                                  'measured': prelift_measured.tolist(),
                                  'max_joint_error_rad': report['prelift_max_joint_error_rad']}), flush=True)
            baseline = hardware.command_pairs
            try:
                if args.can1_only:
                    one_block_trial(hardware, cameras, policy, executor, start_pose.copy(), prompt, report)
                else:
                    control.continuous_rollout(hardware, cameras, policy, executor, start_pose.copy(), prompt,
                                               time.monotonic() + args.rollout_seconds, params, report,
                                               on_observation=record_observation if (args.trajectory or args.frames is not None) else None)
            finally:
                report['policy_command_pairs'] = hardware.command_pairs - baseline
            report['result'] = 'bounded_trial_finished_task_outcome_unverified'
        else:
            report['result'] = 'dryrun_passed_no_commands'
    except BaseException as error:
        report['result'] = 'blocked_or_aborted'
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if hardware:
            report['final_hold'] = hardware.hold()
            report['command_pairs'] = hardware.command_pairs
        if 'policy' in locals():
            report['remote_checkpoint_verified'] = policy.checkpoint
            report['remote_ensemble_verified'] = policy.server_ensemble
            report['remote_checkpoint_sha256'] = policy.checkpoint_sha256
            report['max_active_target_delta_from_home_rad'] = policy.max_active_target_delta
            report['excluded_can0_raw_policy_actions'] = policy.excluded_can0_actions
            if policy.inference_s:
                report['remote_inference_s'] = policy.inference_s
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        resources.close()
        report['elapsed_s'] = time.monotonic() - started
        if args.trajectory and trajectory:
            args.trajectory.parent.mkdir(parents=True, exist_ok=True)
            args.trajectory.write_text(''.join(json.dumps(row) + '\n' for row in trajectory))
            report['trajectory_path'] = str(args.trajectory)
            report['trajectory_samples'] = len(trajectory)
        if args.trace:
            report['trace_path'] = str(args.trace)
            report['trace_ticks'] = TRACE.ticks
            report['trace_chunks'] = TRACE.chunks
            report['trace_tick_errors'] = TRACE.tick_errors
            report['trace_last_error'] = TRACE.last_error
        if args.frames is not None:
            report['frames_path'] = str(args.frames)
            report['frames_saved'] = frames_written['count']
            report['frames_errors'] = frames_written['errors']
            report['frames_last_error'] = frames_written['last_error']
            report['frame_interval_s'] = args.frame_interval
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report.get(k) for k in
                          ['result', 'error', 'checkpoint', 'remote_checkpoint_sha256',
                           'command_pairs', 'policy_command_pairs', 'home', 'prelift',
                           'prelift_max_joint_error_rad', 'rollout', 'final_hold']}, indent=2), flush=True)
    if report['result'] == 'blocked_or_aborted':
        raise SystemExit(2)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, control.interrupted)
    signal.signal(signal.SIGTERM, control.interrupted)
    main()
