#!/usr/bin/env python3
"""回放服务器：顶替 π0 推理服务，用**时间重参数化过的**录制轨迹回答 /infer。

为什么是"顶替服务器"而不是改 trial
----------------------------------
用户要"就算抽屉没拉开，方块臂也要把方块夹住放进去"。那个"必须先拉开抽屉"的
限制不是代码，是策略学到的时序程序（活机器上它只在抽屉臂松爪后 1.97 s 才让
方块臂起动）。唯一真做法是不走策略，把演示的方块相位原样放给它。

让机器人那一侧的代码**一字不动**是这里的关键：客户端（pi0_real_trial_v8.py，
含全部闸门、限速、跟踪校验、归位、报告、trace）就是活的 v8 原版，我只让它在
--url 上连到这个按轨迹回答的本地服务。所以每条安全闸门都还是原版的。

而"原样放"过不去限速器（见 make_traj.py 的说明：演示那一夹超帽 5.6~12.4 倍，
手臂到位与合爪被拉成两条时间线）。所以轨迹在 make_traj.py 里已经重采样到一条
**共享时间刻度**上——所有轴永远在同一参照时刻，相对时序逐字保住，且每拍步长都在
帽内。本服务只是把那份序列按 10 行一块发出去。

协议（与策略服务端相同）
----------------------------------------
  请求  {"state": [14 个 float], "images": {名字: base64 jpg}}
  答复  {"actions": [[14] x 10], "checkpoint": <replay_id>,
         "checkpoint_sha256": <npz 的 sha256>, "inference_s": <float>,
         "ensemble": {"enabled": false, ...}, "replay": {...}}
        或 {"error": "..."} + HTTP 500

三条只属于本服务的约定（都必须印在报告和 README 里）
----------------------------------------------------
1. **抽屉臂（槽 0..6）不回放，钉在起跑位姿**：发给它的 7 个目标每拍都是
   home 的前 7 维。于是它停在归位点上不动，也不会去动上游，闸门也不会因为它卡住。
   （早期版本是"回声它自己的实测位姿"——那等于把限速器接成一个带增益的反馈环，
   跟踪误差会让手臂慢慢爬走；钉常量目标才是干净的位置保持。）
   只有槽 7..13 来自轨迹。
2. **起点靠 arming 判据对齐**：客户端在真动之前会先打一次被动 peek
   （此时手臂还在上一轮留下的位姿），接着才归位。所以回放游标不是"第一次请求就
   开跑"，而是**第一次出现"状态匹配起跑位姿"的请求**才置 0。判据要求同时满足：
      L∞((state - traj_state[0])[7:14]) < --arm-tol   且
      |state[1] - home[1]|               < --arm-j2-tol
   **只比方块臂那 7 维**：演示第 0 拍抽屉臂还夹在抽屉拉环上（离 home 0.33+ rad），
   把全 14 维算进去判据永远不成立、服务端会一直不 arm。
   state[1] 是抽屉臂 J2，归位到 home 里那个值；上一轮跑完手臂停的地方离它有
   1.2+ rad，所以这个"又紧又只钉一维"的判据把 peek 和回放起拍干净地分开。
   一旦 arm 过就不再重新对齐（回放没有"中途换块"这回事）。
   到 --arm-deadline-calls 次请求仍未 arm ⇒ 直接回 error 让 trial 中止，
   而不是静默地把手臂按在起始位姿上不动。
3. **轨迹首部前置了 10 拍"保持不动"**（make_traj.py 的 --lead-hold）：如果被动
   peek 恰好也在归位位姿上打了一次，第一个 10 行块会被 peek 吃掉，前置行让这种
   "跳掉 10 行"退化成无害的 0.33 s 静止。

**本服务不加载任何权重、不做任何推理。** 每一行动作都逐字来自一份录制演示
（只经过了时间重参数化：所有轴共享同一参照时刻，没有任何一轴被单独重排）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROWS = 10
FPS = 30.0
CUBE_ARM = slice(7, 14)     # can0 = 方块臂
DRAWER_ARM = slice(0, 7)    # can1 = 抽屉臂


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traj', default=os.path.join(here, 'cube_phase_traj.npz'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18032)
    p.add_argument('--arm-tol', type=float, default=0.25,
                   help='arming 判据：方块臂 7 维 L∞(state - traj_state[0]) 必须小于这个')
    p.add_argument('--arm-j2-tol', type=float, default=0.06,
                   help='arming 判据：抽屉臂 |state[1] - home[1]| 必须小于这个（rad）')
    p.add_argument('--arm-deadline-calls', type=int, default=12,
                   help='第几次请求还没 arm 就直接报错（0 = 不设限，仅告警）')
    p.add_argument('--drawer-hold', default='home',
                   help="'home' = 抽屉臂钉在 meta 的 hold_drawer_target；"
                        "'measured' = 回声它的实测位姿（调试用，会爬走）")
    p.add_argument('--log', default='', help='可选的逐请求 JSONL')
    return p.parse_args(argv)


class Replay:
    def __init__(self, args):
        self.args = args
        with np.load(args.traj, allow_pickle=False) as z:
            self.action = np.asarray(z['action'], dtype=np.float64)
            self.state = np.asarray(z['state'], dtype=np.float64)
            self.meta = json.loads(str(z['meta']))
        if self.action.ndim != 2 or self.action.shape[1] != 14:
            raise SystemExit('FAIL action 形状 %s 不是 (H,14)' % (self.action.shape,))
        if self.state.shape != self.action.shape:
            raise SystemExit('FAIL state 形状 %s 与 action %s 不一致'
                             % (self.state.shape, self.action.shape))
        if not (np.isfinite(self.action).all() and np.isfinite(self.state).all()):
            raise SystemExit('FAIL 轨迹里有非有限值')
        self.H = int(self.action.shape[0])
        with open(args.traj, 'rb') as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()
        self.replay_id = str(self.meta['replay_id'])
        self.start_state = self.state[0]
        self.home = np.asarray(self.meta['home_target'], dtype=np.float64)
        if self.home.shape != (14,):
            raise SystemExit('FAIL meta.home_target 不是 14 维')
        self.hold = np.asarray(self.meta['hold_drawer_target'], dtype=np.float64)
        if self.hold.shape != (7,):
            raise SystemExit('FAIL meta.hold_drawer_target 不是 7 维')

        self.calls = 0
        self.armed = False
        self.cursor = 0
        self.last_at = None
        self.armed_at_call = None
        self.armed_gap_s = None
        self.done_logged = False
        self.lock = threading.Lock()
        self.logfh = None
        if args.log:
            self.logfh = open(args.log, 'a', buffering=1)

    # --- 逐请求 ---------------------------------------------------------------
    def answer(self, state):
        if state.shape != (14,) or not np.isfinite(state).all():
            return {'error': 'replay server: state must be 14 finite floats'}
        with self.lock:
            self.calls += 1
            n = self.calls
            now = time.monotonic()
            gap = None if self.last_at is None else now - self.last_at
            self.last_at = now

            # 只比方块臂那 7 维（见文件头第 2 条：演示第 0 拍抽屉臂还夹着抽屉拉环）
            dist_cube = float(np.abs(state[CUBE_ARM] - self.start_state[CUBE_ARM]).max())
            dist_j2 = float(abs(state[1] - self.home[1]))
            matched = dist_cube < self.args.arm_tol and dist_j2 < self.args.arm_j2_tol
            armed_now = False
            if not self.armed and matched:
                self.armed = True
                armed_now = True
                self.armed_at_call = n
                self.armed_gap_s = gap
                self.cursor = 0
                print('REPLAY_ARMED call=%d gap_s=%s dist_cube=%.4f dist_j2=%.4f '
                      '(游标置 0，回放从这里起)'
                      % (n, 'n/a' if gap is None else '%.2f' % gap, dist_cube, dist_j2),
                      flush=True)

            if (not self.armed and self.args.arm_deadline_calls > 0
                    and n >= self.args.arm_deadline_calls):
                msg = ('replay never armed after %d calls: 手臂没到过起跑位姿 '
                       '(need 方块臂 L∞<%.3f 且 |state[1]-%.4f|<%.3f; '
                       'now dist_cube=%.4f dist_j2=%.4f)。trial 会在这里中止，'
                       '手臂保持不动。'
                       % (n, self.args.arm_tol, float(self.home[1]),
                          self.args.arm_j2_tol, dist_cube, dist_j2))
                print('REPLAY_NOT_ARMED ' + msg, flush=True)
                self._log({'call': n, 'gap_s': gap, 'dist_cube': dist_cube,
                           'dist_j2': dist_j2, 'matched': matched, 'armed': False,
                           'error': msg})
                return {'error': msg}

            k = self.cursor
            idx = np.minimum(k + np.arange(ROWS), self.H - 1)
            rows = np.empty((ROWS, 14), dtype=np.float64)
            if self.args.drawer_hold == 'measured':
                rows[:, DRAWER_ARM] = state[DRAWER_ARM]
            else:
                rows[:, DRAWER_ARM] = self.hold        # 钉常量：干净的位置保持
            rows[:, CUBE_ARM] = self.action[idx, CUBE_ARM]
            if self.armed:
                self.cursor = min(k + ROWS, self.H)
            if self.cursor >= self.H and not self.done_logged:
                self.done_logged = True
                print('REPLAY_DONE 轨迹放完（%d 拍 = %.2f s），之后每拍保持末行'
                      % (self.H, self.H / FPS), flush=True)
            window = [int(idx[0]), int(idx[-1])]
            self._log({'call': n, 'gap_s': gap, 'dist_cube': dist_cube, 'dist_j2': dist_j2,
                       'matched': matched, 'armed': self.armed, 'cursor_would_be': k,
                       'cursor_next': self.cursor, 'window': window,
                       'observed_state': state.tolist()})
            if n <= 4 or armed_now or (self.cursor >= self.H):
                print('REPLAY_REQ call=%d gap_s=%s dist_cube=%.4f dist_j2=%.4f armed=%s '
                      'window=[%d..%d] cursor->%d'
                      % (n, 'n/a' if gap is None else '%.2f' % gap, dist_cube, dist_j2,
                         self.armed, window[0], window[1], self.cursor), flush=True)
            return {'rows': rows, 'cursor': self.cursor, 'armed': self.armed,
                    'window': window, 'call': n}

    def _log(self, record):
        if self.logfh is None:
            return
        record['t'] = time.time()
        self.logfh.write(json.dumps(record, sort_keys=True) + '\n')

    def status(self):
        return {'service': 'replay', 'model': None, 'replay_id': self.replay_id,
                'checkpoint_sha256': self.sha256, 'horizon': self.H,
                'horizon_seconds': self.H / FPS, 'armed': self.armed,
                'armed_at_call': self.armed_at_call, 'cursor': self.cursor,
                'calls': self.calls, 'source_file': self.meta.get('source_file'),
                'drawer_hold': self.args.drawer_hold,
                'stretch_factor_total': self.meta.get('stretch_factor_total'),
                'start_state_l_inf_to_home_cube_arm':
                    self.meta.get('start_state_l_inf_to_home_cube_arm')}


def make_handler(replay, args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'replay-server/2'

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, replay.status())

        def do_POST(self):
            t0 = time.monotonic()
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            try:
                request = json.loads(raw)
                state = np.asarray(request['state'], dtype=np.float64)
            except Exception as error:
                self._send(400, {'error': 'replay server: bad request: %s' % error})
                return
            answer = replay.answer(state)
            if 'error' in answer:
                self._send(500, {'error': answer['error'], 'checkpoint': replay.replay_id})
                return
            self._send(200, {
                'actions': answer['rows'].tolist(),
                'inference_s': time.monotonic() - t0,
                'checkpoint': replay.replay_id,
                'checkpoint_sha256': replay.sha256,
                'ensemble': {'enabled': False, 'coeff': 0.0, 'horizon': 0,
                             'step': answer['cursor'], 'covering': 0, 'mode': 'replay'},
                'replay': {'armed': answer['armed'], 'call': answer['call'],
                           'cursor': answer['cursor'], 'window': answer['window'],
                           'horizon': replay.H, 'source': replay.meta.get('source_file')},
            })

        def log_message(self, fmt, *a):   # 默认每请求两行太吵，收进自己的日志
            pass

    return Handler


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.traj):
        raise SystemExit('FAIL 缺轨迹 %s（先跑 make_traj.py）' % args.traj)
    replay = Replay(args)
    print('PI0_REMOTE_READY replay %s on http://%s:%d/' % (replay.replay_id, args.host, args.port),
          flush=True)
    print('PI0_REMOTE_CHECKPOINT %s' % replay.replay_id, flush=True)
    print('PI0_REMOTE_SHA256 %s' % replay.sha256, flush=True)
    print('PI0_REMOTE_ENSEMBLE off', flush=True)
    print('PI0_REMOTE_WARMUP 0.0', flush=True)
    print('REPLAY_TRAJ ep=%s H=%d (%.2f s, 原 %s 拍, 拉伸 %.2f 倍, 前置 %s 拍) '
          'drawer_hold=%s arm_tol=%.3f arm_j2_tol=%.3f (home[1]=%.4f)'
          % (replay.meta.get('source_file'), replay.H, replay.H / FPS,
             replay.meta.get('horizon_raw'), float(replay.meta.get('stretch_factor_total', 0)),
             replay.meta.get('lead_hold_rows'), args.drawer_hold, args.arm_tol,
             args.arm_j2_tol, float(replay.home[1])), flush=True)
    print('!!! 本服务不跑模型：每一行动作逐字来自 %s（仅经时间重参数化：所有轴共享同一参照时刻）。'
          '抽屉臂 7 位钉在起跑位姿不动。' % replay.meta.get('source_file'), flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(replay, args))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
