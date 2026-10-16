#!/usr/bin/env python3
"""回放服务器：顶替远端推理服务，用**时间重参数化过的**录制轨迹回答 /infer。

这是"全任务"版：把演示里完整的一套动作（today99 / episode_000041 第 48..750 拍）
原样放出去 ——

    相位 A  抽屉臂伸手 -> 张爪 -> 咬住抽屉拉环 -> 拉开约 11 cm -> 松爪
    相位 B  方块臂伸过去夹起蓝方块、移到抽屉口松爪放下
    相位 C  抽屉臂再次前伸/退回（爪张着）= 把抽屉合上
    相位 D  两条臂平滑回到复位目标（**这一段语料里没有**，是 make_full_traj.py
            按"最后自动归位"的要求补的余弦缓动），再保持 1 s

让机器人那一侧的代码**一字不动**是这里的关键：客户端（含全部闸门、限速、跟踪校验、
归位、报告、trace）就是活的原版，我只让它在 --remote-url 上连到这个按轨迹回答的
本地服务。

协议（与机器上那个远端推理服务相同）
------------------------------------
  请求  {"state": [14 个 float], "images": {名字: base64 jpg}}
  答复  {"actions": [[14] x 10], "checkpoint": <replay_id>,
         "checkpoint_sha256": <npz 的 sha256>, "inference_s": <float>,
         "ensemble": {"enabled": false, ...}, "replay": {...}}
        或 {"error": "..."} + HTTP 500

三条只属于本服务的约定（都必须印在报告和 README 里）
----------------------------------------------------
1. **两条臂都由轨迹供行**。这与 cubephase 那份最大的区别：那份把抽屉臂（槽 0..6）
   钉在归位点上（它要动的是方块臂），这份两条臂都按轨迹走，因为它们都要动。
2. **arming 判据比全 14 维**：`L∞(state - traj_state[0]) < --arm-tol`。
   演示第 0 拍两条臂都坐在起跑位姿上，比全 14 维才正确，也才拦得住被动 peek
   （peek 发生在客户端归位之前，手臂还在上一轮留下的位姿上）。
   一旦 arm 过就不再重新对齐（回放没有"中途换块"这回事）。
   到 --arm-deadline-calls 次请求仍未 arm ⇒ 直接回 error 让 trial 中止，
   而不是静默地把手臂按在起始位姿上不动。
3. **轨迹首部前置了 10 拍"保持不动"**（make_full_traj.py 的 --lead-hold）：如果被动
   peek 恰好也在起跑位姿上打了一次，第一个 10 行块会被 peek 吃掉，前置行让这种
   "跳掉 10 行"退化成无害的 0.33 s 静止。**这一份末尾会自动归位**，所以跑完之后
   手臂就坐在起跑位姿上 —— 原地再跑一次时 peek 会立刻 arm，同样由这 10 拍吸收。

**本服务不加载任何权重、不做任何推理。** 除末尾那一段归位外，每一行动作都逐字来自
一份录制演示（只经过了时间重参数化：所有轴共享同一参照时刻，没有任何一轴被单独重排）。
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
DRAWER_ARM = slice(0, 7)    # can1 = 抽屉臂
CUBE_ARM = slice(7, 14)     # can0 = 方块臂


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traj', default=os.path.join(here, 'full_traj.npz'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18315)
    p.add_argument('--arm-tol', type=float, default=0.10,
                   help='arming 判据：全 14 维 L∞(state - traj_state[0]) 必须小于这个。'
                        '客户端复位后 settle_home 只保证关节 0.035 rad / 夹爪 0.004 m，'
                        '所以 0.10 有约 3 倍余量；而 peek 的位姿（上一轮停的地方）'
                        '通常离起跑位姿 1 rad 量级，拦得住。')
    p.add_argument('--arm-deadline-calls', type=int, default=20,
                   help='第几次请求还没 arm 就直接报错（0 = 不设限，仅告警）')
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
        # 复位目标与起跑状态必须逐位相同（make_full_traj.py 由同一行生成）：
        # 否则客户端复位到的地方与 arming 参照不是同一个点，判据会莫名其妙地紧。
        if float(np.abs(self.home - self.start_state).max()) != 0.0:
            raise SystemExit('FAIL meta.home_target 与 state[0] 不相等（L∞=%.6g）'
                             % float(np.abs(self.home - self.start_state).max()))

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

            err = state - self.start_state
            dist = float(np.abs(err).max())                 # 全 14 维
            dist_draw = float(np.abs(err[DRAWER_ARM]).max())
            dist_cube = float(np.abs(err[CUBE_ARM]).max())
            matched = dist < self.args.arm_tol
            armed_now = False
            if not self.armed and matched:
                self.armed = True
                armed_now = True
                self.armed_at_call = n
                self.armed_gap_s = gap
                self.cursor = 0
                print('REPLAY_ARMED call=%d gap_s=%s dist_all=%.4f (抽屉 %.4f / 方块 %.4f) '
                      '(游标置 0，回放从这里起)'
                      % (n, 'n/a' if gap is None else '%.2f' % gap, dist, dist_draw, dist_cube),
                      flush=True)

            if (not self.armed and self.args.arm_deadline_calls > 0
                    and n >= self.args.arm_deadline_calls):
                msg = ('replay never armed after %d calls: 两条臂都没到过起跑位姿 '
                       '(need 全 14 维 L∞ <%.3f; now dist_all=%.4f 抽屉 %.4f 方块 %.4f)。'
                       'trial 会在这里中止，手臂保持不动。'
                       % (n, self.args.arm_tol, dist, dist_draw, dist_cube))
                print('REPLAY_NOT_ARMED ' + msg, flush=True)
                self._log({'call': n, 'gap_s': gap, 'dist_all': dist, 'dist_drawer': dist_draw,
                           'dist_cube': dist_cube, 'matched': matched, 'armed': False,
                           'error': msg})
                return {'error': msg}

            k = self.cursor
            idx = np.minimum(k + np.arange(ROWS), self.H - 1)
            rows = self.action[idx]                          # 全 14 维都来自轨迹
            if self.armed:
                self.cursor = min(k + ROWS, self.H)
            if self.cursor >= self.H and not self.done_logged:
                self.done_logged = True
                print('REPLAY_DONE 轨迹放完（%d 拍 = %.2f s），之后每拍保持末行'
                      % (self.H, self.H / FPS), flush=True)
            window = [int(idx[0]), int(idx[-1])]
            self._log({'call': n, 'gap_s': gap, 'dist_all': dist, 'dist_drawer': dist_draw,
                       'dist_cube': dist_cube, 'matched': matched, 'armed': self.armed,
                       'cursor_would_be': k, 'cursor_next': self.cursor, 'window': window,
                       'observed_state': state.tolist()})
            if n <= 4 or armed_now or (self.cursor >= self.H):
                print('REPLAY_REQ call=%d gap_s=%s dist_all=%.4f (抽屉 %.4f / 方块 %.4f) '
                      'armed=%s window=[%d..%d] cursor->%d'
                      % (n, 'n/a' if gap is None else '%.2f' % gap, dist, dist_draw, dist_cube,
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
                'segment_ticks': self.meta.get('segment_ticks'),
                'arm_tol': self.args.arm_tol,
                'stretch_factor_total': self.meta.get('stretch_factor_total'),
                'pipeline_grip_tick': self.meta.get('pipeline_grip_tick'),
                'phases': self.meta.get('phases'),
                'return_home': self.meta.get('return_home'),
                'horizon_without_return': self.meta.get('horizon_without_return')}


def make_handler(replay, args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = 'replay-server-pi0/1'

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
                           'horizon': replay.H, 'source': replay.meta.get('source_file'),
                           'segment_ticks': replay.meta.get('segment_ticks')},
            })

        def log_message(self, fmt, *a):   # 默认每请求两行太吵，收进自己的日志
            pass

    return Handler


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.traj):
        raise SystemExit('FAIL 缺轨迹 %s（先跑 make_full_traj.py）' % args.traj)
    replay = Replay(args)
    print('PI0_REMOTE_READY replay %s on http://%s:%d/' % (replay.replay_id, args.host, args.port),
          flush=True)
    print('PI0_REMOTE_CHECKPOINT %s' % replay.replay_id, flush=True)
    print('PI0_REMOTE_SHA256 %s' % replay.sha256, flush=True)
    print('PI0_REMOTE_ENSEMBLE off', flush=True)
    print('PI0_REMOTE_WARMUP 0.0', flush=True)
    print('REPLAY_TRAJ ep=%s 段=%s H=%d (%.2f s, 原 %s 拍, 拉伸 %.2f 倍, 前置 %s 拍) '
          'arm_tol=%.3f（全 14 维）咬住拍=%s 归位段=%s'
          % (replay.meta.get('source_file'), replay.meta.get('segment_ticks'), replay.H,
             replay.H / FPS, replay.meta.get('horizon_raw'),
             float(replay.meta.get('stretch_factor_total', 0)),
             replay.meta.get('lead_hold_rows'), args.arm_tol,
             replay.meta.get('pipeline_grip_tick'),
             (replay.meta.get('return_home') or {}).get('walk')), flush=True)
    print('!!! 本服务不跑模型：每一行动作逐字来自 %s（仅经时间重参数化：所有轴共享同一参照时刻），'
          '末尾另接一段余弦缓动的自动归位（语料里没有这一段）。两条臂都由轨迹供行。'
          % replay.meta.get('source_file'), flush=True)
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
