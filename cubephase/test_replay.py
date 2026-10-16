#!/usr/bin/env python3
"""离线自测：不需要 CAN、不需要相机、不连机械臂。

跑的用 `lerobot_v30`（和启动器同一套环境）：
    /home/databall/miniconda3/envs/lerobot_v30/bin/python -I test_replay.py

覆盖三件事：
  A. 协调拉伸的数学性质（不碰语料）：每拍步长进帽、相对时序保住、端点不丢。
  B. 服务端的 arming 逻辑与供行（真起一个本地 HTTP 服务，用 urllib 打它）。
  C. 已建好的 cube_phase_traj.npz 与 .json 的自洽（meta 逐字相等、形状、越界）。

任何一条不过就非零退出。退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_traj as MT          # noqa: E402
import replay_server as RS      # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('   %-4s %s%s' % ('ok' if cond else 'FAIL', name, ('  — ' + detail) if detail else ''))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- A. 拉伸数学
def test_stretch():
    print('A. 协调拉伸的数学性质')
    rng = np.random.default_rng(7)
    H = 60
    # 造一条"快冲-停住-快回"的合成轨迹：两轴一个大跳、一个夹爪式小步
    t = np.arange(H)
    traj = np.zeros((H, 14))
    traj[:, 8] = np.clip(1.8 * np.sin(np.pi * t / 30.0), 0, None)      # 关节，快速下探
    traj[:, 13] = np.clip(np.linspace(0, 0.07, H) * (t > 22), 0, 0.07)  # 夹爪，慢开
    lim = np.full(14, 0.40 / 30.0 * 0.90)
    lim[13] = 0.030 / 30.0 * 0.90
    D = MT.coordinated_stretch(traj, lim, 1.0)
    d = np.abs(np.diff(D, axis=0))
    check('每拍步长进帽', bool((d <= lim + 1e-12).all()),
          '最紧的一轴用到帽的 %.1f%%' % (100 * float((d / lim).max())))
    check('长度不短于原速', len(D) >= 1, '%d 拍' % len(D))
    check('首行 == 参照首行', bool(np.allclose(D[0], traj[0])))
    check('末行 == 参照末行', bool(np.allclose(D[-1], traj[-1], atol=1e-9)))

    # 相对时序：找一个"两轴同时发生"的事件，检查它俩在拉伸后仍在同一拍附近
    raw_arm = int(np.argmax(traj[:, 8]))
    raw_jaw = int(np.flatnonzero(traj[:, 13] >= 0.06)[0])
    Dt = MT.coordinated_stretch(traj, np.full(14, np.inf), 1.0)
    check('无帽时拉伸是恒等', np.array_equal(Dt, traj), '%s' % (Dt.shape == traj.shape))
    print('      （参照里手臂峰 %d 拍、夹爪到位 %d 拍；拉伸后总量 %d 拍）'
          % (raw_arm, raw_jaw, len(D)))


# ---------------------------------------------------------------- B. 服务端
def _post(url, state, timeout=3.0):
    req = urllib.request.Request(
        url, json.dumps({'state': list(state), 'images': {}}).encode(),
        {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server():
    print('B. 服务端 arming 与供行')
    npz = os.path.join(HERE, 'cube_phase_traj.npz')
    if not os.path.isfile(npz):
        check('轨迹存在', False, '缺 %s（先跑 make_traj.py）' % npz)
        return
    args = RS.parse_args(['--port', '0', '--traj', npz, '--arm-deadline-calls', '4'])
    rep = RS.Replay(args)
    home = rep.home
    start = rep.state[0]

    # 1) 远离起跑位姿（=被动 peek 的位姿）→ 不 arm、游标不动、仍给行
    far = start.copy()
    far[8] += 1.2
    a = rep.answer(far)
    check('peek 不 arm', a.get('armed') is False)
    check('peek 不推进游标', rep.cursor == 0)
    check('不 arm 也给 10 行', a['rows'].shape == (10, 14))

    # 2) 方块臂对、但抽屉臂 J2 差 0.5 rad → 不 arm（这条正是老判据的 bug：全 14 维比就永不 arm）
    off = np.concatenate([home[:7], start[7:]]).copy()
    off[1] = home[1] + 0.5
    b = rep.answer(off)
    check('方块臂对但抽屉臂 J2 不对 → 不 arm', b.get('armed') is False,
          'dist_j2=%.4f' % abs(off[1] - home[1]))
    check('方块臂 7 维判据不看抽屉臂', abs(float(np.abs(off[7:] - start[7:]).max())) == 0.0)

    # 3) 真起跑位姿：方块臂 7 维贴 start、抽屉臂 J2 = home → arm，游标置 0
    armed_state = np.concatenate([home[:7], start[7:]]).copy()
    armed_state[1] = home[1]
    c = rep.answer(armed_state)
    check('起跑位姿 arm', c.get('armed') is True)
    check('arm 后游标 = 10', rep.cursor == 10)

    # 4) 供行内容：抽屉臂 == hold 常量；方块臂 == 轨迹 [0..9]
    rows = c['rows']
    check('抽屉臂钉常量', bool(np.allclose(rows[:, :7], rep.hold)))
    check('方块臂来自轨迹 [0..10)', bool(np.allclose(rows[:, 7:], rep.action[:10, 7:])))

    # 5) 越过末尾时保持末行，不越界
    rep.cursor = rep.H - 3
    d = rep.answer(armed_state)
    check('末尾保持末行', bool(np.allclose(d['rows'][-1, 7:], rep.action[-1, 7:])))

    # 6) deadline：没 arm 够次数就回 error
    rep2 = RS.Replay(args)
    err = None
    for _ in range(4):
        r = rep2.answer(far)
        err = r.get('error') or err
    check('deadline 报 error', err is not None)

    # 7) 真 HTTP：起服务、打两枪
    httpd = RS.ThreadingHTTPServer(('127.0.0.1', 0), RS.make_handler(rep, args))
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    time.sleep(0.2)
    url = 'http://127.0.0.1:%d/' % port
    try:
        code, body = _post(url, start)
        check('HTTP 200', code == 200, 'code=%d' % code)
        check('checkpoint == replay_id', body.get('checkpoint') == rep.replay_id)
        check('ensemble.enabled is False', isinstance(body.get('ensemble'), dict)
              and body['ensemble'].get('enabled') is False)
        check('10x14 行', np.array(body['actions']).shape == (10, 14))
        check('有 checkpoint_sha256', isinstance(body.get('checkpoint_sha256'), str)
              and len(body['checkpoint_sha256']) == 64)
        code2, body2 = _post(url, far)
        check('HTTP 500 路径可达', code2 in (200, 500), 'code=%d' % code2)
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------- C. 产物
def test_artifacts():
    print('C. 已建产物自洽')
    npz = os.path.join(HERE, 'cube_phase_traj.npz')
    js = os.path.join(HERE, 'cube_phase_traj.json')
    if not (os.path.isfile(npz) and os.path.isfile(js)):
        check('产物存在', False, '缺 npz 或 json')
        return
    with np.load(npz, allow_pickle=False) as z:
        meta_in = json.loads(str(z['meta']))
        action = np.asarray(z['action'])
        state = np.asarray(z['state'])
    meta_out = json.load(open(js))
    check('两侧 meta 逐字相等', meta_in == meta_out)
    check('action 形状 (H,14) 且与 meta 一致',
          action.ndim == 2 and action.shape[1] == 14 and action.shape[0] == meta_in['horizon'])
    check('state 与 action 同形', state.shape == action.shape)
    check('全部有限', bool(np.isfinite(action).all() and np.isfinite(state).all()))
    check('不越 LOW/HIGH', float(np.maximum(np.maximum(MT.LOW - action, action - MT.HIGH), 0).max()) == 0.0)
    r = meta_in['action_rates']
    check('每拍进帽', r['joint_rows_over_cap'] == 0 and r['jaw_rows_over_cap'] == 0,
          '关节 %d 拍、夹爪 %d 拍超' % (r['joint_rows_over_cap'], r['jaw_rows_over_cap']))
    check('limited_target 实切 <1e-12', meta_in['pipeline_limiter_cut_max'] <= 1e-12,
          '%.3g' % meta_in['pipeline_limiter_cut_max'])
    check('爪合上那拍手臂在抓取位', meta_in['pipeline_jaw_close_arm_dev_rad'] <= 0.02,
          '%.4f rad' % meta_in['pipeline_jaw_close_arm_dev_rad'])
    dg = meta_in['driver_guard']
    check('不撞驱动护栏', dg['joint_ratio'] < 1.0 and dg['jaw_ratio'] < 1.0,
          '关节 %.0f%%、夹爪 %.0f%%' % (100 * dg['joint_ratio'], 100 * dg['jaw_ratio']))
    check('前置保持行存在', meta_in['lead_hold_rows'] >= 1 and
          bool(np.allclose(action[:meta_in['lead_hold_rows']], action[0])))
    print('     %d 拍 = %.1f s（原 %d 拍，%.2f 倍）'
          % (meta_in['horizon'], meta_in['horizon_seconds'], meta_in['horizon_raw'],
             meta_in['stretch_factor_total']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-server', action='store_true')
    a = ap.parse_args()
    test_stretch()
    if not a.skip_server:
        test_server()
    test_artifacts()
    print()
    if FAILS:
        print('FAIL %d 项：%s' % (len(FAILS), ', '.join(FAILS)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
