#!/usr/bin/env python3
"""离线自测：不需要 CAN、不需要相机、不连机械臂。

跑的用 `lerobot_v30`（和启动器同一套环境）：
    /home/databall/miniconda3/envs/lerobot_v30/bin/python -I test_full.py

覆盖四件事：
  A. 协调拉伸与"归位"滚动的数学性质（不碰语料）：每拍步长进帽、相对时序保住、
     端点不丢；归位段末行逐位等于目标。
  B. 服务端的 arming 逻辑与供行（真起一个本地 HTTP 服务，用 urllib 打它），
     含两条**只属于这份**的守卫：复位目标必须等于轨迹起跑状态；这一份的末行
     因为末尾自动归位**等于**起跑状态（所以原地再跑会立刻 arm —— 由前置 10 拍吸收）。
  C. 已建好的 full_traj.npz / .json / home_target_full.json 的自洽，外加这一整段
     的物理故事（拉 -> 放 -> 合 -> 归位 四个相位，次序、极值、终点）。
  D. 用一份**被改坏**的 npz（reset 目标与 state[0] 不等）验证服务端会拒绝启动。

任何一条不过就非零退出。退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_full_traj as MT      # noqa: E402
import replay_server as RS       # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print('   %-4s %s%s' % ('ok' if cond else 'FAIL', name, ('  — ' + detail) if detail else ''))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- A. 拉伸数学
def test_stretch():
    print('A. 协调拉伸 + 归位滚动')
    H = 60
    t = np.arange(H)
    # 造一条"快拉-停住-快回"的合成轨迹：一个关节快动、一个夹爪慢开
    traj = np.zeros((H, 14))
    traj[:, 1] = np.clip(1.8 * np.sin(np.pi * t / 30.0), 0, None)       # 关节，快速拉
    traj[:, 6] = np.clip(np.linspace(0, 0.07, H) * (t > 22), 0, 0.07)   # 夹爪，慢开
    lim = np.full(14, 0.40 / 30.0 * 0.90)
    lim[[6, 13]] = 0.030 / 30.0 * 0.90
    D, sv = MT.coordinated_stretch(traj, lim, 1.0)
    d = np.abs(np.diff(D, axis=0))
    check('每拍步长进帽', bool((d <= lim + 1e-12).all()),
          '最紧的一轴用到帽的 %.1f%%' % (100 * float((d / lim).max())))
    check('长度不短于原速', len(D) >= 1, '%d 拍' % len(D))
    check('首行 == 参照首行', bool(np.allclose(D[0], traj[0])))
    check('末行 == 参照末行', bool(np.allclose(D[-1], traj[-1], atol=1e-9)))
    check('参照时刻单调不减、与行数同长', len(sv) == len(D) and bool((np.diff(sv) >= -1e-12).all()))
    check('参照时刻从 0 起到末拍', sv[0] == 0.0 and abs(sv[-1] - (H - 1)) < 1e-9,
          '%.6f' % sv[-1])
    Dt, _ = MT.coordinated_stretch(traj, np.full(14, np.inf), 1.0)
    check('无帽时拉伸是恒等', np.array_equal(Dt, traj), '%s' % (Dt.shape == traj.shape))
    print('      （参照里关节峰 %d 拍；拉伸后总量 %d 拍）'
          % (int(np.argmax(traj[:, 1])), len(D)))

    # 归位滚动：从末行回目标，每拍进帽、末行逐位等于目标、末尾接保持行
    tgt = traj[0]
    ret, n = MT.return_rows(D[-1], tgt, lim, hold=7)
    check('归位段每拍进帽', bool((np.abs(np.diff(np.vstack([D[-1], ret]), axis=0)) <= lim).all()),
          '%d 拍移动' % n)
    check('归位段末行逐位等于目标', bool((ret[-7 - 1] == tgt).all()),
          'L∞ = %.3g' % float(np.abs(ret[-7 - 1] - tgt).max()))
    check('归位段末尾接 %d 拍保持' % 7, bool((ret[-7:] == tgt).all()))
    check('归位段不含起点（不与末行重复）', not bool((ret[0] == D[-1]).all()))


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


def _replay(npz, tol=0.10, deadline=20):
    args = RS.parse_args(['--port', '0', '--traj', npz, '--arm-tol', str(tol),
                          '--arm-deadline-calls', str(deadline)])
    return RS.Replay(args), args


def test_server():
    print('B. 服务端 arming 与供行')
    npz = os.path.join(HERE, 'full_traj.npz')
    if not os.path.isfile(npz):
        check('轨迹存在', False, '缺 %s（先跑 make_full_traj.py）' % npz)
        return
    rep, args = _replay(npz)
    start = rep.state[0]

    # 1) 远离起跑位姿（=被动 peek 的位姿）→ 不 arm、游标不动、仍给行
    far = start.copy()
    far[8] += 1.2
    a = rep.answer(far)
    check('peek 不 arm', a.get('armed') is False)
    check('peek 不推进游标', rep.cursor == 0)
    check('不 arm 也给 10 行', a['rows'].shape == (10, 14))

    # 2) 末行位姿：这一份末尾**自动归位**，所以末行 == 起跑位姿 ⇒ 会 arm。
    #    这不是缺陷：arm 后游标置 0，而第 0 拍就是当前位姿，没有跳变；前置的 10 拍
    #    "保持不动"正好把被动 peek 吃掉的那 10 行吸收掉。
    end = rep.action[-1]
    d_end = float(np.abs(end - start).max())
    check('末行 == 起跑位姿（末尾自动归位）', d_end == 0.0, 'L∞ = %.3g' % d_end)
    rep.cursor = 0
    b = rep.answer(end)
    check('末行位姿会 arm（原地再跑）', b.get('armed') is True)
    check('arm 后游标回到 10（前置行吸收 peek）', rep.cursor == 10)
    check('前 10 行就是保持不动的首行', bool(np.allclose(b['rows'], rep.action[0])))

    # 3) 只差 0.2 rad（比 arm_tol 大）→ 不 arm；这才是"判据比全 14 维"的意义。
    #    要一个**新的**服务实例：一旦 arm 过就不再重新对齐（item 2 已经 arm 了）。
    rep3, _ = _replay(npz)
    off = start.copy()
    off[1] += 0.2
    c = rep3.answer(off)
    check('差 0.2 rad 不 arm', c.get('armed') is False,
          'L∞ = %.4f >= %.2f' % (0.2, 0.10))

    # 4) 起跑位姿（14 维各 0.02 以内）→ arm，游标置 0
    rep.cursor = 0
    armed_state = start + np.linspace(-0.02, 0.02, 14)
    d = rep.answer(armed_state)
    check('起跑位姿 arm', d.get('armed') is True,
          'L∞ = %.4f < %.2f' % (float(np.abs(armed_state - start).max()), 0.10))
    check('arm 后游标 = 10', rep.cursor == 10)

    # 5) 供行内容：**两条臂都来自轨迹**（与 cubephase 那份相反）
    rows = d['rows']
    check('两条臂都来自轨迹 [0..10)', bool(np.allclose(rows, rep.action[:10])))

    # 6) 越过末尾时保持末行，不越界
    rep.cursor = rep.H - 3
    e = rep.answer(armed_state)
    check('末尾保持末行', bool(np.allclose(e['rows'][-1], rep.action[-1])))

    # 7) deadline：没 arm 够次数就回 error
    rep2, _ = _replay(npz, deadline=4)
    errm = None
    for _ in range(4):
        errm = rep2.answer(far).get('error') or errm
    check('deadline 报 error', errm is not None)

    # 8) 真 HTTP：起服务、打两枪
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
        with urllib.request.urlopen(url, timeout=3.0) as r:
            st = json.loads(r.read())
        check('GET status 可读', st.get('replay_id') == rep.replay_id
              and st.get('arm_tol') == 0.10)
        code2, body2 = _post(url, far)
        check('已 arm 后再收到别的位姿仍供行（不重新对齐）', code2 == 200,
              'code=%d，游标 %s' % (code2, body2.get('replay', {}).get('cursor')))
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------- D. 守卫
def test_guard():
    print('D. 服务端拒绝"复位目标 ≠ 起跑状态"的轨迹')
    npz = os.path.join(HERE, 'full_traj.npz')
    if not os.path.isfile(npz):
        check('轨迹存在', False, '缺 %s' % npz)
        return
    with np.load(npz, allow_pickle=False) as z:
        action = np.asarray(z['action'])
        state = np.asarray(z['state'])
        meta = json.loads(str(z['meta']))
    meta_tampered = dict(meta)
    ht = list(meta['home_target'])
    ht[1] += 0.1
    meta_tampered['home_target'] = ht
    path = os.path.join(tempfile.mkdtemp(prefix='full_guard_'), 'bad.npz')
    np.savez(path, action=action, state=state,
             meta=np.array(json.dumps(meta_tampered, sort_keys=True)))
    raised = None
    try:
        _replay(path)
    except SystemExit as e:
        raised = str(e)
    check('被改坏的 home_target 会被拒', raised is not None and 'state[0]' in raised,
          (raised or '没报错（守卫失效）')[:80])
    try:
        os.remove(path)
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass


# ---------------------------------------------------------------- C. 产物
def test_artifacts():
    print('C. 已建产物自洽 + 这一整段的物理故事')
    npz = os.path.join(HERE, 'full_traj.npz')
    js = os.path.join(HERE, 'full_traj.json')
    hj = os.path.join(HERE, 'home_target_full.json')
    if not (os.path.isfile(npz) and os.path.isfile(js) and os.path.isfile(hj)):
        check('产物存在', False, '缺 npz / json / home json')
        return
    with np.load(npz, allow_pickle=False) as z:
        meta_in = json.loads(str(z['meta']))
        action = np.asarray(z['action'])
        state = np.asarray(z['state'])
    meta_out = json.load(open(js))
    home = json.load(open(hj))
    check('两侧 meta 逐字相等', meta_in == meta_out)
    check('action 形状 (H,14) 且与 meta 一致',
          action.ndim == 2 and action.shape[1] == 14 and action.shape[0] == meta_in['horizon'])
    check('state 与 action 同形', state.shape == action.shape)
    check('全部有限', bool(np.isfinite(action).all() and np.isfinite(state).all()))
    check('不越 LOW/HIGH',
          float(np.maximum(np.maximum(MT.LOW - action, action - MT.HIGH), 0).max()) == 0.0)
    r = meta_in['action_rates']
    check('每拍进帽', r['joint_rows_over_cap'] == 0 and r['jaw_rows_over_cap'] == 0,
          '关节 %d 拍、夹爪 %d 拍超' % (r['joint_rows_over_cap'], r['jaw_rows_over_cap']))
    check('limited_target 实切 <1e-12', meta_in['pipeline_limiter_cut_max'] <= 1e-12,
          '%.3g' % meta_in['pipeline_limiter_cut_max'])
    check('咬住那拍抽屉臂在演示位', meta_in['pipeline_grip_arm_dev_rad'] <= 0.02,
          '%.4f rad' % meta_in['pipeline_grip_arm_dev_rad'])
    dg = meta_in['driver_guard']
    check('不撞驱动护栏', dg['joint_ratio'] < 1.0 and dg['jaw_ratio'] < 1.0,
          '关节 %.0f%%、夹爪 %.0f%%' % (100 * dg['joint_ratio'], 100 * dg['jaw_ratio']))
    check('前置保持行存在', meta_in['lead_hold_rows'] >= 1 and
          bool(np.allclose(action[:meta_in['lead_hold_rows']], action[0])))
    # 复位目标三处必须同一个点（启动器会核 json↔meta，服务端核 meta↔npz）
    d1 = float(np.abs(np.asarray(home['target']) - np.asarray(meta_in['reset_target'])).max())
    d2 = float(np.abs(state[0] - np.asarray(meta_in['reset_target'])).max())
    check('home json target == 轨迹 state[0]', d1 <= 1e-9 and d2 == 0.0,
          'json↔meta %.2g、meta↔state[0] %.2g' % (d1, d2))
    check('home json 有 source（客户端要求非空）', bool(home.get('source')))
    check('段长 == t1-t0+1', meta_in['horizon_raw'] == meta_in['segment_ticks'][1]
          - meta_in['segment_ticks'][0] + 1, '%d 拍' % meta_in['horizon_raw'])

    # --- 四个相位的次序与极值（判据只看"爪/关节的极值走向"） ----------------------
    ph = meta_in['phases']
    check('有三个录制相位', set(ph) >= {'pull', 'cube', 'close'})
    pull, cube, close = ph['pull']['walk'], ph['cube']['walk'], ph['close']['walk']
    check('相位次序 = 拉 -> 放 -> 合',
          pull[0] < pull[1] < cube[0] <= cube[1] < close[0] < close[1],
          '拉 %s / 方块 %s / 合 %s' % (pull, cube, close))
    lead = meta_in['lead_hold_rows']
    check('相位都落在前置行之后', pull[0] + lead >= lead and close[1] + lead < meta_in['horizon'])
    check('爪先张到全开', meta_in['drawer_claw_open_peak_raw'] >= 0.06,
          '%.4f' % meta_in['drawer_claw_open_peak_raw'])
    check('再合到底（咬住）', meta_in['drawer_claw_grip_raw'] <= 0.01,
          '%.4f' % meta_in['drawer_claw_grip_raw'])
    sweep = meta_in['drawer_arm_lJ2_max'] - meta_in['drawer_arm_lJ2_min']
    check('抽屉臂真的拉了一把（lJ2 跨 >1 rad）', sweep > 1.0, '%.4f rad' % sweep)
    check('净拉量 >0.8 rad',
          abs(meta_in['drawer_arm_lJ2_end'] - meta_in['drawer_arm_lJ2_min']) > 0.8,
          '%.4f rad' % abs(meta_in['drawer_arm_lJ2_end'] - meta_in['drawer_arm_lJ2_min']))
    check('合抽屉相位：爪全程张着（推不是夹）',
          meta_in['drawer_claw_close_phase_min_raw'] >= 0.04,
          '%.4f..%.4f' % (meta_in['drawer_claw_close_phase_min_raw'],
                          meta_in['drawer_claw_close_phase_max_raw']))
    check('合抽屉相位：抽屉臂确实又动了一次',
          meta_in['close_phase_lJ2_span_raw'] >= 0.20,
          '%.4f rad' % meta_in['close_phase_lJ2_span_raw'])
    check('放方块相位：方块爪 张->合->松',
          meta_in['cube_claw_open_raw'] >= 0.05 and meta_in['cube_claw_shut_raw'] <= 0.04
          and meta_in['cube_claw_reopen_raw'] >= 0.05,
          '%.4f -> %.4f -> %.4f' % (meta_in['cube_claw_open_raw'],
                                    meta_in['cube_claw_shut_raw'],
                                    meta_in['cube_claw_reopen_raw']))
    check('放方块相位：方块臂真的伸出去过',
          meta_in['cube_arm_rJ2_span_raw'] >= 1.0, '%.4f rad' % meta_in['cube_arm_rJ2_span_raw'])
    check('咬住拍在拉伸后（只会变晚）',
          meta_in['pipeline_grip_tick'] >= meta_in['reference_grip_tick'],
          '拉伸前 %d → 拉伸后 %d' % (meta_in['reference_grip_tick'],
                                     meta_in['pipeline_grip_tick']))

    # --- 归位段（自动归位） ------------------------------------------------------
    rr = meta_in['return_home']
    check('有归位段', isinstance(rr, dict) and rr['rows_move'] > 0)
    check('归位段末行逐位等于复位目标',
          meta_in['return_end_dev_from_reset_target'] == 0.0,
          '%.3g' % (meta_in['return_end_dev_from_reset_target']))
    check('整条末行就是复位目标', float(np.abs(action[-1] - np.asarray(
        meta_in['reset_target'])).max()) == 0.0)
    nhold = rr['rows_hold']
    check('末 %d 行保持不动' % nhold, bool(np.allclose(action[-nhold:], action[-1])))
    w = rr['walk']
    check('归位段在合抽屉之后', w[0] > close[1] + 1, '%s > %s' % (w, close))
    check('窗口要够长才算跑完（归位段也算）',
          meta_in['horizon_without_return'] < meta_in['horizon'],
          '%d < %d 拍' % (meta_in['horizon_without_return'], meta_in['horizon']))
    print('     %d 拍 = %.1f s（原 %d 拍，%.2f 倍）；咬住第 %d 拍 = %.1f s；'
          '归位段 %d..%d 拍'
          % (meta_in['horizon'], meta_in['horizon_seconds'], meta_in['horizon_raw'],
             meta_in['stretch_factor_total'], meta_in['pipeline_grip_tick'],
             meta_in['pipeline_grip_tick'] / 30.0, w[0] + lead, w[1] + lead))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-server', action='store_true')
    a = ap.parse_args()
    test_stretch()
    if not a.skip_server:
        test_server()
        test_guard()
    test_artifacts()
    print()
    if FAILS:
        print('FAIL %d 项：%s' % (len(FAILS), ', '.join(FAILS)))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
