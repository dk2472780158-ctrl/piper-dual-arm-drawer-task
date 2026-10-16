#!/usr/bin/env python3
"""全任务回放轨迹构建：抽抽屉 -> 放方块 -> 合上抽屉 -> 归位。

和前面两份（cubephase / 拉抽屉单测）的关系
--------------------------------------------
用户要的完整版是"两条臂都动、把演示从头到尾做一遍"。把两份合起来看，演示
`today99 / episode_000041` 的第 **48..750 拍**恰好就是这件事：

    相位 A  第  48–300 拍   抽屉臂：伸手 -> 张爪 -> 合爪咬住拉环 -> 拉开约 11 cm -> 松爪
    相位 B  第 333–600 拍   方块臂：伸过去 -> 夹起蓝方块 -> 移到抽屉口 -> 松爪放下
    相位 C  第 640–717 拍   抽屉臂：再次前伸/退回，**爪全程张开** = 把抽屉合上
    相位 D  第 718–750 拍   静止（演示到此为止）

这些拍号不是猜的：本脚本从语料里**检测**出各条臂的活动窗口，再断言
"A 先于 B、B 先于 C"（见 main() 的相位检测与自检 3/4/5）。

末尾**另加一段**：演示里两条臂都没有回到起始位姿（第 738 拍以后逐字静止），
所以"归位"这一段语料里**没有**。本脚本按用户要求补一段**关节空间平滑回位**
（余弦缓动，每拍步长同样压在帽内），终点 = 复位目标 = 段首录制 state，
于是跑完就回到起点、可以原地再跑一次。

为什么不能"原样放"（与前两份同一个病）
--------------------------------------
执行链每拍压步长：客户端 `limited_target`（关节 0.40 rad/s、夹爪 0.030 m/s）；
驱动护栏 `verified_hardware.py:474/476` 是 `.25*dt+.01` rad / `.006*dt+.001` m。
演示的单拍峰值远超两者（拉抽屉段的 dim4 超 4.0 倍、抽屉爪超 15.6 倍；方块相位
更狠，rJ2 超 5.6 倍、rjaw 超 12.4 倍），限速器会把"冲过去、到点停住"切成斜坡，
**手臂到位与合爪被拉成两条时间线**（实测错位 94 拍 = 3.13 s）。

做法与前面两份完全一致：**协调时间重参数化** ——
把整段重采样到一条共享的参照时刻 s，14 个轴永远处在同一个 s，每拍只把 s 推进到
"最慢的那根轴允许的最大值"。相对时序逐字保住；每拍步长进帽 ⇒ 客户端限速器
变成**空操作**。不碰任何安全阈值、不改客户端一个字节。

首行前置 --lead-hold 拍"保持不动"
--------------------------------
服务端游标一旦 arm 就按 10 行/次推进。如果被动 peek（客户端真动之前打的那一次）
恰好也在起跑位姿上，第一个 10 行块会被 peek 吃掉。前置 10 拍静止把这种"跳掉 10 行"
退化成无害的 0.33 s。

复位目标
--------
同时写出 `home_target_full.json`，`target` = 段首的**录制 state**（= 手臂当时真正
在的地方）。客户端 `hardware.home(target)` 把两条臂开到那里，`settle_home` 只认
"关节 0.035 rad、夹爪 0.004 m 以内且稳 0.5 s"。启动器断言 `json.target` 与轨迹
`state[0]` 逐位相等 —— 这个目标也就是回位段的终点，所以跑完手臂就坐在复位点上。

输出
----
  full_traj.npz     action (H,14) / state (H,14) / meta (0 维 JSON 字符串)
  full_traj.json    同一份 meta，给 shell 启动器用普通 json 读
  home_target_full.json  复位目标（= 段首 state = 回位段终点）
不复现就重跑本脚本；两侧 meta 必须逐字相等（离线测试会核）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

NAMES = ['lJ1', 'lJ2', 'lJ3', 'lJ4', 'lJ5', 'lJ6', 'ljaw',
         'rJ1', 'rJ2', 'rJ3', 'rJ4', 'rJ5', 'rJ6', 'rjaw']
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.array([6, 13])
DRAWER = np.array([0, 1, 2, 3, 4, 5, 6])     # can1 = 左臂 = 抽屉臂
CUBE = np.array([7, 8, 9, 10, 11, 12, 13])   # can0 = 右臂 = 方块臂
LOW = np.tile([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439, 0], 2)
HIGH = np.tile([2.6179, 3.14, 0, 1.745, 1.22, 2.09439, .07], 2)
DRAWER_CLAW = 6
CUBE_CLAW = 13
FPS = 30.0
# 与 pi0_rollout_v8.py / pi0_real_trial_v8.py 的默认值同源。启动器会 export 这两个数，
# 并断言与 meta 里记的一致 —— 轨迹是照着它们建的，对不上就白建。
JOINT_RATE = 0.40        # rad/s，limited_target 的关节限速
GRIPPER_RATE = 0.030     # m/s，PI0_GRIPPER_RATE 默认
ACTION_FILTER_ALPHA = 0.25   # PI0_ACTION_ALPHA 默认；只滤关节、夹爪直通

# 固定段落：episode_000041 的第 48..750 拍（A+B+C+D）
SEG_T0 = 48
SEG_T1 = 750
EPISODE = 'episode_000041.parquet'
# 爪"咬住/夹住"的判据：张开峰之后再落到 <= 这个值（爪值＝开口宽度，小＝合）
GRIP_SHUT = 0.01
# 相位检测：某一维单拍动超过这个量就算"这条臂在动"
MOVE_THRESH = 0.002
MOVE_GAP = 45        # 两段活动之间静这么久就算两件事（相位内部的小停顿不算断开）
MOVE_MIN_PATH = 0.50  # 一段活动的"总路程"（7 维 |Δ| 之和）小于这个就不算一个相位

RETURN_HOLD = 30     # 回位段末尾保持多少拍（1.0 s）


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='/home/databall/drawer_today99_v1/dataset_train')
    p.add_argument('--episode', default=EPISODE)
    p.add_argument('--t0', type=int, default=SEG_T0, help='段起点（含）')
    p.add_argument('--t1', type=int, default=SEG_T1, help='段终点（含）')
    p.add_argument('--home-out', default=os.path.join(here, 'home_target_full.json'))
    p.add_argument('--out', default=os.path.join(here, 'full_traj'))
    p.add_argument('--stretch-margin', type=float, default=0.90,
                   help='每拍步长只用到帽的这个比例（<1 留抖动余量）')
    p.add_argument('--lead-hold', type=int, default=10,
                   help='首部前置多少拍"保持不动"')
    p.add_argument('--max-step', type=float, default=1.0,
                   help='每拍最多推进多少个参照拍（1.0 = 永不快于原速）')
    p.add_argument('--no-return', action='store_true',
                   help='不补"归位"那一段（轨迹就停在演示的收尾位姿上）')
    return p.parse_args(argv)


def load_segment(data, episode, t0, t1):
    """只读那一集、只切那一段。**不做"某条臂必须不动"的断言** —— 这一段两条臂都动。"""
    path = os.path.join(data, 'data', 'chunk-000', episode)
    if not os.path.isfile(path):
        raise SystemExit('FAIL 找不到 %s' % path)
    t = pq.read_table(path, columns=['observation.state', 'action'])
    s = np.asarray(t.column('observation.state').to_pylist(), dtype=float)
    a = np.asarray(t.column('action').to_pylist(), dtype=float)
    if s.shape[1] != 14 or a.shape[1] != 14:
        raise SystemExit('FAIL %s 维数不是 14：state %s action %s' % (path, s.shape, a.shape))
    if len(s) != len(a):
        raise SystemExit('FAIL %s state %d 行 action %d 行' % (episode, len(s), len(a)))
    n = len(s)
    if not (0 <= t0 < t1 < n):
        raise SystemExit('FAIL 段 [%d,%d] 不在 %d 行里' % (t0, t1, n))
    raw_action = np.asarray(a[t0:t1 + 1], dtype=np.float64)
    raw_state = np.asarray(s[t0:t1 + 1], dtype=np.float64)
    for tag, arr in (('action', raw_action), ('state', raw_state)):
        if not np.isfinite(arr).all():
            raise SystemExit('FAIL %s 里有非有限值' % tag)
    return raw_action, raw_state, path, n


def _at(traj, s):
    """参照时刻 s（实数）处的线性插值。"""
    last = len(traj) - 1
    if s >= last:
        return traj[last].copy()
    j = int(np.floor(s))
    f = s - j
    return traj[j] * (1.0 - f) + traj[j + 1] * f


def coordinated_stretch(traj, limits, max_step):
    """把 traj 重采样到一条共享时间刻度：所有轴永远在同一参照时刻，每拍把该时刻推进到
    "最慢那根轴允许的最大值"。相对时序因此逐字保住。

    返回 (out, svals)：out 是重采样后的行，svals[k] 是第 k 行对应的**段内参照拍号**
    （实数，0-based）。相位边界靠它从原拍号映射到输出拍号。"""
    out = [traj[0].copy()]
    svals = [0.0]
    s = 0.0
    last = len(traj) - 1
    guard = 0
    while s < last - 1e-9:
        prev = out[-1]
        hi = min(float(max_step), last - s)

        def ok(ds):
            return bool(np.all(np.abs(_at(traj, s + ds) - prev) <= limits))

        if ok(hi):
            ds = hi
        else:
            lo = 0.0
            for _ in range(48):
                mid = (lo + hi) / 2.0
                if ok(mid):
                    lo = mid
                else:
                    hi = mid
            ds = lo
        if ds < 1e-9:
            raise SystemExit('FAIL 拉伸在原地卡住（s=%.4f）' % s)
        s += ds
        out.append(_at(traj, s))
        svals.append(s)
        guard += 1
        if guard > 200000:
            raise SystemExit('FAIL 拉伸循环不收敛')
    return np.array(out), np.array(svals)


def real_pipeline(D, alpha, joint_rate, gripper_rate):
    """把服务端将要发的 D 走一遍**真实的客户端管线**：

        _apply_action_filter(α, 关节序贯低通、夹爪直通)
        -> limited_target(commanded, row, dt, joint_rate, gripper_rate)

    返回 (滤波后 F, 限速后 C)。只要 |F-C| 的 max 为 0，就说明限速器一次都没动手。"""
    F = np.array(D, dtype=np.float64, copy=True)
    carried = None
    for row in F:
        if carried is None:
            carried = row.copy()
        else:
            joints_now = alpha * row[JOINTS] + (1.0 - alpha) * carried[JOINTS]
            carried = row.copy()          # 夹爪永远来自本行
            carried[JOINTS] = joints_now
        row[:] = carried
    dt = 1.0 / FPS
    lim = np.full(14, joint_rate * dt)
    lim[GRIPPERS] = gripper_rate * dt
    C = [F[0].copy()]
    for k in range(1, len(F)):
        prev = C[-1]
        C.append(np.clip(prev + np.clip(F[k] - prev, -lim, lim), LOW, HIGH))
    return F, np.array(C)


def rates(block, joint_rate, gripper_rate):
    """两个夹爪都算。"""
    dj = np.abs(np.diff(block, axis=0))[:, JOINTS]
    dg = np.abs(np.diff(block, axis=0))[:, GRIPPERS]
    jcap, gcap = joint_rate / FPS, gripper_rate / FPS
    return {'joint_step_max_rad': float(dj.max()),
            'joint_rows_over_cap': int((dj > jcap + 1e-12).any(axis=1).sum()),
            'jaw_step_max_m': float(dg.max()),
            'jaw_step_max_dim': int(GRIPPERS[int(np.unravel_index(dg.argmax(), dg.shape)[1])]),
            'jaw_rows_over_cap': int((dg > gcap + 1e-12).any(axis=1).sum()),
            'rows': int(len(dj)),
            'joint_cap_rad_per_tick': jcap, 'jaw_cap_m_per_tick': gcap}


def grip_tick(block, col, shut=GRIP_SHUT, lo=0, hi=None):
    """"咬住"那一拍：块内先把该爪张开到峰，峰之后第一次落到 <= shut。
    返回 (峰值拍, 咬住拍)；找不到返回 (峰值拍, None)。拍号是块内 0-based。"""
    sub = block[lo:hi]
    peak = int(np.argmax(sub[:, col]))
    after = np.flatnonzero(sub[peak:, col] <= shut)
    return lo + peak, (lo + peak + int(after[0]) if len(after) else None)


def move_windows(raw_action, cols, gap=MOVE_GAP, thresh=MOVE_THRESH, min_path=MOVE_MIN_PATH):
    """检测"这组维在动"的时间窗。返回 [(起拍, 止拍, 总路程)]（0-based、含端点）。
    两段之间静超过 gap 拍就断开；把间隔 <= gap 的合并；总路程太小的一闪而过（单个
    采样点的抖动）丢掉 —— 否则一个相位内部的小停顿、或方块臂被带着抖一下，都会被
    当成一个相位。"""
    hit = np.flatnonzero((np.abs(np.diff(raw_action[:, cols], axis=0)) > thresh).any(axis=1))
    if not len(hit):
        return []
    out = []
    a = b = int(hit[0])
    for t in hit[1:]:
        t = int(t)
        if t - b > gap:
            out.append((a, b))
            a = t
        b = t
    out.append((a, b))
    kept = []
    for a, b in out:
        path = float(np.abs(np.diff(raw_action[a:b + 1, cols], axis=0)).sum())
        if path >= min_path:
            kept.append((a, b, path))
    return kept


def return_rows(last, target, limits, hold=RETURN_HOLD):
    """从 last **平滑**（余弦缓动）回到 target，每拍步长严格进帽；末尾再接 hold 拍不动。
    返回 (rows, n_move)：rows 不含起点（起点就是 last）。"""
    last = np.asarray(last, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - last
    n = 1
    while True:
        s = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / float(n)))
        q = last[None, :] + s[:, None] * delta[None, :]
        if bool(np.all(np.abs(np.diff(q, axis=0)) <= limits)) or n >= 200000:
            break
        n = int(n * 1.2) + 1
    q[-1] = target                                  # 末行逐位等于目标，不留浮点尾巴
    q = q[1:]
    hold_rows = np.tile(target, (max(0, int(hold)), 1))
    return np.vstack([q, hold_rows]), n


def main():
    args = parse_args()
    raw_action, raw_state, path, n_rows = load_segment(
        args.data, args.episode, args.t0, args.t1)
    H_raw = len(raw_action)
    print('语料 %s' % args.data)
    print('   源 %s 第 %d..%d 拍（含）= %d 拍 = %.2f s（全集 %d 拍）'
          % (args.episode, args.t0, args.t1, H_raw, H_raw / FPS, n_rows))

    # --- 相位检测：只看语料，把"哪条臂什么时候动"量出来 -------------------------
    dw = move_windows(raw_action, DRAWER)
    cw = move_windows(raw_action, CUBE)
    print('   抽屉臂活动窗（段内拍号，括号里是总路程 rad）= %s'
          % [(a, b, round(p, 3)) for a, b, p in dw])
    print('   方块臂活动窗（段内拍号，括号里是总路程 rad）= %s'
          % [(a, b, round(p, 3)) for a, b, p in cw])
    if len(dw) < 2:
        raise SystemExit('FAIL 抽屉臂只检测到 %d 段活动 —— 这份轨迹要的是"拉抽屉 + 合抽屉"'
                         '两段；端点选错了' % len(dw))
    if not cw:
        raise SystemExit('FAIL 方块臂全程没动 —— 段里应当有"放方块"那一段')
    pull_raw = (dw[0][0], dw[0][1])
    close_raw = (dw[-1][0], dw[-1][1])
    main_cube = max(cw, key=lambda w: w[2])          # 取总路程最长的那一段
    cube_raw = (main_cube[0], main_cube[1])
    if not (pull_raw[1] < cube_raw[0] < cube_raw[1] < close_raw[0]):
        raise SystemExit('FAIL 相位次序不对：拉 %s / 方块 %s / 合 %s —— '
                         '期望"拉 -> 方块 -> 合"' % (pull_raw, cube_raw, close_raw))
    # 合抽屉那一段抽屉爪必须全程张着（它是推，不是夹）
    claw_close_min = float(raw_action[close_raw[0]:close_raw[1] + 1, DRAWER_CLAW].min())
    claw_close_max = float(raw_action[close_raw[0]:close_raw[1] + 1, DRAWER_CLAW].max())
    if claw_close_min < 0.04:
        raise SystemExit('FAIL 合抽屉那一段抽屉爪最小 %.4f（<0.04）—— 那一段的爪应当一直张着'
                         % claw_close_min)

    limits = np.full(14, JOINT_RATE / FPS * args.stretch_margin)
    limits[GRIPPERS] = GRIPPER_RATE / FPS * args.stretch_margin
    # 让步长**严格**小于帽：恰好等于帽时客户端 limited_target 那步浮点运算会留下 ~1e-19
    # 的"实切"。1e-9 的收紧在物理上是 1e-11 rad，等于 0，但能保证它一次都不触发。
    limits *= (1.0 - 1e-9)
    action, svals = coordinated_stretch(raw_action, limits, args.max_step)
    state, _ = coordinated_stretch(raw_state, np.full(14, np.inf), args.max_step)
    if len(state) != len(action):
        state = state[:len(action)] if len(state) > len(action) else np.vstack(
            [state, np.tile(state[-1], (len(action) - len(state), 1))])
    H_walk = len(action)

    # 拉伸是线性插值：端点值恰好压在盒边界上时（例：lJ3 恰为 0）会带出 ~1e-10 的浮点残差。
    # 先断言残差只是噪声（不是真越界），再夹回盒内 —— 夹的量在物理上等于 0。
    _pre = np.maximum(np.maximum(LOW - action, action - HIGH), 0.0)
    if float(_pre.max()) > 1e-6:
        raise SystemExit('FAIL 拉伸后真越界，最大 %.6g（不只是浮点噪声）' % float(_pre.max()))
    action = np.clip(action, LOW, HIGH)

    reset_target = raw_state[0].copy()

    def to_walk(t_raw):
        """段内参照拍号 -> 前置保持之后的输出拍号。"""
        return int(np.searchsorted(svals, float(t_raw), side='left'))

    phases = {
        'pull': {'raw': [int(pull_raw[0]), int(pull_raw[1])],
                 'walk': [to_walk(pull_raw[0]), to_walk(pull_raw[1])],
                 'what': '抽屉臂伸手 -> 张爪 -> 咬住拉环 -> 拉开约 11 cm -> 松爪'},
        'cube': {'raw': [int(cube_raw[0]), int(cube_raw[1])],
                 'walk': [to_walk(cube_raw[0]), to_walk(cube_raw[1])],
                 'what': '方块臂伸过去夹起蓝方块、移到抽屉口松爪放下'},
        'close': {'raw': [int(close_raw[0]), int(close_raw[1])],
                  'walk': [to_walk(close_raw[0]), to_walk(close_raw[1])],
                  'what': '抽屉臂再次前伸/退回，爪全程张开 = 把抽屉合上'},
    }

    # --- 归位段（语料里没有，按用户要求补） -------------------------------------
    return_meta = None
    if args.no_return:
        print('   >> --no-return：不补归位段，轨迹停在演示的收尾位姿上')
    else:
        ret, n_move = return_rows(action[-1], reset_target, limits, RETURN_HOLD)
        action = np.vstack([action, ret])
        state = np.vstack([state, np.tile(reset_target, (len(ret), 1))])
        return_meta = {'walk': [H_walk + 1, H_walk + n_move],
                       'rows_move': int(n_move),
                       'rows_hold': int(RETURN_HOLD),
                       'seconds': (n_move + RETURN_HOLD) / FPS,
                       'target_is_reset_target': True,
                       'note': '关节空间余弦缓动，每拍步长同样进帽；末行逐位等于复位目标。'
                               '演示里没有这条路线（演示第 738 拍之后两条臂逐字静止）。'}
        action = np.clip(action, LOW, HIGH)         # 端点已在盒内、直线插值不会出盒；夹一下只是收尾
        print('   归位段（新增，语料里没有）：%d 拍移动 + %d 拍保持 = %.2f s，终点 = 复位目标'
              % (n_move, RETURN_HOLD, return_meta['seconds']))
    H_noreturn = H_walk + (0 if args.no_return else return_meta['rows_move'])
    H_no_hold = len(action)

    # --- 自检 1：每拍步长都在帽内 ----------------------------------------------
    rr = rates(action, JOINT_RATE, GRIPPER_RATE)
    if rr['joint_rows_over_cap'] or rr['jaw_rows_over_cap']:
        raise SystemExit('FAIL 拉伸后仍有拍超帽：关节 %d 拍、夹爪 %d 拍'
                         % (rr['joint_rows_over_cap'], rr['jaw_rows_over_cap']))
    excess = np.maximum(np.maximum(LOW - action, action - HIGH), 0.0)
    if float(excess.max()) > 0.0:
        raise SystemExit('FAIL 拉伸后有行越出 LOW/HIGH，最大 %.6g；'
                         '客户端的 validated_actions 会拒绝并触发重采样，游标会跳行'
                         % float(excess.max()))

    # --- 自检 2：走一遍真实客户端管线，限速器必须一次都不动手 -------------------
    for _alpha in (ACTION_FILTER_ALPHA, 1.0, 0.5, 0.1):
        _F, _C = real_pipeline(action, _alpha, JOINT_RATE, GRIPPER_RATE)
        _cut = float(np.abs(_F - _C).max())
        if _cut > 1e-12:
            raise SystemExit('FAIL alpha=%g 时限速器实切 %.6g（应 <1e-12）；轨迹没被压到帽内'
                             % (_alpha, _cut))
    F, C = real_pipeline(action, ACTION_FILTER_ALPHA, JOINT_RATE, GRIPPER_RATE)
    cut = float(np.abs(F - C).max())

    # --- 自检 2b：不许撞驱动护栏 verified_hardware.py:474/476 ------------------
    dt = 1.0 / FPS
    j_allow, g_allow = 0.25 * dt + 0.01, 0.006 * dt + 0.001
    dC = np.abs(np.diff(C, axis=0))
    j_max, g_max = float(dC[:, JOINTS].max()), float(dC[:, GRIPPERS].max())
    if j_max >= j_allow or g_max >= g_allow:
        raise SystemExit('FAIL 会撞驱动护栏：关节 %.5f/%.5f、夹爪 %.5f/%.5f'
                         % (j_max, j_allow, g_max, g_allow))
    driver_guard = {'joint_allow_rad_per_tick': j_allow,
                    'jaw_allow_m_per_tick': g_allow,
                    'joint_step_max_rad': j_max, 'jaw_step_max_m': g_max,
                    'joint_ratio': j_max / j_allow, 'jaw_ratio': g_max / g_allow}

    # --- 自检 3：拉抽屉那一段，咬住的那一拍抽屉臂必须在演示的咬住位上 -----------
    ref_peak, ref_t = grip_tick(raw_action, DRAWER_CLAW, lo=pull_raw[0], hi=pull_raw[1] + 1)
    if ref_t is None:
        raise SystemExit('FAIL 参照段里抽屉爪张到 %.4f 之后再没合到 %.3f —— 这集不是一次拉抽屉'
                         % (raw_action[ref_peak, DRAWER_CLAW], GRIP_SHUT))
    grasp = raw_action[ref_t, DRAWER]
    got_peak, got_t = grip_tick(C, DRAWER_CLAW, lo=0, hi=phases['pull']['walk'][1] + 1)
    dev = float(np.abs(C[got_t, DRAWER] - grasp).max()) if got_t is not None else float('inf')
    if got_t is None or dev > 0.02:
        raise SystemExit('FAIL 爪咬住那拍抽屉臂离演示位 %.4f rad（>0.02）—— 时序没保住' % dev)
    # 相位 A 里抽屉臂真的拉了一把
    lj2_min = float(raw_action[pull_raw[0]:pull_raw[1] + 1, 1].min())
    lj2_max = float(raw_action[pull_raw[0]:pull_raw[1] + 1, 1].max())
    lj2_end = float(raw_action[pull_raw[1], 1])
    if not (lj2_max > 1.0 and abs(lj2_end - lj2_min) > 0.8):
        raise SystemExit('FAIL 拉抽屉那一段的 lJ2 不成立：%.4f..%.4f 末 %.4f'
                         % (lj2_min, lj2_max, lj2_end))

    # --- 自检 4：放方块那一段，方块爪必须"张开 -> 合上 -> 再张开" ----------------
    cb = raw_action[cube_raw[0]:cube_raw[1] + 1, CUBE_CLAW]
    cb_peak = int(np.argmax(cb))
    cb_open = float(cb[cb_peak])
    cb_after = cb[cb_peak:]
    cb_shut_i = int(np.argmin(cb_after))
    cb_shut = float(cb_after[cb_shut_i])
    cb_reopen = float(cb[cb_peak + cb_shut_i:].max()) if cb_shut_i < len(cb_after) else cb_shut
    if not (cb_open >= 0.05 and cb_shut <= 0.04 and cb_reopen >= 0.05):
        raise SystemExit('FAIL 方块爪的"张开->合上->再张开"不成立：%.4f -> %.4f -> %.4f'
                         % (cb_open, cb_shut, cb_reopen))
    rj2_span = float(raw_action[cube_raw[0]:cube_raw[1] + 1, 8].max()
                     - raw_action[cube_raw[0]:cube_raw[1] + 1, 8].min())
    if rj2_span < 1.0:
        raise SystemExit('FAIL 方块臂在"放方块"那一段只动了 %.4f rad —— 不像是取放' % rj2_span)

    # --- 自检 5：合抽屉那一段，抽屉臂确实再动了一次 ----------------------------
    close_lj2 = raw_action[close_raw[0]:close_raw[1] + 1, 1]
    close_span = float(close_lj2.max() - close_lj2.min())
    if close_span < 0.20:
        raise SystemExit('FAIL 合抽屉那一段抽屉臂 lJ2 只动了 %.4f rad（<0.20）' % close_span)

    # --- 自检 6：归位段的末行必须逐位等于复位目标 ------------------------------
    ret_dev = float(np.abs(action[-1] - reset_target).max()) if not args.no_return else None
    if not args.no_return and ret_dev != 0.0:
        raise SystemExit('FAIL 归位段末行与复位目标差 %.6g —— 跑完手臂不会坐在复位点上'
                         % ret_dev)

    # --- 前置"保持不动"行 ------------------------------------------------------
    lead = max(0, int(args.lead_hold))
    if lead:
        action = np.vstack([np.tile(action[0], (lead, 1)), action])
        state = np.vstack([np.tile(state[0], (lead, 1)), state])
    H = len(action)

    home_obj = {
        'target': [round(float(v), 10) for v in reset_target],
        'source': '%s:%s 第 %d 拍（含）的录制 observation.state；由 make_full_traj.py 生成'
                  % (path, args.episode, args.t0),
        'description': '全任务回放的复位目标 = 演示起跑那一拍手臂真正所在的位置'
                       '（抽屉关着、两条臂都坐在 J2 地板/J3 天花板上、两个爪都合着、'
                       '蓝方块在圆盘上）。回放从这一拍起放，所以复位到它，第 0 拍才不是跳变；'
                       '它同时是末尾"归位段"的终点 —— 跑完手臂回到这里，可以原地再跑一次。',
        'model_slot_can_order': ['can1', 'can0'],
        'units_per_arm': ['joint radians x6', 'gripper metres'],
        'derived_by': 'make_full_traj.py（段 [%d,%d]，读写同一行，保证 json.target == traj.state[0] '
                      '== 归位段末行）' % (args.t0, args.t1),
        'operator_scene_required': ['抽屉完全关上（这一版是机器自己拉开：它夹的是抽屉正面左上'
                                    '那个黑色拉环，那是抽屉自己的把手，不用另外摆）',
                                    '蓝方块在圆盘上、位置与演示一致（圆盘左半边）；'
                                    '这一版方块臂会去夹它并放进抽屉',
                                    '两个爪都是合着的、里面没有东西',
                                    '工作区里没有人手'],
        'validation': '全部 14 维落在 pi0_rollout_v8.LOW/HIGH 之内（max excess %.6g）；'
                      '启动器断言本文件 target 与轨迹 state[0] 逐位相等，离线测试再核一遍。'
                      % float(np.maximum(np.maximum(LOW - reset_target, reset_target - HIGH),
                                         0.0).max()),
        'note': '与 cubephase_20261015/home_target_fulltask.json 是同一类东西（全任务起点），'
                '但那份来自 live home JSON、来源不同；本文件的 target 直接取本集第 %d 拍的'
                '录制 state，与轨迹逐位自洽。'
                % args.t0,
    }
    _oob = float(np.maximum(np.maximum(LOW - reset_target, reset_target - HIGH), 0.0).max())
    if _oob > 0.0:
        raise SystemExit('FAIL 复位目标越出 LOW/HIGH %.6g —— 客户端加载时就会拒绝' % _oob)
    if not home_obj['source']:
        raise SystemExit('FAIL 复位目标缺 source（客户端要求非空）')

    meta = {
        'replay_id': 'replay://drawer_full_today99/%s/t%d_%d/h%d'
                     % (args.episode.replace('.parquet', ''), args.t0, args.t1, H),
        'kind': 'recorded_demo_replay_time_stretched',
        'model': None,
        'source_file': args.episode,
        'segment_ticks': [int(args.t0), int(args.t1)],
        'horizon_raw': H_raw,
        'horizon_stretched_walk': H_walk,
        'horizon': H,
        'horizon_seconds': H / FPS,
        'lead_hold_rows': lead,
        'stretch_margin': float(args.stretch_margin),
        'max_step_ref_per_tick': float(args.max_step),
        'stretch_factor_walk': H_walk / H_raw,
        'stretch_factor_total': H / H_raw,
        'phases': phases,
        'return_home': return_meta,
        'horizon_without_return': H_noreturn + lead,
        'start_row': action[0].tolist(),
        'end_row': action[-1].tolist(),
        'reset_target': [float(v) for v in reset_target],
        'home_target': [float(v) for v in reset_target],
        'home_json': os.path.abspath(args.home_out),
        'limits_max_excess_over_low_high': float(excess.max()),
        'limits_preclamp_residual': float(_pre.max()),
        'action_rates': rr,
        'action_rates_raw': rates(raw_action, JOINT_RATE, GRIPPER_RATE),
        'pipeline_filter_alpha': ACTION_FILTER_ALPHA,
        'pipeline_limiter_cut_max': cut,
        'driver_guard': driver_guard,
        'pipeline_grip_tick': got_t,
        'pipeline_grip_arm_dev_rad': dev,
        'reference_grip_tick': ref_t,
        'reference_open_peak_tick': ref_peak,
        'drawer_claw_open_peak_raw': float(raw_action[ref_peak, DRAWER_CLAW]),
        'drawer_claw_grip_raw': float(raw_action[ref_t, DRAWER_CLAW]),
        'drawer_claw_end_raw': float(raw_action[-1, DRAWER_CLAW]),
        'drawer_arm_lJ2_min': lj2_min,
        'drawer_arm_lJ2_max': lj2_max,
        'drawer_arm_lJ2_end': lj2_end,
        'drawer_claw_close_phase_min_raw': claw_close_min,
        'drawer_claw_close_phase_max_raw': claw_close_max,
        'close_phase_lJ2_span_raw': close_span,
        'cube_claw_open_raw': cb_open,
        'cube_claw_shut_raw': cb_shut,
        'cube_claw_reopen_raw': cb_reopen,
        'cube_arm_rJ2_span_raw': rj2_span,
        'cube_arm_first_move_tick_raw': int(cube_raw[0]),
        'return_end_dev_from_reset_target': ret_dev,
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'axis_names': NAMES,
    }

    with open(args.out + '.json', 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write('\n')
    np.savez(args.out + '.npz', action=action, state=state,
             meta=np.array(json.dumps(meta, sort_keys=True)))
    with open(args.home_out, 'w') as fh:
        json.dump(home_obj, fh, indent=2, sort_keys=True)
        fh.write('\n')

    print('   协调拉伸后      %d 拍 = %.2f s（%.2f 倍）' % (H_walk, H_walk / FPS, H_walk / H_raw))
    print('   + 前置 %d 拍保持不动 ⇒ 演示部分到第 %d 拍 = %.1f s'
          % (lead, H_noreturn + lead, (H_noreturn + lead) / FPS))
    print('   合计            %d 拍 = %.1f s（含归位段）' % (H, H / FPS))
    for k in ('pull', 'cube', 'close'):
        w, r = phases[k]['walk'], phases[k]['raw']
        print('   相位 %-6s 原 %3d..%-3d 拍 → 输出 %4d..%-4d 拍 = %5.1f..%5.1f s   %s'
              % (k, r[0], r[1], w[0] + lead, w[1] + lead,
                 (w[0] + lead) / FPS, (w[1] + lead) / FPS, phases[k]['what']))
    if return_meta:
        w = return_meta['walk']
        print('   相位 %-6s 输出 %4d..%-4d 拍 = %5.1f..%5.1f s   %s'
              % ('return', w[0] + lead, w[1] + lead,
                 (w[0] + lead) / FPS, (w[1] + lead) / FPS,
                 '两条臂平滑回到复位目标，再保持 %.1f s' % (return_meta['rows_hold'] / FPS)))
    print('   拉伸后关节单拍 max %.5f / 帽 %.5f rad（超限 %d 拍）'
          % (rr['joint_step_max_rad'], rr['joint_cap_rad_per_tick'], rr['joint_rows_over_cap']))
    print('   拉伸后夹爪单拍 max %.5f / 帽 %.5f m （超限 %d 拍，最紧那根是 dim%d）'
          % (rr['jaw_step_max_m'], rr['jaw_cap_m_per_tick'], rr['jaw_rows_over_cap'],
             rr['jaw_step_max_dim']))
    print('   越出 LOW/HIGH 的行 %d 个（必须 0，否则闸门会触发重采样）'
          % int((excess > 0).sum()))
    print('   >> 真实客户端管线后，limited_target 实切量 = %.3g（必须 <1e-12）' % cut)
    print('   >> 驱动护栏余量：关节 %.5f/%.5f（%.0f%%）、夹爪 %.5f/%.5f（%.0f%%）'
          % (driver_guard['joint_step_max_rad'], driver_guard['joint_allow_rad_per_tick'],
             100 * driver_guard['joint_ratio'], driver_guard['jaw_step_max_m'],
             driver_guard['jaw_allow_m_per_tick'], 100 * driver_guard['jaw_ratio']))
    print('   >> 拉抽屉：爪张开峰第 %d 拍（%.4f）→ 咬住第 %d 拍（%.4f）；重采样后咬住第 %d 拍'
          % (ref_peak, meta['drawer_claw_open_peak_raw'], ref_t,
             meta['drawer_claw_grip_raw'], got_t))
    print('   >> 咬住那拍抽屉臂离演示位 %.4f rad（必须 <=0.02）；拉量 lJ2 %.4f -> 峰 %.4f -> 末 %.4f'
          % (dev, lj2_min, lj2_max, lj2_end))
    print('   >> 放方块：方块爪 %.4f（张）-> %.4f（合）-> %.4f（松）；方块臂 rJ2 跨 %.4f rad'
          % (cb_open, cb_shut, cb_reopen, rj2_span))
    print('   >> 合抽屉：抽屉臂 lJ2 跨 %.4f rad，爪全程 %.4f..%.4f（张着 = 推不是夹）'
          % (close_span, claw_close_min, claw_close_max))
    if ret_dev is not None:
        print('   >> 归位段末行与复位目标差 %.3g（必须 0）' % ret_dev)
    print('   写了 %s / %s' % (args.out + '.npz', args.out + '.json'))
    print('   复位目标（= traj.state[0] = 归位段末行）写进 %s' % args.home_out)
    print('   启动器建议窗口 --rollout-seconds %d' % (int(np.ceil(H / FPS)) + 6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
