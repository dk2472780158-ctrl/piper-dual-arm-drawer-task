#!/usr/bin/env python3
"""切出"拉抽屉"那一段录制轨迹，并把它重采样到一条**共享时间刻度**上。

要解决的问题
------------
用户要单测"机器自己能不能把抽屉拉开"。演示里那一段就是这个动作本身：

    today99 / episode_000041 的第 48..300 拍（8.43 s）
      第 48–110  伸手到关着的抽屉前（抽屉臂 0..6 维动，方块臂 7..13 维**逐字不动**）
      第 163     爪张到全开（0.0700）
      第 214     爪合到底（<=0.01）＝ **咬住抽屉正面左上的黑色拉环**（最要紧的一拍）
      第 220–270 拉：dim1 1.505 -> 1.075（0.43 rad，按前臂半径折算 ≈11 cm，
                 与"手拉开约 11 cm"的既有记录一致）
      第 280–290 松开（爪张回 0.0589）
      第 290–300 停住不动
    方块臂第 333 拍才起动 —— 所以段终点取 300，这一段里方块臂是常量（实测偏离
    首行 0.00000）。

为什么不能"原样放"
------------------
执行链每拍压步长（客户端 `limited_target` 关节 0.40 rad/s、夹爪 0.030 m/s；
驱动护栏 `verified_hardware.py:474/476` 是 `.25*dt+.01` rad / `.006*dt+.001` m）。
这一段实测单拍峰值：抽屉臂 dim4 **0.04780 rad**（超客户端帽 4.0 倍）、dim1 的
快速段 0.0144 rad/拍（超 1.2 倍）、抽屉爪 0.01560 m（超 15.6 倍）。

限速器把"冲过去、到点停住"切成斜坡 ⇒ **咬住那一拍与手臂到位被拉成两条时间线**
（这就是 cubephase 那一份修掉的同一个病）。拉抽屉对时序敏感的地方正是这里：
爪必须**在手臂停在拉环上的那一拍**咬住，早一点咬空、晚一点手臂已经越过去了。

这一段的物理过程已由抽帧核对过（只读语料，未碰活相机）：抽屉爪在 202–220 拍从全开
合到 0.0029 m，并在 218–279 拍一直保持 0.0026–0.0032 m —— 两指之间**夹着约 3 mm
厚的**东西，是真夹住，不是空合；抽屉约 230 拍开始动、290–300 拍拉到位置；拉环是
抽屉自己的把手（不是另摆的圈），臂做完拉的动作后在 282–294 拍松开、停在抽屉前不动。

本脚本的做法：协调时间重参数化
--------------------------------
把整段重采样到**一条共享的参照时刻 s** 上，所有 14 轴永远处在同一个 s；每拍只把
s 推进到"最慢的那一根轴允许的最大值"（`coordinated_stretch`）。于是：

  * 相对时序**逐字保住** —— 演示里"手臂停住"和"咬住"落在同一个参照时刻，
    重采样后仍然如此。
  * 每拍步长被压到帽内 ⇒ 客户端 `limited_target` 变成**空操作**，发什么就是什么。
  * 不碰任何安全阈值、不改任何一行客户端代码。

代价是时间变长（实测约 1.8 倍，8.4 s -> 约 15 s）。对这件事无害：**抽屉不会动**，
慢一点照样咬得住。

首行前置 --lead-hold 拍"保持不动"
--------------------------------
服务端的游标一旦 arm 就按 10 行/次推进。如果被动 peek（客户端真动之前会先打一次，
此时手臂还在上一轮留下的位姿上）恰好也在起跑位姿上，第一个 10 行块会被 peek 吃掉。
前置 10 拍保持不动行，这种"跳掉 10 行"就退化成无害的 0.33 s 静止。

复位目标
--------
本脚本同时写出 `home_target_pull.json`，它的 `target` = 段首的**录制 state**
（= 手臂当时真正在的地方）。客户端 `hardware.home(target)` 会把两条臂开到那里，
`settle_home` 只认"关节 0.035 rad、夹爪 0.004 m 以内且稳 0.5 s"。启动器会断言
`json.target` 与轨迹 `state[0]` 逐位相等（两边都由本脚本从同一行生成）。

输出
----
  pull_traj.npz        action (H,14) / state (H,14) / meta (0 维 JSON 字符串)
  pull_traj.json       同一份 meta，给 shell 启动器用普通 json 读
  home_target_pull.json 复位目标（= 段首 state）
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
DRAWER = np.array([0, 1, 2, 3, 4, 5, 6])
CUBE = np.array([7, 8, 9, 10, 11, 12, 13])
LOW = np.tile([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439, 0], 2)
HIGH = np.tile([2.6179, 3.14, 0, 1.745, 1.22, 2.09439, .07], 2)
DRAWER_CLAW = 6
FPS = 30.0
# 与 pi0_rollout_v8.py / pi0_real_trial_v8.py 的默认值同源。启动器会 export 这两个
# 数，并断言与 meta 里记的一致 —— 轨迹是照着它们建的，对不上就白建。
JOINT_RATE = 0.40        # rad/s，limited_target 的关节限速
GRIPPER_RATE = 0.030     # m/s，PI0_GRIPPER_RATE 默认
ACTION_FILTER_ALPHA = 0.25   # PI0_ACTION_ALPHA 默认；只滤关节、夹爪直通

# 固定段落：episode_000041 的第 48..300 拍
SEG_T0 = 48
SEG_T1 = 300
EPISODE = 'episode_000041.parquet'
# 抽屉爪"咬住"的判据：张开峰之后再落到 <= 这个值（爪值＝开口宽度，小＝合）
GRIP_SHUT = 0.01


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='/home/databall/drawer_today99_v1/dataset_train')
    p.add_argument('--episode', default=EPISODE)
    p.add_argument('--t0', type=int, default=SEG_T0, help='段起点（含）')
    p.add_argument('--t1', type=int, default=SEG_T1, help='段终点（含）')
    p.add_argument('--home-out', default=os.path.join(here, 'home_target_pull.json'))
    p.add_argument('--out', default=os.path.join(here, 'pull_traj'))
    p.add_argument('--stretch-margin', type=float, default=0.90,
                   help='每拍步长只用到帽的这个比例（<1 留抖动余量）')
    p.add_argument('--lead-hold', type=int, default=10,
                   help='首部前置多少拍"保持不动"')
    p.add_argument('--max-step', type=float, default=1.0,
                   help='每拍最多推进多少个参照拍（1.0 = 永不快于原速）')
    return p.parse_args(argv)


def load_segment(data, episode, t0, t1):
    """只读那一集、只切那一段。段里方块臂必须逐字不动，否则说明端点选错了。"""
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
    cube_drift = float(np.abs(raw_action[:, CUBE] - raw_action[0, CUBE]).max())
    if cube_drift > 0.01:
        raise SystemExit('FAIL 段的方块臂动了 %.5f rad（>0.01）：端点应该切在方块臂起动之前'
                         % cube_drift)
    return raw_action, raw_state, path, cube_drift


def _at(traj, s):
    """参照时刻 s（实数）处的线性插值。"""
    last = len(traj) - 1
    if s >= last:
        return traj[last].copy()
    j = int(np.floor(s))
    f = s - j
    return traj[j] * (1.0 - f) + traj[j + 1] * f


def coordinated_stretch(traj, limits, max_step):
    """把 traj 重采样到一条共享时间刻度：所有轴永远在同一参照时刻，
    每拍把该时刻推进到"最慢那根轴允许的最大值"。相对时序因此逐字保住。"""
    out = [traj[0].copy()]
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
        guard += 1
        if guard > 100000:
            raise SystemExit('FAIL 拉伸循环不收敛')
    return np.array(out)


def real_pipeline(D, alpha, joint_rate, gripper_rate):
    """把服务端将要发的 D 走一遍**真实的客户端管线**：

        _apply_action_filter(α, 关节序贯低通、夹爪直通)
        -> limited_target(commanded, row, dt, joint_rate, gripper_rate)

    返回 (滤波后 F, 限速后 C)。启动器断言 |F-C| 的 max == 0：只要为 0，
    就说明限速器一次都没动手，我发什么机器人就收到什么。"""
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
    """两个夹爪都算（这一段关键的是抽屉爪 dim6，不是方块爪 dim13）。"""
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


def grip_tick(block, col=DRAWER_CLAW, shut=GRIP_SHUT):
    """"咬住"那一拍：先把该爪张开到峰，峰之后第一次落到 <= shut。
    返回 (峰值拍, 咬住拍)。"""
    peak = int(np.argmax(block[:, col]))
    after = np.flatnonzero(block[peak:, col] <= shut)
    return peak, (peak + int(after[0]) if len(after) else None)


def main():
    args = parse_args()
    raw_action, raw_state, path, cube_drift = load_segment(
        args.data, args.episode, args.t0, args.t1)
    H_raw = len(raw_action)
    print('语料 %s' % args.data)
    print('   源 %s 第 %d..%d 拍（含）= %d 拍 = %.2f s'
          % (args.episode, args.t0, args.t1, H_raw, H_raw / FPS))
    print('   段内方块臂 7 维偏离首行 = %.5f rad（必须 ~0：这一段只该动抽屉臂）' % cube_drift)

    limits = np.full(14, JOINT_RATE / FPS * args.stretch_margin)
    limits[GRIPPERS] = GRIPPER_RATE / FPS * args.stretch_margin
    # 让步长**严格**小于帽：恰好等于帽时，客户端 limited_target 那步浮点运算会留下
    # ~1e-19 的"实切"。1e-9 的收紧在物理上是 1e-11 rad，等于 0，但能保证它一次都不触发。
    limits *= (1.0 - 1e-9)
    action = coordinated_stretch(raw_action, limits, args.max_step)
    state = coordinated_stretch(raw_state, np.full(14, np.inf), args.max_step)
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

    # --- 自检 1：每拍步长都在帽内（⇒ 客户端限速器空操作、闸门不拒绝） -------------
    rr = rates(action, JOINT_RATE, GRIPPER_RATE)
    if rr['joint_rows_over_cap'] or rr['jaw_rows_over_cap']:
        raise SystemExit('FAIL 拉伸后仍有拍超帽：关节 %d 拍、夹爪 %d 拍'
                         % (rr['joint_rows_over_cap'], rr['jaw_rows_over_cap']))
    excess = np.maximum(np.maximum(LOW - action, action - HIGH), 0.0)
    if float(excess.max()) > 0.0:
        raise SystemExit('FAIL 拉伸后有行越出 LOW/HIGH，最大 %.6g；'
                         '客户端的 validated_actions 会拒绝并触发重采样，游标会跳行'
                         % float(excess.max()))

    # --- 自检 2：走一遍真实客户端管线，限速器必须一次都不动手 ---------------------
    for _alpha in (ACTION_FILTER_ALPHA, 1.0, 0.5, 0.1):
        _F, _C = real_pipeline(action, _alpha, JOINT_RATE, GRIPPER_RATE)
        _cut = float(np.abs(_F - _C).max())
        if _cut > 1e-12:
            raise SystemExit('FAIL alpha=%g 时限速器实切 %.6g（应 <1e-12）；轨迹没被压到帽内'
                             % (_alpha, _cut))
    F, C = real_pipeline(action, ACTION_FILTER_ALPHA, JOINT_RATE, GRIPPER_RATE)
    cut = float(np.abs(F - C).max())

    # --- 自检 2b：不许撞驱动护栏 verified_hardware.py:474/476 -------------------
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

    # --- 自检 3：爪咬住的那一拍，抽屉臂必须在演示的咬住位上 ---------------------
    ref_peak, ref_t = grip_tick(raw_action)
    if ref_t is None:
        raise SystemExit('FAIL 参照段里抽屉爪张到 %.4f 之后再没合到 %.3f —— 这集不是一次拉抽屉'
                         % (raw_action[ref_peak, DRAWER_CLAW], GRIP_SHUT))
    grasp = raw_action[ref_t, DRAWER]
    got_peak, got_t = grip_tick(C)
    dev = float(np.abs(C[got_t, DRAWER] - grasp).max()) if got_t is not None else float('inf')
    if got_t is None or dev > 0.02:
        raise SystemExit('FAIL 爪咬住那拍抽屉臂离演示位 %.4f rad（>0.02）—— 时序没保住' % dev)

    # --- 前置"保持不动"行 --------------------------------------------------------
    lead = max(0, int(args.lead_hold))
    if lead:
        action = np.vstack([np.tile(action[0], (lead, 1)), action])
        state = np.vstack([np.tile(state[0], (lead, 1)), state])
    H = len(action)

    reset_target = raw_state[0]
    home_obj = {
        'target': [round(float(v), 10) for v in reset_target],
        'source': '%s:%s 第 %d 拍（含）的录制 observation.state；由 make_pull_traj.py 生成'
                  % (path, args.episode, args.t0),
        'description': '拉抽屉单测的复位目标 = 演示里"伸手到关着的抽屉前"那一拍手臂真正所在的位置'
                       '（抽屉关着、抽屉臂的爪合到底、方块臂在原位）。轨迹从这一拍起放，'
                       '所以复位到它，回放的第 0 拍才不是一次跳变。',
        'model_slot_can_order': ['can1', 'can0'],
        'units_per_arm': ['joint radians x6', 'gripper metres'],
        'derived_by': 'make_pull_traj.py（段 [%d,%d]，读写同一行，保证 json.target == traj.state[0]）'
                      % (args.t0, args.t1),
        'operator_scene_required': ['抽屉完全关上（这一版就是让机器自己拉开：'
                                    '它夹的是抽屉正面左上那个黑色拉环，'
                                    '那是抽屉自己的把手，不用另外摆）',
                                    '爪里没有东西',
                                    '蓝方块留在圆盘上（这一段方块臂不动）',
                                    '工作区里没有人手'],
        'validation': '全部 14 维落在 pi0_rollout_v8.LOW/HIGH 之内（%s）；方块臂在这一段里'
                      '逐字不动（偏离首行 %.5f rad）。启动器会断言本文件 target 与轨迹 '
                      'state[0] 逐位相等，并与 cubephase 那份停放位的距离一起打印。'
                      % ('max excess %.6g' % float(np.maximum(
                          np.maximum(LOW - reset_target, reset_target - HIGH), 0.0).max()),
                         cube_drift),
        'note': '这是**拉抽屉单测**的复位目标，不是全任务起点。全任务起点见 '
                'cubephase_20261015/home_target_fulltask.json（与演示第 0 拍同构，'
                '与本文件差 0.095 rad 左右，全在方块臂的腕上）。',
    }
    _oob = float(np.maximum(np.maximum(LOW - reset_target, reset_target - HIGH), 0.0).max())
    if _oob > 0.0:
        raise SystemExit('FAIL 复位目标越出 LOW/HIGH %.6g —— 客户端加载时就会拒绝' % _oob)
    if not home_obj['source']:
        raise SystemExit('FAIL 复位目标缺 source（客户端要求非空）')

    meta = {
        'replay_id': 'replay://drawer_pull_today99/%s/t%d_%d/h%d'
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
        'drawer_arm_lJ2_min': float(raw_action[:, 1].min()),
        'drawer_arm_lJ2_max': float(raw_action[:, 1].max()),
        'drawer_arm_lJ2_end': float(raw_action[-1, 1]),
        'drawer_arm_sweep_l_inf_action': float(
            np.abs(raw_action[-1, DRAWER] - raw_action[0, DRAWER]).max()),
        'drawer_arm_sweep_l_inf_state': float(
            np.abs(raw_state[-1, DRAWER] - raw_state[0, DRAWER]).max()),
        'cube_arm_drift_raw': cube_drift,
        'cube_arm_drift_in_traj': float(np.abs(action[:, CUBE] - action[0, CUBE]).max()),
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

    print('   协调拉伸后      %d 拍 = %.2f s（%.2f 倍）+ 前置 %d 拍保持不动 ⇒ 共 %d 拍 = %.2f s'
          % (H_walk, H_walk / FPS, H_walk / H_raw, lead, H, H / FPS))
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
    print('   >> 抽屉爪：张开峰在第 %d 拍（%.4f）→ 咬住在第 %d 拍（%.4f）；重采样后咬住在第 %d 拍'
          % (ref_peak, meta['drawer_claw_open_peak_raw'], ref_t,
             meta['drawer_claw_grip_raw'], got_t))
    print('   >> 咬住那拍抽屉臂离演示位 %.4f rad（必须 <=0.02）' % dev)
    print('   >> 抽屉臂 lJ2：%.4f -> 峰 %.4f -> 末 %.4f（这段的拉 = %.4f rad）'
          % (meta['drawer_arm_lJ2_min'], meta['drawer_arm_lJ2_max'],
             meta['drawer_arm_lJ2_end'],
             abs(meta['drawer_arm_lJ2_end'] - meta['drawer_arm_lJ2_min'])))
    print('   写了 %s / %s' % (args.out + '.npz', args.out + '.json'))
    print('   复位目标（traj.state[0] 逐位相同）写进 %s' % args.home_out)
    print('   启动器建议窗口 --rollout-seconds %d' % (int(np.ceil(H / FPS)) + 6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
