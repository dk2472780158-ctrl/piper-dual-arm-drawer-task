#!/usr/bin/env python3
"""切出"方块相位"的开环轨迹，并把它重采样到一条**共享时间刻度**上。

要解决的问题
------------
用户的要求："就算抽屉没拉开，方块臂也要把方块夹住放进去"。那条"必须先拉开抽屉"
的限制不是代码，是策略学到的时序程序（活机器上它只在抽屉臂松爪后 1.97 s 才让方块
臂起动）。所以不走策略，把演示的方块相位原样放给 can0。

但"原样放"过不去。执行链每拍压步长：

    客户端 limited_target : 关节 0.40 rad/s、夹爪 0.030 m/s
    驱动护栏 verified_hardware.py:474/476 : .25*dt+.01 rad、.006*dt+.001 m

而演示那一夹是快的：方块臂 rJ2 单拍峰值 0.07471 rad（超客户端帽 5.6 倍）、
方块爪 rjaw 0.01240 m（超 12.4 倍），86/86 集全超。限速器把"加速冲过去、到点停住"
切成斜坡，于是**手臂到位和合爪被拉成两条时间线**（实测错位 94 拍 = 3.13 s）——
爪在手臂还没到的地方就合，或者合完手臂早走了。这就是"下探有问题"。

本脚本的做法：协调时间重参数化
--------------------------------
把整段轨迹重采样到一条**共享**的参照时刻 s 上，所有 14 轴永远处在同一个 s。
每拍只把 s 推进到"最慢的那一轴允许的最大值"。于是：

  * 相对时序**逐字保住** —— 演示里"手臂到位"和"合爪"落在同一个参照时刻，
    重采样后仍然如此。爪在合的时候手臂就在演示的抓取位上。
  * 每拍步长被压到帽内 ⇒ 客户端 limited_target 变成**空操作**，发什么就是什么。
  * 不碰任何安全阈值、不改任何一行客户端代码。

代价是时间变长（实测 1.68 倍，14.9 s → 25.0 s）。对这件事无害：**方块不会动**，
慢一点照样夹得到。

为什么只挑一集、不放中位轨迹
--------------------------
回放是把录下来的行逐拍喂给机器，中位轨迹会喂出任何演示都没做过的位姿。所以代表集
按"离全集中位最近"（公共相位长度内的 L∞）选，再在这最近的 --arm-candidates 集里挑
**首行 state 的方块臂 7 维离起跑位姿最近**的一集。长度取该集**自己**的相位长度，
不是全集最小值（用最小值截断会让手臂停在半路）。

首行前置 --lead-hold 拍"保持不动"
--------------------------------
服务端的 arming 游标一旦 arm 就按 10 行/次推进。如果被动 peek 恰好也在归位位姿上
打了一次，第一个 10 行块会被 peek 吃掉。前置 10 拍保持不动行，这种"跳掉 10 行"就
变成无害的 0.33 s 静止（而且它本来就该静止在归位位姿上）。

输出
----
  cube_phase_traj.npz    action (H,14) / state (H,14) / meta (0 维 JSON 字符串)
  cube_phase_traj.json   同一份 meta，给 shell 启动器用普通 json 读
不复现就重跑本脚本；两侧 meta 必须逐字相等（离线测试会核）。
"""
from __future__ import annotations

import argparse
import glob
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
LOW = np.tile([-2.6179, 0, -2.967, -1.745, -1.22, -2.09439, 0], 2)
HIGH = np.tile([2.6179, 3.14, 0, 1.745, 1.22, 2.09439, .07], 2)
JAW = 13
FPS = 30.0
# 与 pi0_rollout_v8.py / pi0_real_trial_v8.py 的默认值同源。启动器会 export 这两个
# 数，并断言与 meta 里记的一致 —— 轨迹是照着它们建的，对不上就白建。
JOINT_RATE = 0.40        # rad/s，limited_target 的关节限速
GRIPPER_RATE = 0.030     # m/s，PI0_GRIPPER_RATE 默认
ACTION_FILTER_ALPHA = 0.25   # PI0_ACTION_ALPHA 默认；只滤关节、夹爪直通


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='/home/databall/drawer_today99_v1/dataset_train')
    p.add_argument('--home-json', default=os.path.join(here, 'home_target_cubephase.json'))
    p.add_argument('--out', default=os.path.join(here, 'cube_phase_traj'))
    p.add_argument('--phase-jump', type=float, default=0.15,
                   help='方块臂任一维偏离首行超过这个量 = 相位已起动（rad / m）')
    p.add_argument('--arm-candidates', type=int, default=20,
                   help='在中位最近的前 N 集里挑首行最贴起跑位姿的那一集')
    p.add_argument('--stretch-margin', type=float, default=0.90,
                   help='每拍步长只用到帽的这个比例（<1 留抖动余量）')
    p.add_argument('--lead-hold', type=int, default=10,
                   help='首部前置多少拍"保持不动"')
    p.add_argument('--max-step', type=float, default=1.0,
                   help='每拍最多推进多少个参照拍（1.0 = 永不快于原速）')
    return p.parse_args(argv)


def load(data, phase_jump):
    paths = sorted(glob.glob(os.path.join(data, 'data', 'chunk-*', '*.parquet')))
    if not paths:
        raise SystemExit('FAIL 没找到 parquet：%s' % os.path.join(data, 'data', 'chunk-*', '*.parquet'))
    S, A, K, NAME = [], [], [], []
    for p in paths:
        t = pq.read_table(p, columns=['observation.state', 'action'])
        s = np.asarray(t.column('observation.state').to_pylist(), dtype=float)
        a = np.asarray(t.column('action').to_pylist(), dtype=float)
        if s.shape[1] != 14 or a.shape[1] != 14:
            raise SystemExit('FAIL %s 维数不是 14：state %s action %s' % (p, s.shape, a.shape))
        n = min(len(s), len(a))                 # 数据完整性自检：state/action 行数必须等长
        if len(s) != len(a):
            print('  ! %s state %d 行 action %d 行，取前 %d 行' % (os.path.basename(p), len(s), len(a), n))
        s, a = s[:n], a[:n]
        d = np.max(np.abs(s[:, 7:14] - s[0, 7:14]), axis=1)
        k = np.flatnonzero(d > phase_jump)
        if not len(k):
            continue
        S.append(s)
        A.append(a)
        K.append(int(k[0]))
        NAME.append(os.path.basename(p))
    if not S:
        raise SystemExit('FAIL 没有一集在方块臂上出现 >%g 的偏离' % phase_jump)
    return S, A, np.array(K, dtype=int), NAME, len(paths)


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
            raise SystemExit('FAIL 拉伸在原地卡住（s=%.4f，槽 %d）' % (s, int(np.argmax(
                np.abs(_at(traj, s + 1.0) - prev) - limits) if s < last else 0)))
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
    dj = np.abs(np.diff(block, axis=0))[:, JOINTS]
    dg = np.abs(np.diff(block, axis=0))[:, JAW]
    jcap, gcap = joint_rate / FPS, gripper_rate / FPS
    return {'joint_step_max_rad': float(dj.max()),
            'joint_rows_over_cap': int((dj > jcap + 1e-12).any(axis=1).sum()),
            'jaw_step_max_m': float(dg.max()),
            'jaw_rows_over_cap': int((dg > gcap + 1e-12).sum()),
            'rows': int(len(dj)),
            'joint_cap_rad_per_tick': jcap, 'jaw_cap_m_per_tick': gcap}


def cross(v, thr):
    idx = np.flatnonzero(v >= thr)
    return int(idx[0]) if len(idx) else None


def main():
    args = parse_args()
    S, A, K, NAME, n_total = load(args.data, args.phase_jump)
    shortest = int(min(len(s) - k for s, k in zip(S, K)))
    print('语料 %s' % args.data)
    print('   parquet %d 个，有方块相位的 %d 个，公共相位长度 %d 拍（%.2f s）'
          % (n_total, len(S), shortest, shortest / FPS))

    blk = np.array([a[k:k + shortest] for a, k in zip(A, K)])
    n = len(blk)
    dist = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = float(np.abs(blk[i] - blk[j]).max())
    median_l_inf = np.median(dist, axis=1)

    home = np.asarray(json.load(open(args.home_json))['target'], dtype=float)
    if home.shape != (14,):
        raise SystemExit('FAIL %s 的 target 不是 14 维：%s' % (args.home_json, home.shape))

    # 选集：先取离中位最近的前 N 集，再在这些里挑**方块臂那 7 维**首行最贴起跑位姿的。
    # 只看方块臂：抽屉臂在演示第 0 拍还夹在抽屉拉环上（离 home 0.33+ rad），算进去就选歪了。
    order = np.argsort(median_l_inf)
    pool = order[:max(1, int(args.arm_candidates))]
    start_gap_cube = np.array([float(np.abs(S[i][K[i], 7:14] - home[7:14]).max()) for i in range(n)])
    start_gap_all = np.array([float(np.abs(S[i][K[i]] - home).max()) for i in range(n)])
    i = int(pool[int(np.argmin(start_gap_cube[pool]))])

    raw_action = np.asarray(A[i][K[i]:], dtype=np.float64)
    raw_state = np.asarray(S[i][K[i]:], dtype=np.float64)
    H_raw = len(raw_action)
    for tag, arr in (('action', raw_action), ('state', raw_state)):
        if not np.isfinite(arr).all():
            raise SystemExit('FAIL %s 里有非有限值' % tag)

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

    # 拉伸是线性插值：端点值恰好压在盒边界上时（例：rJ3 恰为 0）会带出 ~1e-10 的浮点残差。
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
    # 对多个 alpha 都试：运行时 PI0_ACTION_ALPHA 是可调的，任何一个值下都不许被切。
    for _alpha in (ACTION_FILTER_ALPHA, 1.0, 0.5, 0.1):
        _F, _C = real_pipeline(action, _alpha, JOINT_RATE, GRIPPER_RATE)
        _cut = float(np.abs(_F - _C).max())
        if _cut > 1e-12:
            raise SystemExit('FAIL alpha=%g 时限速器实切 %.6g（应 <1e-12）；轨迹没被压到帽内'
                             % (_alpha, _cut))
    F, C = real_pipeline(action, ACTION_FILTER_ALPHA, JOINT_RATE, GRIPPER_RATE)
    cut = float(np.abs(F - C).max())

    # --- 自检 2b：不许撞驱动护栏 verified_hardware.py:474/476 -------------------
    # delta > .25*dt + .01（关节）/ .006*dt + .001（夹爪）会抛错打断整跑。
    dt = 1.0 / FPS
    j_allow, g_allow = 0.25 * dt + 0.01, 0.006 * dt + 0.001
    dC = np.abs(np.diff(C, axis=0))
    j_max, g_max = float(dC[:, JOINTS].max()), float(dC[:, JAW].max())
    if j_max >= j_allow or g_max >= g_allow:
        raise SystemExit('FAIL 会撞驱动护栏：关节 %.5f/%.5f、夹爪 %.5f/%.5f'
                         % (j_max, j_allow, g_max, g_allow))
    driver_guard = {'joint_allow_rad_per_tick': j_allow,
                    'jaw_allow_m_per_tick': g_allow,
                    'joint_step_max_rad': j_max, 'jaw_step_max_m': g_max,
                    'joint_ratio': j_max / j_allow, 'jaw_ratio': g_max / g_allow}

    # --- 自检 3：爪合上的那一拍，手臂必须在演示的抓取位上 -------------------------
    ref_t = cross(raw_action[:, JAW], 0.06)
    if ref_t is None:
        raise SystemExit('FAIL 参照集里爪从没到过 0.06，这集不是一次抓取')
    grasp = raw_action[ref_t, 7:14]
    got_t = cross(C[:, JAW], 0.06)
    dev = float(np.abs(C[got_t, 7:14] - grasp).max()) if got_t is not None else float('inf')
    if got_t is None or dev > 0.02:
        raise SystemExit('FAIL 爪合上那拍手臂离抓取位 %.4f rad（>0.02）—— 时序没保住' % dev)

    # --- 前置"保持不动"行 --------------------------------------------------------
    lead = max(0, int(args.lead_hold))
    if lead:
        action = np.vstack([np.tile(action[0], (lead, 1)), action])
        state = np.vstack([np.tile(state[0], (lead, 1)), state])
    H = len(action)

    meta = {
        'replay_id': 'replay://cube_phase_today99/%s/h%d' % (NAME[i].replace('.parquet', ''), H),
        'kind': 'recorded_demo_replay_time_stretched',
        'model': None,
        'source_file': NAME[i],
        'source_episode_index': i,
        'source_episodes_total': n_total,
        'source_episodes_with_phase': len(S),
        'trigger_tick': int(K[i]),
        'trigger_jump_threshold': float(args.phase_jump),
        'horizon_raw': H_raw,
        'horizon_stretched_walk': H_walk,
        'horizon': H,
        'horizon_seconds': H / FPS,
        'lead_hold_rows': lead,
        'stretch_margin': float(args.stretch_margin),
        'max_step_ref_per_tick': float(args.max_step),
        'stretch_factor_walk': H_walk / H_raw,
        'stretch_factor_total': H / H_raw,
        'common_shortest_phase_ticks': shortest,
        'family_median_l_inf': float(median_l_inf[i]),
        'selection_pool': int(len(pool)),
        'start_state_l_inf_to_home_cube_arm': float(start_gap_cube[i]),
        'start_state_l_inf_to_home_all_dims': float(start_gap_all[i]),
        'end_state_l_inf_to_start_state': float(np.abs(state[-1] - state[0]).max()),
        'home_json': os.path.abspath(args.home_json),
        'home_target': home.tolist(),
        'hold_drawer_target': home[0:7].tolist(),
        'limits_max_excess_over_low_high': float(excess.max()),
        'limits_preclamp_residual': float(_pre.max()),
        'action_rates': rr,
        'action_rates_raw': rates(raw_action, JOINT_RATE, GRIPPER_RATE),
        'pipeline_filter_alpha': ACTION_FILTER_ALPHA,
        'pipeline_limiter_cut_max': cut,
        'driver_guard': driver_guard,
        'pipeline_jaw_close_tick': got_t,
        'pipeline_jaw_close_arm_dev_rad': dev,
        'reference_jaw_close_tick': ref_t,
        'cube_arm_J2_peak': float(action[:, 8].max()),
        'cube_arm_J3_min': float(action[:, 9].min()),
        'start_row': action[0].tolist(),
        'end_row': action[-1].tolist(),
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'axis_names': NAMES,
    }

    npz = args.out + '.npz'
    np.savez(npz, action=action, state=state,
             meta=np.array(json.dumps(meta, sort_keys=True)))
    with open(args.out + '.json', 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write('\n')

    print('选中 %s（第 %d 集）触发拍 %d' % (NAME[i], i, K[i]))
    print('   相位长度        %d 拍 = %.2f s' % (H_raw, H_raw / FPS))
    print('   协调拉伸后      %d 拍 = %.2f s（%.2f 倍）+ 前置 %d 拍保持不动 ⇒ 共 %d 拍 = %.2f s'
          % (H_walk, H_walk / FPS, H_walk / H_raw, lead, H, H / FPS))
    print('   离全集中位 L∞   = %.4f rad（全集最低 %.4f）' % (median_l_inf[i], median_l_inf.min()))
    print('   首行方块臂 7 维离起跑位姿 L∞ = %.4f rad   <-- 服务端 arming 判据卡这个'
          % meta['start_state_l_inf_to_home_cube_arm'])
    print('   首行全 14 维离起跑位姿 L∞    = %.4f rad（抽屉臂此刻还夹着抽屉拉环，所以大）'
          % meta['start_state_l_inf_to_home_all_dims'])
    print('   拉伸后关节单拍 max %.5f / 帽 %.5f rad（超限 %d 拍）'
          % (rr['joint_step_max_rad'], rr['joint_cap_rad_per_tick'], rr['joint_rows_over_cap']))
    print('   拉伸后夹爪单拍 max %.5f / 帽 %.5f m （超限 %d 拍）'
          % (rr['jaw_step_max_m'], rr['jaw_cap_m_per_tick'], rr['jaw_rows_over_cap']))
    print('   越出 LOW/HIGH 的行 %d 个（必须 0，否则闸门会触发重采样）'
          % int((excess > 0).sum()))
    print('   >> 真实客户端管线后，limited_target 实切量 = %.3g（必须 <1e-12；真被切是 1e-3 量级）' % cut)
    print('   >> 驱动护栏余量：关节 %.5f/%.5f（%.0f%%）、夹爪 %.5f/%.5f（%.0f%%）'
          % (driver_guard['joint_step_max_rad'], driver_guard['joint_allow_rad_per_tick'],
             100 * driver_guard['joint_ratio'], driver_guard['jaw_step_max_m'],
             driver_guard['jaw_allow_m_per_tick'], 100 * driver_guard['jaw_ratio']))
    print('   >> 爪合上(>=0.06)在第 %d 拍，那拍手臂离演示抓取位 %.4f rad（参照里是第 %d 拍）'
          % (got_t, dev, ref_t))
    print('   方块臂 rJ2 峰 %.4f；rJ3 最低 %.4f' % (meta['cube_arm_J2_peak'], meta['cube_arm_J3_min']))
    print('   写了 %s 和 %s' % (npz, args.out + '.json'))
    print('   起跑位姿来自 %s（启动器归位到这里）' % meta['home_json'])
    print('   启动器建议窗口 --rollout-seconds %d' % (int(np.ceil(H / FPS)) + 6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
