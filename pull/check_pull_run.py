#!/usr/bin/env python3
"""读一次"拉抽屉"回放的报告，只回四件事（不做推断，不编数字）。

    1) 回放有没有真的跑起来（armed / 行数）
    2) 抽屉爪"咬住"那一刻，抽屉臂到没到演示的咬住位（用 trajectory.jsonl 的实测 state 对）
    3) 驱动护栏有没有报 slew（安全栈有没有被打断）
    4) 卡在哪一阶段（status / 错误字段）
    另外打印两条臂各自的实测首末位移：这一段应当是**抽屉臂大幅动、方块臂 ≈0**。

用法：
    python3 check_pull_run.py --report <R.json> [--trajectory <R.trajectory.jsonl>]
                              [--meta pull_traj.json]

只读；不写任何文件。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
DRAWER_CLAW = 6
CUBE_CLAW = 13
GRIP_SHUT = 0.01


def err(msg):
    print('FAIL ' + msg)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--trajectory', default='')
    ap.add_argument('--meta', default='')
    a = ap.parse_args()
    if not os.path.isfile(a.report):
        return err('没有报告 %s' % a.report)
    rep = json.load(open(a.report))

    print('=== 报告 %s' % a.report)
    print('   result        = %s' % rep.get('result'))
    print('   status        = %s' % rep.get('status'))
    for k in ('phase', 'stage', 'error', 'message', 'reason', 'traceback'):
        if rep.get(k):
            print('   %-13s = %s' % (k, str(rep[k])[:240].replace('\n', ' | ')))
    dry = rep.get('result') == 'dryrun_passed_no_commands'

    ro = rep.get('rollout') or {}
    print('   chunks        = %s' % ro.get('chunks'))
    print('   action_rows   = %s' % ro.get('action_rows_sent'))
    print('   hold_ticks    = %s（>0 = 服务端供块断过，手臂在那几拍被冻住）' % ro.get('hold_ticks'))
    print('   underrun      = %s' % (ro.get('underrun') is not None))
    print('   max_joint_track_err_rad = %s' % ro.get('max_joint_tracking_error_rad'))
    if rep.get('gripper_values_clamped'):
        print('   gripper_clamped = %s' % rep['gripper_values_clamped'])
    if rep.get('rejected_policy_chunks_total'):
        print('   rejected_chunks = %s  ← 闸门拒绝过（回放里不该出现）'
              % rep['rejected_policy_chunks_total'])
    if rep.get('commands_sent') is not None:
        print('   commands_sent = %s' % rep['commands_sent'])
    if rep.get('home_source'):
        print('   home_source   = %s' % str(rep['home_source'])[:200])
    if rep.get('home'):
        h = rep['home']
        if isinstance(h, dict):
            for k in ('max_joint_error_rad', 'max_gripper_error_m', 'error', 'ok'):
                if k in h:
                    print('   复位 result   %-20s = %s' % (k, h[k]))

    ar = rep.get('action_rows_sent')
    if ar is None:                     # 报告里它挂在 rollout 下，不在顶层
        ar = ro.get('action_rows_sent')
    if ar is not None and not ar and not dry:
        print('   ⚠ 一行都没发出去 —— 回放没跑（看服务端日志的 REPLAY_ARMED / REPLAY_NOT_ARMED）')

    # 咬住判定：拿 trajectory.jsonl 的实测 state
    traj = a.trajectory or (a.report[:-5] + '.trajectory.jsonl')
    grip_plan = None
    # trajectory.jsonl 是**每块一条**（每 10 拍一条），不是每拍一条。
    # 拿记录序号直接跟"第几拍"比是量纲错配，必须先乘回块长。
    per_chunk = None
    if ar and ro.get('chunks'):
        per_chunk = max(1, int(round(float(ar) / float(ro['chunks']))))
        print('   记录密度      = 每 %s 拍一条（%s 条 = %s 拍）'
              % (per_chunk, ro.get('chunks'), ar))
    if a.meta and os.path.isfile(a.meta):
        m = json.load(open(a.meta))
        grip_plan = m.get('pipeline_grip_tick')
        print('   计划：咬住拍 = %s（第 %.1f s）、总量 %s 拍 = %.1f s'
              % (grip_plan, (grip_plan or 0) / 30.0, m.get('horizon'), m.get('horizon_seconds')))
    if os.path.isfile(traj):
        rows = []
        with open(traj) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if rows:
            def get(r, key):
                for cand in ('measured', 'state', 'observation_state', 'command', 'commanded'):
                    if cand in r:
                        return r[cand]
                return None
            st = [get(r, 'measured') for r in rows]
            st = [s for s in st if isinstance(s, list) and len(s) == 14]
            print('   trajectory.jsonl 行数 = %d（每块一条，不是每拍一条；这是控制日志，不是录像）'
                  % len(rows))
            if st:
                arr = st
                dc = [r[DRAWER_CLAW] for r in arr]        # 抽屉爪
                cc = [r[CUBE_CLAW] for r in arr]          # 方块爪
                print('   实测抽屉爪 范围 = %.4f .. %.4f；方块爪 范围 = %.4f .. %.4f'
                      % (min(dc), max(dc), min(cc), max(cc)))
                peak = max(range(len(dc)), key=lambda i: dc[i])
                after = [i for i in range(peak, len(dc)) if dc[i] <= GRIP_SHUT]
                if not after:
                    print('   ⚠ 实测抽屉爪张到 %.4f 之后再没合到 %.3f —— 没有咬住动作'
                          % (dc[peak], GRIP_SHUT))
                else:
                    got = after[0]
                    print('   实测抽屉爪张开峰在第 %d 条记录（%.4f）→ 咬住在第 %d 条（%.4f）'
                          % (peak, dc[peak], got, dc[got]))
                    if per_chunk:
                        lo, hi = got * per_chunk, (got + 1) * per_chunk
                        print('     ⇒ 约第 %d–%d 拍（分辨率 %d 拍：记录是块采样，不是逐拍）'
                              % (lo, hi, per_chunk))
                        if grip_plan is not None:
                            d = max(0, lo - grip_plan) if lo > grip_plan else max(0, grip_plan - hi)
                            if d <= per_chunk:
                                print('     计划里是第 %d 拍 ⇒ 同拍级（受 ±%d 拍分辨率所限）'
                                      % (grip_plan, per_chunk))
                            else:
                                print('     计划里是第 %d 拍 ⇒ 差约 %d 拍（%.2f s）；'
                                      '块记录只能给到这个精度'
                                      % (grip_plan, d, d / 30.0))
                mj = [max(abs(r[j]) for j in JOINTS) for r in arr]
                print('   实测 |关节| 峰 = %.4f' % max(mj))
                # 两条臂分开看首末位移：这一段应当**抽屉臂大动、方块臂 ≈0**
                draw = max(abs(arr[-1][j] - arr[0][j]) for j in range(7))
                cube = max(abs(arr[-1][j] - arr[0][j]) for j in range(7, 14))
                print('   实测首末位移 抽屉臂 7 维 L∞ = %.4f（这一段它就是主角，应当明显 >0）'
                      '；方块臂 7 维 L∞ = %.4f（应当 ≈0）' % (draw, cube))
                print('   实测末态 抽屉臂J2=%.4f 抽屉爪=%.4f 方块臂J2=%.4f 方块爪=%.4f'
                      % (arr[-1][1], arr[-1][DRAWER_CLAW], arr[-1][8], arr[-1][CUBE_CLAW]))
        else:
            print('   trajectory.jsonl 是空的')
    else:
        print('   （没有 trajectory.jsonl，跳过咬住的实测判定）')

    print()
    # 限时窗口跑满就是这个标签，它是**正常收尾**，不是异常：
    # 报告本来就不看画面，"任务成没成"只能由人眼/实测状态判。
    finished = rep.get('result') in ('bounded_trial_finished_task_outcome_unverified',
                                     'trial_finished_task_outcome_unverified')
    if dry:
        print('结论：这是**干跑**（电机没使能，一行命令都没发）。它不是真机跑。')
    elif rep.get('status') in (0, 'ok', 'success') or finished:
        print('结论：这一跑正常结束（%s＝限时窗口跑满；报告不看画面，'
              '成没成看上面对实测爪/位姿那几行 —— 而这一段的判据只有一条：'
              '抽屉有没有被拉开）。' % rep.get('result'))
    else:
        print('结论：没有正常结束（result=%s status=%s），先看上面的 error/phase。'
              % (rep.get('result'), rep.get('status')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
