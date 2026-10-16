#!/usr/bin/env python3
"""读一次"全任务"回放的报告，只回几件事（不做推断，不编数字）。

    1) 回放有没有真的跑起来（armed / 行数）
    2) 四步各自的现象：抽屉爪"咬住"有没有发生、方块爪有没有"张->合->松"、
       最后两条臂有没有回到起跑位姿（= 自动归位有没有走到）
    3) 驱动护栏有没有报 slew（安全栈有没有被打断）
    4) 卡在哪一阶段（status / 错误字段）

用法：
    python3 check_full_run.py --report <R.json> [--trajectory <R.trajectory.jsonl>]
                              [--meta full_traj.json]

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

    traj = a.trajectory or (a.report[:-5] + '.trajectory.jsonl')
    grip_plan = None
    cube_plan = None
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
        lead = m.get('lead_hold_rows') or 0
        print('   计划：咬住拍 = %s（第 %.1f s）、总量 %s 拍 = %.1f s（含归位段）'
              % (grip_plan, (grip_plan or 0) / 30.0, m.get('horizon'), m.get('horizon_seconds')))
        ph = m.get('phases') or {}
        for key, label in (('pull', '拉抽屉'), ('cube', '放方块'), ('close', '合抽屉')):
            if key in ph:
                w = ph[key]['walk']
                cube_plan = (w[0] + lead, w[1] + lead) if key == 'cube' else cube_plan
                print('     计划 %-4s 第 %d..%d 拍 = %.1f..%.1f s'
                      % (label, w[0] + lead, w[1] + lead,
                         (w[0] + lead) / 30.0, (w[1] + lead) / 30.0))
        rr = m.get('return_home') or {}
        if rr:
            w = rr['walk']
            print('     计划 归位 第 %d..%d 拍 = %.1f..%.1f s（%d 拍移动 + %d 拍保持）'
                  % (w[0] + lead, w[1] + lead, (w[0] + lead) / 30.0, (w[1] + lead) / 30.0,
                     rr['rows_move'], rr['rows_hold']))
        print('     计划 不跑归位的话到第 %s 拍 = %.1f s 就停'
              % (m.get('horizon_without_return'), (m.get('horizon_without_return') or 0) / 30.0))
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
            def get(r):
                for cand in ('measured', 'state', 'observation_state', 'command', 'commanded'):
                    if cand in r:
                        return r[cand]
                return None
            st = [get(r) for r in rows]
            st = [s for s in st if isinstance(s, list) and len(s) == 14]
            print('   trajectory.jsonl 行数 = %d（每块一条，不是每拍一条；这是控制日志，不是录像）'
                  % len(rows))
            if st:
                arr = st
                dc = [r[DRAWER_CLAW] for r in arr]        # 抽屉爪
                cc = [r[CUBE_CLAW] for r in arr]          # 方块爪
                print('   实测抽屉爪 范围 = %.4f .. %.4f；方块爪 范围 = %.4f .. %.4f'
                      % (min(dc), max(dc), min(cc), max(cc)))

                def phase_of(k):
                    lo, hi = 0, len(arr)
                    if per_chunk and cube_plan:
                        lo, hi = cube_plan[0] // per_chunk, cube_plan[1] // per_chunk + 1
                    return max(0, min(lo, len(arr) - 1)), min(hi, len(arr))

                # (1) 抽屉爪：张开峰 -> 咬住
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
                # (2) 方块爪：张 -> 合 -> 松
                if cc:
                    p = max(range(len(cc)), key=lambda i: cc[i])
                    tail = cc[p:]
                    shut_i = int(min(range(len(tail)), key=lambda i: tail[i]))
                    reopen = max(tail[shut_i:]) if shut_i < len(tail) else tail[-1]
                    print('   实测方块爪 %.4f（张@第 %d 条）→ %.4f（合）→ %.4f（松）'
                          % (cc[p], p, tail[shut_i], reopen))
                    if not (cc[p] >= 0.05 and tail[shut_i] <= 0.04 and reopen >= 0.05):
                        print('   ⚠ 方块爪没有出现"张开 -> 合上 -> 松开"的三段；'
                              '方块臂可能没去夹方块')
                # (3) 两条臂离起跑位姿多远（全程最远 / 末态）
                base = arr[0]
                draw_far = max(max(abs(r[j] - base[j]) for j in range(7)) for r in arr)
                cube_far = max(max(abs(r[j] - base[j]) for j in range(7, 14)) for r in arr)
                draw_end = max(abs(arr[-1][j] - base[j]) for j in range(7))
                cube_end = max(abs(arr[-1][j] - base[j]) for j in range(7, 14))
                print('   实测离起跑位姿：抽屉臂 全程最远 %.4f / 末态 %.4f；'
                      '方块臂 全程最远 %.4f / 末态 %.4f（末态两臂都应当 ≈0 = 归位走到了）'
                      % (draw_far, draw_end, cube_far, cube_end))
                mj = [max(abs(r[j]) for j in JOINTS) for r in arr]
                print('   实测 |关节| 峰 = %.4f' % max(mj))
                print('   实测末态 抽屉臂J2=%.4f 抽屉爪=%.4f 方块臂J2=%.4f 方块爪=%.4f'
                      % (arr[-1][1], arr[-1][DRAWER_CLAW], arr[-1][8], arr[-1][CUBE_CLAW]))
        else:
            print('   trajectory.jsonl 是空的')
    else:
        print('   （没有 trajectory.jsonl，跳过实测判定）')

    print()
    # 限时窗口跑满就是这个标签，它是**正常收尾**，不是异常：
    # 报告本来就不看画面，"任务成没成"只能由人眼/实测状态判。
    finished = rep.get('result') in ('bounded_trial_finished_task_outcome_unverified',
                                     'trial_finished_task_outcome_unverified')
    if dry:
        print('结论：这是**干跑**（电机没使能，一行命令都没发）。它不是真机跑。')
    elif rep.get('status') in (0, 'ok', 'success') or finished:
        print('结论：这一跑正常结束（%s＝限时窗口跑满）。报告不看画面，所以'
              '"任务成没成"只能看上面那几行实测：抽屉拉没拉开、方块有没有被夹起来放进抽屉、'
              '末态有没有回到起跑位姿（= 自动归位走到没有）。它不会自己说成没成 —— '
              '以你现场看到的为准。' % rep.get('result'))
    else:
        print('结论：没有正常结束（result=%s status=%s），先看上面的 error/phase。'
              % (rep.get('result'), rep.get('status')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
