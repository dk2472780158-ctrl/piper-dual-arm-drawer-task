#!/usr/bin/env python3
"""姿态闸门 —— 无真机：只被动读一次反馈，判断**客户端使能之前那道 preload 闸门**会不会挡。

它做两件事，两件都不动硬件：

  1) 跑官方那段只读代码读一次稳定位姿 —— 官方 `reset_piper_pose.py` **不加 --execute**
     ⇒ 内部 `read_only=not args.execute` ⇒ PIPERDual 只收反馈、不使能、不发帧，
     读到 Current 后打印 "DRY-RUN only." 就返回。它自带 ±3.20 恢复限位。
  2) 把读到的 14 维位姿喂给**活客户端那一份** `measured_preload()`（直接按路径 import
     活文件，阈值不是我抄的副本），看它会不会 raise。

为什么不能只看官方脚本那句 nominal 警告：两个盒子**不是同一个**。官方脚本的 nominal 是
SDK 名义盒（J2 ≥ -0.05、J3 ≤ 0.06、夹爪 ≤ 0.08），而客户端 preload 用的是策略动作盒
（J2 ≥ 0、J3 ≤ 0、夹爪 ≤ 0.07，关节容差 0.01 rad、夹爪 0.001 m）。所以会出**官方说没事、
客户端照样 abort** 的组合（例如 J2 = -0.005 或夹爪 0.073）。这就是本文件存在的理由。

不碰 CAN 控制帧、不使能电机、不开相机、不移动任何关节。
退出码：0 = 客户端那道闸门会放行；1 = 会 abort；2 = 读不到（别据此判断）。
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

CONDA_SH = Path('/home/databall/miniconda3/etc/profile.d/conda.sh')
READ_TIMEOUT_S = 120.0


def parse_current(text: str):
    """从官方脚本的输出里抠出 Current 那一行 14 个数（numpy 会折行，所以按到 ']' 为止取）。"""
    m = re.search(r'^Current:\s*\[(.*?)\]', text, re.S | re.M)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).replace('\n', ' ').split(',') if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) != 14:
        return None
    return np.asarray(values, dtype=np.float64)


def read_pose(reset_dir: Path) -> str:
    """跑官方只读路径，返回它的全部输出。stdin 给 DEVNULL：只读路径本来就不读键盘，
    万一它某天开始读，EOF 也只会让它退出，绝不会变成"确认"。"""
    script = reset_dir / 'reset_piper_pose.py'
    inner = (
        'set -e\n'
        'if [ -f "$1" ]; then . "$1"; conda activate lerobot_v30; fi\n'
        'cd "$2"\n'
        'exec python -B "$3" --arm both --left-can can1 --right-can can0\n'
    )
    cmd = ['bash', '-c', inner, 'pose-check',
           str(CONDA_SH), str(reset_dir), str(script)]
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=READ_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        got = (exc.output or b'').decode('utf-8', 'replace')
        return got + f'\n[check_pose] 超过 {READ_TIMEOUT_S:.0f} 秒没读出来，已放弃\n'
    return done.stdout.decode('utf-8', 'replace')


def load_live_gate(hw: Path):
    """按路径加载活客户端那一份 verified_hardware（它顶层只 import 标准库 + numpy，
    SDK 只在 home() 里实例化），拿它的常量与 measured_preload 用。"""
    spec = importlib.util.spec_from_file_location('vh_live_pose_gate', str(hw))
    if spec is None or spec.loader is None:
        raise SystemExit(f'FAIL 加载不了 {hw}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('measured_preload', 'LOWER', 'UPPER', 'JOINTS', 'GRIPPERS'):
        if not hasattr(module, name):
            raise SystemExit(f'FAIL {hw} 里没有 {name}（活文件结构变了，先人工核对）')
    return module


def source_tolerances(hw: Path):
    """只为了在报告里写出容差数值（判决仍然由 measured_preload 自己下）。"""
    text = hw.read_text(encoding='utf-8')
    out = {}
    for key, pattern in (('joint', r'difference\[JOINTS\]\)\s*>\s*([0-9.]+)'),
                         ('gripper', r'difference\[GRIPPERS\]\)\s*>\s*([0-9.]+)')):
        m = re.search(pattern, text)
        if m:
            out[key] = m.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hw', required=True,
                    help='活的 verified_hardware.py（拿它的 measured_preload 与盒子常量）')
    ap.add_argument('--reset-dir', required=True, help='官方 reset_piper_pose.py 所在目录')
    args = ap.parse_args()

    hw = Path(args.hw)
    reset_dir = Path(args.reset_dir)
    for path in (hw, reset_dir / 'reset_piper_pose.py'):
        if not path.is_file():
            print(f'FAIL 缺文件 {path}')
            return 2

    print('[1/3] 官方只读路径读一次反馈（read_only、不使能、不移动、不发帧）...')
    text = read_pose(reset_dir)
    print('------- 官方输出开始（逐字照抄）')
    print(text.rstrip('\n'))
    print('------- 官方输出结束')
    if 'DRY-RUN only' not in text:
        if 'FATAL' in text and 'current feedback' in text:
            print('PI0_POSE_CHECK OUT-OF-RANGE —— 实测位姿出界到连官方的 ±3.20 恢复限位都'
                  '兜不住（官方脚本自己就报 FATAL 了）。客户端必然在使能前 abort。')
            return 1
        print('PI0_POSE_CHECK UNKNOWN —— 官方只读路径没跑出结果（上面那份输出里没有 '
              '"DRY-RUN only"）。别据此判断；先看上面报了什么。')
        return 2

    pose = parse_current(text)
    if pose is None:
        print('PI0_POSE_CHECK UNKNOWN —— 抠不出 Current 那 14 个数（官方输出格式变了）。')
        return 2

    print('[2/3] 拿实测位姿喂活客户端的 measured_preload（阈值现场从活文件读，不是副本）')
    gate = load_live_gate(hw)
    lower, upper = np.asarray(gate.LOWER), np.asarray(gate.UPPER)
    joints, grippers = np.asarray(gate.JOINTS), np.asarray(gate.GRIPPERS)
    excess = np.abs(np.clip(pose, lower, upper) - pose)
    tol = source_tolerances(hw)

    print('      实测位姿 = [' + ', '.join(f'{v:.6f}' for v in pose) + ']')
    off = np.flatnonzero(excess > 0.0)
    if not off.size:
        print('      14 维全在盒内（越界量 0）')
    else:
        print('      越界的维度（越界量 = 实测被 clip 掉多少）：')
        for i in off:
            kind = '关节' if i in joints else '夹爪'
            limit = f'关节 {tol["joint"]}' if i in joints else f'夹爪 {tol["gripper"]}'
            tag = '关节' if i in joints else '夹爪'
            print(f'        dim{i} {tag} 实测 {pose[i]:+.6f}，盒子 '
                  f'[{lower[i]:+.6f}, {upper[i]:+.6f}]，越界 {excess[i]:.6f}'
                  f'（{limit} 以内算边界噪声，放行）')

    print('[3/3] 判决')
    try:
        gate.measured_preload(pose)
    except ValueError as exc:
        print(f'PI0_POSE_CHECK OUT-OF-RANGE —— {exc}')
        print('      客户端会在使能之前 abort，一行动作都发不出去。')
        print('      修法：bash /home/databall/reset_piper.sh（官方归位，真动），')
        print('      或者直接 bash run_full.sh motion —— 真机模式默认会先自动归位。')
        return 1
    worst_j = float(np.max(excess[joints])) if joints.size else 0.0
    worst_g = float(np.max(excess[grippers])) if grippers.size else 0.0
    print(f'PI0_POSE_CHECK OK 实测位姿会被接受（最大越界：关节 {worst_j:.6f} rad / '
          f'夹爪 {worst_g:.6f} m，都在容差内）⇒ 客户端那道 preload 闸门会放行。')
    if off.size:
        print('      注意：上面那些维度**已经被算过**了 —— 容差之内的边界噪声客户端会自己 '
              'clip 回去，不算越界；但手臂已经在限位上坐着了，别再让它往那个方向下垂。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
