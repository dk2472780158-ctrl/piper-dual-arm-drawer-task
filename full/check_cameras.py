#!/usr/bin/env python3
"""相机闸门自检 —— 无真机：不连 CAN、不使能电机、不发任何动作。

复刻活客户端开跑前对相机做的三道判断，判据与超时都照抄：

  1) 别名绑定：``/dev/camera_<别名>`` 必须存在；它指向的那个 video 节点的 USB 父口，
     必须就是客户端写死的那个串号所在的 USB 口。
     （机械臂/相机断电重启会让 USB 重新枚举、节点号变化 —— 这一步就是抓"别名还指着
     老的、或者指错了相机"的。）
  2) 流没被占：客户端那套推流是一股一份；已经有别的消费端连着，就不能再插进去。
  3) 三路都出帧：按 ``tcp://127.0.0.1:<port>`` 开 MPEG-TS，帧必须是 480x640x3，
     三路都要在 max_age 内到齐、彼此相差不超过 0.25 s。

只开 TCP 流、读几帧、立刻全关：不碰相机设备本身、不碰 CAN、不碰电机。
相机表（名字/别名/串号/端口）从客户端源码里解析出来，**不在本文件里重复写一遍**，
所以这份文件里没有任何序列号。
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WANT_SHAPE = (480, 640, 3)
OPEN_TIMEOUT_MS = 5000          # 客户端 cv2.CAP_PROP_OPEN_TIMEOUT_MSEC
READ_TIMEOUT_MS = 1000          # 客户端 cv2.CAP_PROP_READ_TIMEOUT_MSEC
DEADLINE_S = 8.0                # 客户端 CameraStreams.__enter__ 的等待上限
MAX_AGE_S = .6                  # 客户端 camera_max_age_s 默认值
SPREAD_S = .25                  # 客户端 snapshot 的帧间一致性上限
FRAMES_WANT = 20                # 每路至少读到这么多帧才算"真的在出帧"


def cameras_from(client: Path):
    """从客户端源码解析相机表：名字 -> (别名, 串号, 端口)。"""
    tree = ast.parse(client.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'CAMERAS' for t in node.targets):
            table = ast.literal_eval(node.value)
            return {str(k): tuple(v) for k, v in table.items()}
    raise SystemExit('FAIL 客户端源码里没有 CAMERAS 表（相机表结构变了，先人工核对）')


def usb_parent(path) -> str:
    """客户端同款：把 sysfs 路径收敛成 USB 物理口（去掉接口后缀）。"""
    found = None
    for part in Path(path).parts:
        if re.fullmatch(r'\d+-\d+(?:\.\d+)*', part):
            found = part
    if found is None:
        raise RuntimeError(f'路径里没有 USB 父口: {path}')
    return found


def check_binding(cameras):
    """闸门 1：别名 -> video 节点 -> USB 父口，必须与串号所在的 USB 父口一致。"""
    import pyrealsense2 as rs
    by_serial = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        by_serial[serial] = usb_parent(device.get_info(rs.camera_info.physical_port))
    rows, bad = [], []
    for name, (alias, serial, _port) in cameras.items():
        link = Path('/dev/camera_' + alias)
        try:
            node = link.resolve(strict=True)
        except OSError:
            bad.append(f'{name}: 别名 /dev/camera_{alias} 不存在（udev 规则没生效？）')
            continue
        usb = usb_parent((Path('/sys/class/video4linux') / node.name / 'device').resolve(strict=True))
        want = by_serial.get(serial)
        if want != usb:
            bad.append(f'{name}: 别名指向 {node.name}（USB {usb}），'
                       f'但串号 {serial} 在 USB {want} —— 差点开错相机')
        rows.append((name, alias, node.name, usb, serial))
    return rows, bad


def check_free(cameras):
    """闸门 2：三个端口上没有别的消费端（一股一份）。"""
    done = subprocess.run(['ss', '-tn', 'state', 'established'],
                          capture_output=True, text=True, timeout=3)
    busy = []
    for _name, (_alias, _serial, port) in cameras.items():
        if re.search(rf':{port}(?:\s|$)', done.stdout):
            busy.append(port)
    return busy


def check_frames(cameras, frames_want):
    """闸门 3：三路都出帧、帧尺寸对、够新、彼此同步。"""
    import cv2
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    caps, err, got, luma, first, last = {}, {}, {}, {}, {}, {}
    for name, (_alias, _serial, port) in cameras.items():
        caps[name] = cv2.VideoCapture(f'tcp://127.0.0.1:{port}', cv2.CAP_FFMPEG, params)
        err[name], got[name], luma[name] = None, 0, 0.0
    deadline = time.monotonic() + DEADLINE_S
    try:
        while time.monotonic() < deadline:
            for name, cap in caps.items():
                if err[name] or got[name] >= frames_want:
                    continue
                if not cap.isOpened():
                    err[name] = 'Cannot open MPEG-TS stream'
                    continue
                ok, frame = cap.read()
                if not ok or frame is None or frame.shape != WANT_SHAPE:
                    err[name] = 'Missing/invalid camera frame'
                    continue
                now = time.monotonic()
                got[name] += 1
                luma[name] = float(np.asarray(frame, dtype=np.float64)[:, :, ::-1].mean())
                first.setdefault(name, now)
                last[name] = now
            if all(got[name] >= frames_want for name in caps):
                break
    finally:
        for cap in caps.values():
            cap.release()
    now = time.monotonic()
    rows = []
    for name, (_alias, _serial, port) in cameras.items():
        span = last.get(name, now) - first.get(name, now) if name in last else 0.0
        rows.append({'name': name, 'port': port, 'frames': got[name],
                     'fps': (got[name] - 1) / span if span > 1e-6 else 0.0,
                     'luma': luma[name], 'age': now - last[name] if name in last else float('inf'),
                     'err': err[name]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--client', required=True, help='活客户端源码（只读它解析相机表）')
    ap.add_argument('--frames', type=int, default=FRAMES_WANT)
    ap.add_argument('--max-age', type=float, default=MAX_AGE_S)
    args = ap.parse_args()

    client = Path(args.client)
    cameras = cameras_from(client)
    print(f'=== 相机闸门自检（无真机：不碰 CAN、不使能电机、不发动作）  共 {len(cameras)} 路')

    print('[1/3] 别名绑定（别名 -> video 节点 -> USB 父口 必须与串号对得上）')
    rows, bad = check_binding(cameras)
    for name, alias, node, usb, serial in rows:
        print(f'    {name:<10} /dev/camera_{alias} -> {node}   USB {usb}   串号 {serial}   OK')
    if bad:
        for line in bad:
            print('    FAIL ' + line)
        print('PI0_CAM_CHECK FAIL 别名绑定不对。断电重启机械臂会让 USB 重枚举、节点号变化；'
              '先重建别名再验：sudo udevadm trigger / 看 udev 规则')
        return 1

    print('[2/3] 端口没有被别的消费端占住（推流一股一份）')
    busy = check_free(cameras)
    if busy:
        print(f'    FAIL 这些端口上已经有别的消费端：{busy}')
        print('PI0_CAM_CHECK FAIL 先关掉那个预览/客户端，或者让推流服务重起再验')
        return 1
    print('    ' + ' / '.join(str(p) for _n, (_a, _s, p) in cameras.items()) + '  都空着   OK')

    print(f'[3/3] 三路出帧（每路 {args.frames} 帧、尺寸 {WANT_SHAPE}、帧龄 < {args.max_age}s、'
          f'彼此相差 < {SPREAD_S}s）')
    rows = check_frames(cameras, args.frames)
    failed = []
    for row in rows:
        if row['err']:
            print(f"    {row['name']:<10} port={row['port']}  FAIL {row['err']}")
            failed.append(row['name'])
            continue
        print(f"    {row['name']:<10} port={row['port']}  帧={row['frames']:<3} "
              f"≈{row['fps']:.1f} fps  亮度={row['luma']:.1f}  帧龄={row['age']:.2f}s  OK")
        if row['age'] > args.max_age:
            failed.append(row['name'])
            print(f"        ↑ 帧太旧（> {args.max_age}s）")
    if failed:
        print(f'PI0_CAM_CHECK FAIL 这些路没在出帧：{failed}')
        print('    多半是推流服务还在按"老的 video 节点"读（断电重启后就死了）。')
        print('    修法：bash /opt/lab-edge/bin/piper_idle_fanout.sh stop && '
              'PIPER_FANOUT_PYTHON=<原来那个 python> bash /opt/lab-edge/bin/piper_idle_fanout.sh start')
        return 1

    ages = [row['age'] for row in rows]
    spread = max(ages) - min(ages)
    if spread > SPREAD_S:
        print(f'PI0_CAM_CHECK FAIL 三路不同步：帧龄彼此相差 {spread:.2f}s > {SPREAD_S}s')
        return 1
    print(f'    三路帧龄相差 {spread:.3f}s   OK')

    # 自己断开后推流端必须把连接收回去；不然紧跟着的客户端会撞上
    # "Single-client camera streams already in use" 而 abort。
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not check_free(cameras):
            break
        time.sleep(.1)
    left = check_free(cameras)
    if left:
        print(f'PI0_CAM_CHECK FAIL 我断开之后推流端还占着这些端口：{left}')
        print('    紧跟着的客户端会被"流已被占用"挡住；先重起推流服务再来。')
        return 1
    print('    断开后端口立刻空出来（接得住紧跟着的客户端）   OK')
    print('PI0_CAM_CHECK OK 三路相机现在都能出帧；这一步过了再往下走（真机另说）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
