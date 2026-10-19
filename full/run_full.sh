#!/usr/bin/env bash
# 全任务回放启动器 -- 用**时间重参数化过的录制轨迹**顶替远端推理服务。
#
# 目的：把演示里完整的一套动作连起来跑一遍，顺序就是用户要的那四步：
#     1) 抽屉臂伸手 -> 张爪 -> 咬住抽屉正面左上那个黑色拉环 -> 把抽屉拉开约 11 cm -> 松爪
#     2) 方块臂伸过去夹起蓝方块、移到抽屉口、松爪放下
#     3) 抽屉臂再次前伸/退回（爪全程张着）= 把抽屉合上
#     4) **自动归位**：两条臂平滑回到复位目标，再保持 1 s
# 那一段全部来自 today99 / episode_000041 第 48..750 拍；只有第 4 步是补的
# （语料里演示没回家，见 make_full_traj.py）。不是策略、不是模型。
#
# 与 cubephase 那份的区别：那份把抽屉臂钉在归位点上（它要动方块臂）；这份两条臂
# 都按轨迹走。**客户端一个字节都不改**：TRIAL 指向本包 shared/ 里那份原版客户端（与活文件逐字节相同），
# 全部闸门、限速、跟踪校验、归位、报告、trace 都还是原版；只把启动位姿指到本目录的
# home_target_full.json（= 演示起跑那一拍手臂真正在的地方 = 归位段的终点）。
#
# 三条硬约束（脚本会自己挡住）：
#   1) 关节/夹爪速率旋钮必须等于轨迹构建时的值。
#   2) arming 容差必须比客户端 settle_home 的 0.035 rad 松，又不至于把别的位姿当起跑位姿。
#   3) 端口上若已有监听者，只有确认是**本脚本起的那个 replay 服务**才复用。
#
# 五种模式：
#   check    只跑自检（产物/轨迹/旋钮/服务/链路），到第 [4/5] 步就停，**不连机械臂、不开相机**。
#   cam      无真机相机闸门：只看三路相机现在能不能出帧（= 客户端开跑前那三道判断），
#            不连机械臂、不使能电机、不发任何动作；到这一步就停。
#   pose     无真机姿态闸门：用官方 reset_piper_pose.py 的**只读**模式被动读一次反馈，
#            看实测位姿有没有掉出 SDK 名义盒（不使能、不移动）；到这一步就停。
#   dry-run  再往前走一步：让客户端在"电机不使能"下问一次（一行命令都不发出去）。
#   motion   真机：**先跑官方归位**（PI0_RESET，默认 1）-> 复位到起跑位姿 -> 按回放跑
#            -> 跑完自动归位。由你自己按。
#
# dry-run / motion 都会在开跑前先过一遍相机闸门：相机坏的话 2 秒就报出来，
# 而不是等客户端跑到第 8 秒才 blocked_or_aborted（2026-10-16 就吃过这个亏）。
#
# 为什么要官方归位：客户端在使能之前有一道 measured_preload 闸门 —— 实测位姿被 clip
# 到 SDK 名义盒，任何关节被削掉 > 0.01 rad 就 abort（Raw pose outside preload
# tolerance，0 命令）。断电重启/久放之后手臂会下垂到限位外面（J2 下界 0、J3 上界 0
# 正好是它平时坐的位置），这道闸门就会挡。官方那个 reset 自带恢复限位 ±3.20，
# 只朝目标单向把越界关节收回来，正是对症的。PI0_RESET=0 跳过，auto=只在不越界时跳过。
set -uo pipefail

PORT=${PI0_PORT:-18315}
ARMTOL=${PI0_ARM_TOL:-0.10}
LDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 本包内的共享件（与活文件逐字节相同的副本）
CPHOME=$LDIR/../shared/home_target_cubephase.json                 # 只读它来打印"复位要跨多少"
# 这几条是"机器上的位置"。卡片终端（平台容器）里看不到那棵树，由入口层 pi0_env.sh
# 用下面这些 PI0_* 覆盖成容器里的真身 —— 所以没有一条必须存在。
LROOT=${PI0_DEPLOY_ROOT:-/home/databall/pi0_drawer_mixed_74_v1}
PY=${PI0_PY:-/home/databall/pi0_infer_3090/.venv/bin/python}
SPY=${PI0_REPLAY_PY:-/home/databall/miniconda3/envs/lerobot_v30/bin/python}
TRIAL=$LDIR/../shared/pi0_real_trial_v8.py
LINKCHK=$LDIR/../shared/check_pi0_link_v8.py
CAMCHK=$LDIR/check_cameras.py       # 无真机相机闸门（客户端开跑前那三道判断的复刻）
CAMSRC=$LDIR/../shared/pi0_rollout_v8.py     # 只读它解析相机表（别名/串号/端口），不执行
POSECHK=$LDIR/check_pose.py         # 无真机姿态闸门（调官方只读路径 + 活客户端那把闸门函数）
HW=$LDIR/../shared/verified_hardware.py   # 只读它拿 measured_preload 与盒子常量
# 官方归位（上游既有件，本脚本只调用它的两个入口，不复制、不改）
RESETSH=${PI0_RESET_SH:-/home/databall/reset_piper.sh}                     # 一键：立即使能 + 带两臂回训练起始位姿
RESETDIR=${PI0_RESET_DIR:-/home/databall/VLA/lerobot_piper}                  # 只读跑一次姿态检查，用它自己的解释器环境
HOME_JSON=$LDIR/home_target_full.json
SERVER=$LDIR/replay_server.py
BUILDER=$LDIR/make_full_traj.py
TRAJ=$LDIR/full_traj
SERVLOG=$LDIR/server_${PORT}.log
LOCK=${PI0_LOCK:-/home/databall/pico_piper_teleop/runtime/hardware.lock}

# 客户端读的环境变量：名字是客户端定死的，改不了（改了它就退回默认值）。
# 这里显式 export，好让下面第 [2/5] 步能核对运行旋钮与轨迹构建值一致。
export PI0_HOME_JSON=$HOME_JSON
export PI0_PRELIFT_RAD=0
export PI0_GRIPPER_RATE=${PI0_GRIPPER_RATE:-0.030}
export PI0_JOINT_RATE=${PI0_JOINT_RATE:-0.40}
export PI0_ACTION_ALPHA=${PI0_ACTION_ALPHA:-0.25}
export PI0_FILTER_GAP_S=${PI0_FILTER_GAP_S:-1.0}
export PI0_IMAGE_ORDER=${PI0_IMAGE_ORDER:-rgb}

# 本脚本自己的旋钮（都可以用环境变量改）：
#   PI0_PORT=18314      换端口            PI0_ARM_TOL=0.10    arming 容差
#   PI0_REBUILD=1       强制重建轨迹      PI0_DATA=<路径>     换语料根
#   PI0_COUNTDOWN=5     开跑前倒数 5 秒   PI0_TRACE=0         不写 trace
#   PI0_REUSE_SERVER=1  真机也复用已在跑的服务（默认不复用：真机必须从第 0 拍放）
#   PI0_RESET=0|auto    开跑前官方归位：默认 1（每次都归）、0 跳过、auto 只在位姿越界时归

MODE=${1:-dry-run}
WINDOW=${2:-auto}
case "$MODE" in
  check|cam|pose|dry-run|motion) ;;
  *) echo "usage: bash run_full.sh [check|cam|pose|dry-run|motion] [seconds|auto]"; exit 2 ;;
esac

for f in "$SERVER" "$BUILDER" "$TRIAL" "$LINKCHK" "$CAMCHK" "$CAMSRC" "$POSECHK" "$HW" \
         "$HOME_JSON" "$CPHOME" "$RESETSH"; do
  [ -f "$f" ] || { echo "FAIL 缺文件 $f"; exit 1; }
done

echo "=== [0/5] 产物自检（ARTIFACTS.md5）"
cd "$LDIR" || exit 1
if [ -f ARTIFACTS.md5 ]; then
  if ! MD5OUT=$(md5sum -c ARTIFACTS.md5 2>&1); then
    echo "FAIL 有文件对不上，停在这里："; printf '%s\n' "$MD5OUT" | grep -v ': OK$'; exit 1
  fi
  echo "全部匹配"
else
  echo "（无 ARTIFACTS.md5，跳过）"
fi

echo "=== [1/5] 硬件锁"
if ! flock -n "$LOCK" true; then
  echo "FAIL 硬件锁被占：teleop / 录制 / 别的 rollout 还在跑。等它结束再跑这个。"
  exit 1
fi
echo "锁空着"

echo "=== [2/5] 轨迹与旋钮"
if [ ! -f "$TRAJ.npz" ] || [ "${PI0_REBUILD:-0}" = 1 ]; then
  echo "    构建轨迹（只读语料，并且重写 $HOME_JSON）..."
  PI0_DATA=${PI0_DATA:-/home/databall/drawer_today99_v1/dataset_train}
  "$SPY" -I -u "$BUILDER" --data "$PI0_DATA" --home-out "$HOME_JSON" --out "$TRAJ" || exit 1
fi
[ -f "$TRAJ.npz" ] || { echo "FAIL 没建成轨迹 $TRAJ.npz"; exit 1; }
[ -f "$TRAJ.json" ] || { echo "FAIL 没建成轨迹 $TRAJ.json"; exit 1; }

# 从 meta 读长度/拉伸倍数/相位/归位段，核对运行旋钮与复位目标
python3 - "$TRAJ.json" "$HOME_JSON" "$PI0_JOINT_RATE" "$PI0_GRIPPER_RATE" "$ARMTOL" "$CPHOME" <<'EOF' || exit 1
import json, math, sys
m = json.load(open(sys.argv[1]))
home = json.load(open(sys.argv[2]))
jr, gr, tol = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cp = json.load(open(sys.argv[6]))
FPS = 30.0
LEAD = m['lead_hold_rows']
meta_jr = m['action_rates']['joint_cap_rad_per_tick'] * FPS
meta_gr = m['action_rates']['jaw_cap_m_per_tick'] * FPS
print('    replay_id        = %s' % m['replay_id'])
print('    源演示           = %s 第 %s 拍（含）= %d 拍 = %.2f s'
      % (m['source_file'], m['segment_ticks'], m['horizon_raw'], m['horizon_raw'] / FPS))
print('    协调拉伸后       = %d 拍 = %.2f s（%.2f 倍）+ 前置 %d 拍保持不动'
      % (m['horizon_stretched_walk'], m['horizon_stretched_walk'] / FPS,
         m['stretch_factor_walk'], LEAD))
print('    总量             = %d 拍 = %.1f s ⇒ 建议窗口 --rollout-seconds %d'
      % (m['horizon'], m['horizon_seconds'], int(math.ceil(m['horizon_seconds'])) + 6))
print('    四步时间线（拉伸后、含前置 %d 拍）：' % LEAD)
for key, label in (('pull', '1) 拉抽屉'), ('cube', '2) 放方块'), ('close', '3) 合抽屉')):
    w = m['phases'][key]['walk']
    print('      %-10s %5.1f .. %5.1f s（第 %d..%d 拍）  %s'
          % (label, (w[0] + LEAD) / FPS, (w[1] + LEAD) / FPS, w[0] + LEAD, w[1] + LEAD,
             m['phases'][key]['what']))
rr = m.get('return_home') or {}
if rr:
    w = rr['walk']
    print('      %-10s %5.1f .. %5.1f s（第 %d..%d 拍）  %d 拍移动 + %d 拍保持'
          % ('4) 自动归位', (w[0] + LEAD) / FPS, (w[1] + LEAD) / FPS, w[0] + LEAD, w[1] + LEAD,
             rr['rows_move'], rr['rows_hold']))
print('    不跑归位的话     = 第 %d 拍 = %.1f s 就停在放好方块的位姿上'
      % (m['horizon_without_return'], m['horizon_without_return'] / FPS))
x = m['limits_max_excess_over_low_high']
if x == 0.0:
    print('    越出 LOW/HIGH    = 0 行（必须 0，否则闸门会触发重采样、游标跳行）')
else:
    print('    越出 LOW/HIGH    = 有！最大 %.4g  —— 不该出现，停' % x); sys.exit(1)
print('    驱动护栏余量     = 关节 %.5f/%.5f 夹爪 %.5f/%.5f（都必须 <1）'
      % (m['driver_guard']['joint_step_max_rad'], m['driver_guard']['joint_allow_rad_per_tick'],
         m['driver_guard']['jaw_step_max_m'], m['driver_guard']['jaw_allow_m_per_tick']))
print('    limited_target 实切量 = %.6f（必须 0 = 限速器一次都没动手）'
      % m['pipeline_limiter_cut_max'])
print('    抽屉爪           = 张开峰 %.4f@第 %d 拍 → 咬住 %.4f@第 %d 拍；'
      '重采样后咬住落在第 %d 拍 = %.1f s'
      % (m['drawer_claw_open_peak_raw'], m['reference_open_peak_tick'],
         m['drawer_claw_grip_raw'], m['reference_grip_tick'],
         m['pipeline_grip_tick'], m['pipeline_grip_tick'] / FPS))
print('    咬住那拍抽屉臂离演示位 = %.4f rad（必须 <= 0.02）'
      % m['pipeline_grip_arm_dev_rad'])
print('    拉多少           = lJ2 峰 %.4f → 末 %.4f（净 %.4f rad，≈11 cm）'
      % (m['drawer_arm_lJ2_max'], m['drawer_arm_lJ2_end'],
         abs(m['drawer_arm_lJ2_end'] - m['drawer_arm_lJ2_min'])))
print('    勾抽屉那一段     = 抽屉臂 lJ2 跨 %.4f rad，爪全程 %.4f..%.4f（张着 = 推不是夹）'
      % (m['close_phase_lJ2_span_raw'], m['drawer_claw_close_phase_min_raw'],
         m['drawer_claw_close_phase_max_raw']))
print('    放方块那一段     = 方块爪 %.4f（张）→ %.4f（合）→ %.4f（松）；'
      '方块臂 rJ2 跨 %.4f rad'
      % (m['cube_claw_open_raw'], m['cube_claw_shut_raw'], m['cube_claw_reopen_raw'],
         m['cube_arm_rJ2_span_raw']))
tgt = list(home['target'])
rt = list(m['reset_target'])
d_home = max(abs(a - b) for a, b in zip(tgt, rt))
print('    复位目标         = home_target_full.json 与轨迹 reset_target 的 L∞ = %.3g'
      '（必须 <=1e-9；json 那份小数点后第 10 位四舍五入过，所以不是逐位 0）' % d_home)
print('    跑完停在         = 末行与复位目标差 %.3g（必须 0 ⇒ 跑完就坐在起跑位姿上，可原地再跑）'
      % m['return_end_dev_from_reset_target'])
# 复位要从"上一轮停的地方"（≈ 上一套的停放位）走到起跑位姿，跨多少
cph = list(cp['target'])
d_cp = [abs(a - b) for a, b in zip(tgt, cph)]
dim = max(range(14), key=lambda i: d_cp[i])
print('    开跑前复位要跨   = 相对上一套停放位 L∞ %.4f rad（最大那维 dim%d；'
      '客户端复位是关节空间直线插值，限速 0.0133 rad/拍 ⇒ 至少 %.0f 拍 ≈ %.1f s）'
      % (max(d_cp), dim, max(d_cp) / (jr / FPS), max(d_cp) / jr))
ok = True
if m['pipeline_limiter_cut_max'] > 1e-12:
    print('FAIL limited_target 实切量 %.6g（应 <1e-12）—— 轨迹没压进帽里，时序保不住'
          % m['pipeline_limiter_cut_max']); ok = False
if m['pipeline_grip_arm_dev_rad'] > 0.02:
    print('FAIL 咬住那拍抽屉臂离演示位 %.4f rad（>0.02）—— 时序没保住'
          % m['pipeline_grip_arm_dev_rad']); ok = False
if m['drawer_claw_close_phase_min_raw'] < 0.04:
    print('FAIL 合抽屉那一段抽屉爪最小 %.4f（<0.04）—— 那一段的爪应当一直张着'
          % m['drawer_claw_close_phase_min_raw']); ok = False
if not (m['cube_claw_open_raw'] >= 0.05 and m['cube_claw_shut_raw'] <= 0.04
        and m['cube_claw_reopen_raw'] >= 0.05):
    print('FAIL 方块爪的"张开->合上->再张开"不成立：%.4f -> %.4f -> %.4f'
          % (m['cube_claw_open_raw'], m['cube_claw_shut_raw'], m['cube_claw_reopen_raw']))
    ok = False
if not m.get('return_home'):
    print('FAIL 没有自动归位段 —— 这一版要的就是跑完自己回位，别用 --no-return 建的轨迹')
    ok = False
elif m['return_end_dev_from_reset_target'] != 0.0:
    print('FAIL 归位段末行与复位目标差 %.6g —— 跑完不会坐在复位点上'
          % m['return_end_dev_from_reset_target']); ok = False
if d_home > 1e-9:
    print('FAIL 复位目标与轨迹起跑状态不一致（%.3g）：客户端会复位到 A 点、'
          '而 arming 参照是 B 点，永远 arm 不上' % d_home); ok = False
dg = m['driver_guard']
if dg['joint_step_max_rad'] >= dg['joint_allow_rad_per_tick'] or dg['jaw_step_max_m'] >= dg['jaw_allow_m_per_tick']:
    print('FAIL 有拍会撞驱动护栏 verified_hardware.py:474/476，整跑会被打断'); ok = False
if not (0.05 <= tol <= 0.15):
    print('FAIL arming 容差 %.3f 不在 [0.05,0.15] 内。客户端 settle_home 只保证 '
          '0.035 rad，松到 0.25 会让别的位姿有误触发的余地。' % tol); ok = False
if abs(jr - meta_jr) > 1e-9:
    print('FAIL 关节速率旋钮 %.4f 与构建时的 %.4f 不一致。轨迹是照着构建值压的，'
          '改了就白建（要么 PI0_REBUILD=1 重建，要么把值调回去）。' % (jr, meta_jr)); ok = False
if abs(gr - meta_gr) > 1e-9:
    print('FAIL 夹爪速率旋钮 %.4f 与构建时的 %.4f 不一致。同一条。' % (gr, meta_gr)); ok = False
sys.exit(0 if ok else 1)
EOF

if [ "$WINDOW" = auto ]; then
  WINDOW=$(python3 -c "import json,math;m=json.load(open('$TRAJ.json'));print(int(math.ceil(m['horizon_seconds']))+6)")
fi
echo "    运行旋钮         = joint_rate=$PI0_JOINT_RATE gripper_rate=$PI0_GRIPPER_RATE alpha=$PI0_ACTION_ALPHA"
echo "    两条臂策略       = 都按轨迹发（不做任何策略推理、不调用任何模型）"
echo "    arming 容差      = $ARMTOL（全 14 维 L∞）"
echo "    动作窗口         = ${WINDOW}s"
python3 - "$TRAJ.json" "$WINDOW" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); w = float(sys.argv[2]) * 30.0
need = m['horizon']; nor = m['horizon_without_return']
if w < need:
    print('    ⚠ 窗口 %.0fs = %d 拍，短于轨迹 %d 拍：会跑到第 %d 拍就停。'
          % (float(sys.argv[2]), w, need, int(w)))
    if w < nor:
        print('      （连归位段都进不去 —— 想在原地看着它自己回位，窗口至少 %.0f s）'
              % (m['horizon_seconds'] + 3))
EOF

echo "=== [3/5] 回放服务（端口 $PORT）"
NEED_START=1
if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT "; then
  LISTENER=$(ss -ltnp 2>/dev/null | grep "127.0.0.1:$PORT " | head -1)
  PID=$(printf '%s\n' "$LISTENER" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  OURCMD=""
  if [ -n "$PID" ] && [ -r "/proc/$PID/cmdline" ]; then
    OURCMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
  fi
  case "$OURCMD" in
    *"$SERVER"*)
      if [ "$MODE" = motion ] && [ "${PI0_REUSE_SERVER:-0}" != 1 ]; then
        # 回放服务是**有状态**的：游标走过就不再回头。干跑一次会把游标带到末尾，
        # 真机再复用它就等于每拍只发末行 —— 手臂根本不会动。真机必须从第 0 拍起。
        echo "端口 $PORT 上是本套回放服务（pid $PID），但真机要从头放：停掉它、起重起的"
        kill "$PID" 2>/dev/null
        for _ in $(seq 20); do ss -ltn | grep -q "127.0.0.1:$PORT " || break; sleep 0.5; done
        if ss -ltn | grep -q "127.0.0.1:$PORT "; then
          echo "FAIL 端口 $PORT 没释放（旧服务没死）。手工处理：kill $PID"
          exit 1
        fi
      else
        echo "端口 $PORT 上已是本套回放服务（pid $PID），沿用（干跑；游标状态不重置）"
        NEED_START=0
      fi
      ;;
    *) echo "FAIL 端口 $PORT 的监听者不是本套回放服务，不能复用："
       echo "     $LISTENER"
       if [ -z "$OURCMD" ]; then
         echo "     注：这个监听者在**别的用户**下跑（进程命令行读不到，多半上次是 root 起的），"
         echo "        本用户既不能确认它、也不能直接结束它。"
         echo "        要腾出这个端口：ss -ltnp 看清 pid 后 sudo kill <pid>，或换个端口跑。"
       fi
       echo "     本套服务应当是：$SERVER"
       echo "     处理：换一个空端口重跑，例如 PI0_PORT=18314 bash run_full.sh $MODE"
       exit 1 ;;
  esac
fi
if [ "$NEED_START" = 1 ]; then
  [ -x "$SPY" ] || { echo "FAIL 缺解释器 $SPY"; exit 1; }
  if [ -e "$SERVLOG" ] && [ ! -w "$SERVLOG" ]; then
    # 上一次若是别的用户（平台以 root 跑过一次）留下的日志，本用户改不动它，
    # 下面 `: >` 与 `>>` 都会 Permission denied，服务起不来却只报"没就绪"。
    ALT="$SERVLOG.$(date +%Y%m%d_%H%M%S)"
    if mv "$SERVLOG" "$ALT" 2>/dev/null; then
      echo "旧日志 $SERVLOG 不归本用户，已挪到 $(basename "$ALT")"
    else
      SERVLOG="$ALT"
      echo "旧日志 $SERVLOG 挪不动，这次写到 $SERVLOG"
    fi
  fi
  : > "$SERVLOG"
  nohup setsid "$SPY" -I -u "$SERVER" --port "$PORT" --traj "$TRAJ.npz" \
        --arm-tol "$ARMTOL" --arm-deadline-calls 20 >>"$SERVLOG" 2>&1 &
  echo "已启动 pid $!，等就绪（本地服务，秒级）..."
  ready=0
  for _ in $(seq 30); do
    grep -q PI0_REMOTE_READY "$SERVLOG" 2>/dev/null && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" != 1 ]; then
    echo "FAIL 30 秒内没就绪，日志尾巴："; tail -n 20 "$SERVLOG"; exit 1
  fi
fi
grep -h "PI0_REMOTE_READY\|PI0_REMOTE_CHECKPOINT\|PI0_REMOTE_SHA256\|PI0_REMOTE_ENSEMBLE\|PI0_REMOTE_WARMUP\|REPLAY_TRAJ" "$SERVLOG" 2>/dev/null | tail -6
CKPT=$(sed -n 's/^PI0_REMOTE_CHECKPOINT //p' "$SERVLOG" | tail -1)
[ -n "$CKPT" ] || { echo "FAIL 读不到 replay_id"; exit 1; }

echo "=== [4/5] 链路自检（合成灰图问一次，核对 replay_id，不动电机）"
cd "$LROOT" || exit 1
export PYTHONPATH=$LDIR/../shared
"$PY" "$LINKCHK" --url "http://127.0.0.1:$PORT/" --expected-checkpoint "$CKPT" || {
  echo "FAIL 链路自检没过（服务没答 / 动作形状不对 / replay_id 对不上）"; exit 1; }

if [ "$MODE" = check ]; then
  echo "=== 自检模式：到这一步就停，**没有**连机械臂、没有开相机、没有使能任何电机。"
  echo "    上面全过 ⇒ 产物、轨迹、旋钮、服务、链路都对得上。"
  echo "    真机：bash run_full.sh motion   （约 ${WINDOW}s，由你自己按）"
  echo "    回放服务留在端口 $PORT 上（真机模式会自己停掉重起，从第 0 拍放）"
  exit 0
fi

# 相机闸门 —— 无真机。客户端要跑到第 8 秒才会因为相机坏掉而 blocked_or_aborted，
# 这里用同一套判据提前挡下来（别名绑定 / 端口被占 / 三路出帧）。
echo "=== [4b/5] 相机闸门（无真机：不碰 CAN、不使能电机、不发动作）"
if ! "$PY" -I -u "$CAMCHK" --client "$CAMSRC"; then
  echo "FAIL 相机闸门没过 —— 先把推流/别名收拾好再往下走。"
  echo "    客户端会在开跑约 8 秒后报 blocked_or_aborted；那**不是**轨迹或服务的问题。"
  exit 1
fi

if [ "$MODE" = cam ]; then
  echo "=== 相机模式：到这一步就停。"
  echo "    三路相机现在都能出帧 ⇒ 客户端开跑前那三道相机判断都会过。"
  echo "    这一步**没有**连机械臂、没有使能电机、没有发任何动作。"
  echo "    真机：bash run_full.sh motion   （约 ${WINDOW}s，由你自己按）"
  exit 0
fi

# 姿态闸门 —— 无真机。客户端在使能之前有一道 measured_preload：实测位姿被 clip 到策略
# 动作盒，任何关节被削掉 > 0.01 rad（夹爪 > 0.001 m）就 abort（Raw pose outside preload
# tolerance，0 命令）。这里用**官方那支脚本的只读模式**（不给 --execute ⇒ read_only=True）
# 被动读一次反馈，再把位姿喂给**活客户端那一份** measured_preload 下判决。
pose_check() {          # $1 = 写日志的路径；0 = 放行、1 = 会 abort、2 = 读不到
  local log=$1
  "$PY" -I -u "$POSECHK" --hw "$HW" --reset-dir "$RESETDIR" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

if [ "$MODE" = pose ]; then
  echo "=== 姿态闸门（无真机：被动读一次反馈，不使能、不移动、不发帧）"
  POSE_LOG=$(mktemp "${TMPDIR:-/tmp}/pi0_pose_check.XXXXXX")
  pose_check "$POSE_LOG"; POSE_RC=$?
  echo "    （全文在上，含官方那份 Current/Target/Delta 与逐维越界量；日志 $POSE_LOG）"
  case "$POSE_RC" in
    0) echo "    ⇒ 客户端那道 preload 闸门会放行。真机：bash run_full.sh motion"; exit 0 ;;
    1) echo "    ⇒ 先归位再跑（见上面那两行修法）"; exit 1 ;;
    *) echo "    ⇒ 没读出来，别据此判断；先看上面报了什么"; exit 2 ;;
  esac
fi

STAMP=$(date +%Y%m%d_%H%M%S)
R=$LROOT/inference/reports/pi0_full_task_$STAMP
EXTRA=""

# [4c/5] 开跑前官方归位 —— **真动**：官方脚本立即使能并带两条臂回训练起始位姿。
# 放在这里的原因：客户端使能之前那道 measured_preload 闸门容不下任何关节越出名义盒
# 0.01 rad，而两条臂平时正坐在 J2 下界 0 / J3 上界 0 上，断电重启或久放之后一旦下垂
# 就挡住（2026-10-16 上午两次 0 命令 abort 就是这个）。官方 reset 自带恢复限位 ±3.20。
if [ "$MODE" = motion ]; then
  RESETMODE=${PI0_RESET:-1}
  echo "=== [4c/5] 开跑前官方归位（PI0_RESET=$RESETMODE：1=每次都归 / auto=只在越界时归 / 0=跳过）"
  NEED_RESET=1
  if [ "$RESETMODE" = 0 ]; then
    NEED_RESET=0
    echo "    按要求跳过。⚠ 若实测位姿已越出名义盒，客户端会在使能前 abort（0 命令）"
    echo "      想先确认：bash run_full.sh pose（被动读一次，不动）"
  elif [ "$RESETMODE" = auto ]; then
    echo "    先被动读一次位姿，看要不要归（不使能、不移动）..."
    POSE_LOG=$(mktemp "${TMPDIR:-/tmp}/pi0_pose_check.XXXXXX")
    pose_check "$POSE_LOG"; POSE_RC=$?
    if [ "$POSE_RC" = 0 ]; then
      NEED_RESET=0
      echo "    实测位姿已在名义盒内 ⇒ 不归位，直接往下走"
    elif [ "$POSE_RC" = 1 ]; then
      echo "    实测位姿越出名义盒 ⇒ 归位"
    else
      echo "    读不到位姿（找不到官方姿态脚本）⇒ 按需要归处理"
    fi
  fi
  if [ "$NEED_RESET" = 1 ]; then
    echo "    ⚠ 这一步是**真动**：$RESETSH 会立即给两条臂使能，并沿受限走廊把它们"
    echo "      带到训练起始位姿（它自带恢复限位 ±3.20，只朝目标单向把越界关节收回来）。"
    echo "      现在：清空工作区、托住两条臂、手放急停上。不想跑就加 PI0_RESET=0。"
    RESET_LOG=${PI0_RESET_LOG:-/tmp/pi0_reset_$STAMP.txt}
    COUNT=${PI0_COUNTDOWN:-0}
    if [ "$COUNT" -gt 0 ] 2>/dev/null; then
      echo "    倒数 $COUNT 秒（倒完**立刻**开始归位，不是等客户端）..."
      for i in $(seq "$COUNT" -1 1); do printf '  %2d\n' "$i"; sleep 1; done
      SKIP_COUNTDOWN=1
    fi
    if ! bash "$RESETSH" 2>&1 | tee "$RESET_LOG"; then
      echo "FAIL 官方归位脚本退出码非 0（全文在上）；日志 $RESET_LOG"
      echo "     别往下走：客户端一定会在使能之前 abort。"
      exit 1
    fi
    if ! grep -q PIPER_RESET_SUCCESS "$RESET_LOG"; then
      echo "FAIL 没看到 PIPER_RESET_SUCCESS（只认这一行才算归位成功）；日志 $RESET_LOG"
      exit 1
    fi
    echo "    官方归位完成：$(grep -m1 PIPER_RESET_SUCCESS "$RESET_LOG")"
    echo "    （两条臂现在是使能保持、停在归位目标上；接下来那次复位是从这里再挪到"
    echo "      本回放的起跑位姿）"
  fi
fi
if [ "$MODE" = motion ]; then
  echo "=== [5/5] 真机动作窗口 ${WINDOW}s"
  echo "    上游：第 [4c/5] 步已用官方归位把两条臂带回训练起始位姿（除非 PI0_RESET=0）"
  echo "    流程：preflight -> 相机核对 -> 被动 peek -> 复位到起跑位姿（这步是真动："
  echo "          两条臂都会开到演示起跑那一拍的位姿，抽屉臂主要跨 j2/L∞ 约 1.0 rad，"
  echo "          限速 0.0133 rad/拍、要连续稳 0.5 s 才算到位）-> 跳过 prelift -> 按回放轨迹跑，30 Hz"
  echo "    随后（无需人工干预）：拉抽屉 -> 放方块 -> 合上抽屉 -> **自动归位**，跑完自己停"
  echo "    抽帧核对过的事实（都是从语料视频里逐帧看出来的，不是推断）："
  echo "      · 爪在 202-220 拍合到 0.0029 m 并保持到 279 拍（两指间夹着约 3 mm 厚的东西"
  echo "        = 真咬住抽屉正面那个黑色圆环拉手，不是空合）；抽屉约 230 拍开始动、"
  echo "        290-300 拍拉到位置并一直开着；臂在 282-294 拍松开、停在抽屉前不动。"
  echo "      · 第 3 步（合上抽屉）是**推**不是夹：末端压在抽屉正面左半、爪两片张开跨在"
  echo "        面板边缘两侧；抽屉在 672-681 拍之间一口气滑回关位，之后末端贴着面板到"
  echo "        约 710 拍才退开。"
  echo "      · 蓝方块最后在**关上的抽屉里面**（圆盘此时已空）；整段工作区里没有出现过人手。"
  echo "    ⚠ 开头那次复位、以及末尾那次自动归位，走的都是**关节空间直线/余弦缓动**"
  echo "      （客户端 hardware.home 的实现），演示里没有这两条路线，所以离线证不了"
  echo "      途中不碰抽屉。第一次跑：手放急停上；可以先 dry-run 一遍看复位目标对不对。"
  echo "    ⚠ 窗口给短了就看不到归位（脚本会提示到第几拍停）。"
  COUNT=${PI0_COUNTDOWN:-0}
  if [ "${SKIP_COUNTDOWN:-0}" = 1 ]; then
    echo "    （倒数已经在归位那一步用掉了，这里不再等）"
  elif [ "$COUNT" -gt 0 ] 2>/dev/null; then
    for i in $(seq "$COUNT" -1 1); do printf '  %2d\n' "$i"; sleep 1; done
  fi
  EXTRA="--enable-motion --no-prelift --rollout-seconds $WINDOW --trajectory $R.trajectory.jsonl"
  [ -n "${PI0_TRACE:-1}" ] && [ "${PI0_TRACE:-1}" != 0 ] && EXTRA="$EXTRA --trace $R.trace.jsonl"
else
  echo "=== [5/5] 干跑（电机不使能）"
fi

"$PY" -u "$TRIAL" $EXTRA --remote-url "http://127.0.0.1:$PORT/" \
      --expected-remote-checkpoint "$CKPT" --report "$R.json" 2>&1 | tee "$R.log"
TRIAL_EXIT=${PIPESTATUS[0]}
echo "EXIT=$TRIAL_EXIT"
echo "REPORT=$R.json"
echo "LOG=$R.log"
[ -f "$R.trajectory.jsonl" ] && echo "TRAJECTORY=$R.trajectory.jsonl"

echo "=== 回放侧判定"
if grep -q REPLAY_ARMED "$SERVLOG"; then
  echo "ARMED = 是（$(grep -m1 REPLAY_ARMED "$SERVLOG")）"
  echo "        复位精度看上面那个 dist_all：< $ARMTOL 才 arm，越接近 0 越好"
elif [ "$MODE" = dry-run ]; then
  echo "ARMED = 否（干跑：电机不使能，手臂没归位，服务端本来就不该 arm —— 正常）"
else
  echo "ARMED = 否 —— 服务端从没见到匹配起跑位姿的请求，回放没跑（手臂应当没动）"
fi
if grep -q REPLAY_DONE "$SERVLOG"; then echo "轨迹放完 = 是（含末尾自动归位）"
elif [ "$MODE" = dry-run ]; then echo "轨迹放完 = n/a（干跑）"
else echo "轨迹放完 = 否（窗口不够或提前中止）"; fi
python3 "$LDIR/check_full_run.py" --report "$R.json" --meta "$TRAJ.json" 2>/dev/null \
  || echo "（check_full_run.py 读不了这份报告，手工看 $R.json）"
exit "$TRIAL_EXIT"
