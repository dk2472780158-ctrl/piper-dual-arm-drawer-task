#!/usr/bin/env bash
# cubephase 回放启动器 -- 用**时间重参数化过的录制轨迹**顶替 π0 推理服务。
#
# 目的：用户要"就算抽屉没拉开，方块臂也要把方块夹住放进去"。
#
# 与 run_cp.sh 的区别只有一处：推理服务不是模型，是 replay_server.py
# （逐字回放一段录制演示，只做过时间重参数化）。**客户端一个字节都不改**：
# TRIAL 指向的是本包 shared/ 里那份 pi0_real_trial_v8.py（与活文件逐字节相同），
# 全部闸门、限速、跟踪校验、归位、报告、trace 都还是原版。
#
# 三条硬约束（脚本会自己挡住）：
#   1) PI0_JOINT_RATE / PI0_GRIPPER_RATE 必须等于轨迹构建时的值。
#      轨迹是照着那两个数把每拍步长压到帽内建的；运行时改大改小都会让
#      limited_target 重新动手，时序就白保了。
#   2) 参数 --rollout-seconds 必须盖住拉伸后的长度（脚本从 meta 读，默认 +6 s）。
#   3) 端口上若已有监听者，只有确认是**本脚本起的那个 replay 服务**才复用。
set -uo pipefail

PORT=${PI0_PORT:-18312}
LDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LROOT=/home/databall/pi0_drawer_mixed_74_v1
PY=${PI0_PY:-/home/databall/pi0_infer_3090/.venv/bin/python}
SPY=/home/databall/miniconda3/envs/lerobot_v30/bin/python
TRIAL=$LDIR/../shared/pi0_real_trial_v8.py
LINKCHK=$LDIR/../shared/check_pi0_link_v8.py
HOME_JSON=$LDIR/../shared/home_target_cubephase.json
SERVER=$LDIR/replay_server.py
BUILDER=$LDIR/make_traj.py
TRAJ=$LDIR/cube_phase_traj
SERVLOG=$LDIR/server_${PORT}.log
LOCK=/home/databall/pico_piper_teleop/runtime/hardware.lock

# 客户端原版就是这些默认值；这里显式 export，好让下面第 [2/5] 步能核对。
export PI0_HOME_JSON=$HOME_JSON
export PI0_PRELIFT_RAD=0
export PI0_GRIPPER_RATE=${PI0_GRIPPER_RATE:-0.030}
export PI0_JOINT_RATE=${PI0_JOINT_RATE:-0.40}
export PI0_ACTION_ALPHA=${PI0_ACTION_ALPHA:-0.25}
export PI0_FILTER_GAP_S=${PI0_FILTER_GAP_S:-1.0}
export PI0_IMAGE_ORDER=${PI0_IMAGE_ORDER:-rgb}

MODE=${1:-dry-run}
WINDOW=${2:-auto}
case "$MODE" in
  dry-run|motion) ;;
  *) echo "usage: bash run_replay.sh [dry-run|motion] [seconds|auto]"; exit 2 ;;
esac

for f in "$SERVER" "$BUILDER" "$TRIAL" "$LINKCHK" "$HOME_JSON"; do
  [ -f "$f" ] || { echo "FAIL 缺文件 $f"; exit 1; }
done

echo "=== [0/5] 产物自检（ARTIFACTS.md5）"
cd "$LDIR" || exit 1
if [ -f ARTIFACTS.md5 ]; then
  if ! md5sum -c ARTIFACTS.md5 >/tmp/rp_md5.txt 2>&1; then
    echo "FAIL 有文件对不上，停在这里："; grep -v ': OK$' /tmp/rp_md5.txt; exit 1
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
  echo "    构建轨迹（只读语料 $PI0_DATA）..."
  PI0_DATA=${PI0_DATA:-/home/databall/drawer_today99_v1/dataset_train}
  "$SPY" -I -u "$BUILDER" --data "$PI0_DATA" --home-json "$HOME_JSON" --out "$TRAJ" || exit 1
fi
[ -f "$TRAJ.npz" ] || { echo "FAIL 没建成轨迹 $TRAJ.npz"; exit 1; }

# 从 meta 读长度 / 拉伸倍数 / 构建时的速率帽，并核对运行时的旋钮
python3 - "$TRAJ.json" "$PI0_JOINT_RATE" "$PI0_GRIPPER_RATE" <<'EOF' || exit 1
import json, math, sys
m = json.load(open(sys.argv[1]))
jr, gr = float(sys.argv[2]), float(sys.argv[3])
FPS = 30.0
meta_jr = m['action_rates']['joint_cap_rad_per_tick'] * FPS
meta_gr = m['action_rates']['jaw_cap_m_per_tick'] * FPS
print('    replay_id        = %s' % m['replay_id'])
print('    源演示           = %s（触发拍 %d）' % (m['source_file'], m['trigger_tick']))
print('    相位长度         = %d 拍 = %.2f s' % (m['horizon_raw'], m['horizon_raw'] / FPS))
print('    协调拉伸后       = %d 拍 = %.2f s（%.2f 倍）+ 前置 %d 拍保持不动'
      % (m['horizon_stretched_walk'], m['horizon_stretched_walk'] / FPS,
         m['stretch_factor_walk'], m['lead_hold_rows']))
print('    总量             = %d 拍 = %.1f s ⇒ 建议窗口 --rollout-seconds %d'
      % (m['horizon'], m['horizon_seconds'], int(math.ceil(m['horizon_seconds'])) + 6))
x = m['limits_max_excess_over_low_high']
if x == 0.0:
    print('    越出 LOW/HIGH    = 0 行（必须 0，否则闸门会触发重采样、游标跳行）')
else:
    print('    越出 LOW/HIGH    = 有！最大 %.4g  —— 不该出现，停' % x)
    sys.exit(1)
print('    驱动护栏余量     = 关节 %.5f/%.5f 夹爪 %.5f/%.5f（都必须 <1）'
      % (m['driver_guard']['joint_step_max_rad'], m['driver_guard']['joint_allow_rad_per_tick'],
         m['driver_guard']['jaw_step_max_m'], m['driver_guard']['jaw_allow_m_per_tick']))
print('    limited_target 实切量 = %.6f（必须 0 = 限速器一次都没动手）'
      % m['pipeline_limiter_cut_max'])
print('    爪合上那拍手臂离演示抓取位 = %.4f rad（必须 <= 0.02）'
      % m['pipeline_jaw_close_arm_dev_rad'])
print('    起跑位姿：方块臂 7 维 L∞(state0, home) = %.4f rad'
      % m['start_state_l_inf_to_home_cube_arm'])
ok = True
if m['pipeline_limiter_cut_max'] > 1e-12:
    print('FAIL limited_target 实切量 %.6g（应 <1e-12）—— 轨迹没压进帽里，时序保不住' % m['pipeline_limiter_cut_max']); ok = False
if m['pipeline_jaw_close_arm_dev_rad'] > 0.02:
    print('FAIL 爪合上那拍手臂离演示抓取位 %.4f rad（>0.02）—— 时序没保住' % m['pipeline_jaw_close_arm_dev_rad']); ok = False
dg = m['driver_guard']
if dg['joint_step_max_rad'] >= dg['joint_allow_rad_per_tick'] or dg['jaw_step_max_m'] >= dg['jaw_allow_m_per_tick']:
    print('FAIL 有拍会撞驱动护栏 verified_hardware.py:474/476，整跑会被打断'); ok = False
if abs(jr - meta_jr) > 1e-9:
    print('FAIL PI0_JOINT_RATE=%.4f 与构建时的 %.4f 不一致。轨迹是照着构建值压的，'
          '改了就白建（要么 PI0_REBUILD=1 重建，要么把值调回去）。' % (jr, meta_jr)); ok = False
if abs(gr - meta_gr) > 1e-9:
    print('FAIL PI0_GRIPPER_RATE=%.4f 与构建时的 %.4f 不一致。同一条。' % (gr, meta_gr)); ok = False
sys.exit(0 if ok else 1)
EOF

if [ "$WINDOW" = auto ]; then
  WINDOW=$(python3 -c "import json,math;m=json.load(open('$TRAJ.json'));print(int(math.ceil(m['horizon_seconds']))+6)")
fi
echo "    运行旋钮         = joint_rate=$PI0_JOINT_RATE gripper_rate=$PI0_GRIPPER_RATE alpha=$PI0_ACTION_ALPHA"
echo "    抽屉臂策略       = 钉在起跑位姿前 7 维，不回放、不动"
echo "    动作窗口         = ${WINDOW}s"

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
       echo "     本套服务应当是：$SERVER"
       echo "     处理：换一个空端口重跑，例如 PI0_PORT=18313 bash run_replay.sh $MODE"
       exit 1 ;;
  esac
fi
if [ "$NEED_START" = 1 ]; then
  [ -x "$SPY" ] || { echo "FAIL 缺解释器 $SPY"; exit 1; }
  : > "$SERVLOG"
  nohup setsid "$SPY" -I -u "$SERVER" --port "$PORT" --traj "$TRAJ.npz" \
        --arm-deadline-calls 20 >>"$SERVLOG" 2>&1 &
  echo "已启动 pid $!，等就绪（本地服务，秒级）..."
  ready=0
  for _ in $(seq 30); do
    grep -q PI0_REMOTE_READY "$SERVLOG" 2>/dev/null && { ready=1; break; }
    sleep 1
  done
  [ "$ready" = 1 ] || { echo "FAIL 30 秒内没就绪，日志尾巴："; tail -n 20 "$SERVLOG"; exit 1; }
fi
grep -h "PI0_REMOTE_READY\|PI0_REMOTE_CHECKPOINT\|PI0_REMOTE_SHA256\|PI0_REMOTE_ENSEMBLE\|PI0_REMOTE_WARMUP\|REPLAY_TRAJ" "$SERVLOG" 2>/dev/null | tail -6
CKPT=$(sed -n 's/^PI0_REMOTE_CHECKPOINT //p' "$SERVLOG" | tail -1)
[ -n "$CKPT" ] || { echo "FAIL 读不到 checkpoint（replay_id）"; exit 1; }

echo "=== [4/5] 链路自检（合成灰图问一次，核对 checkpoint，不动电机）"
cd "$LROOT" || exit 1
export PYTHONPATH=$LDIR/../shared
"$PY" "$LINKCHK" --url "http://127.0.0.1:$PORT/" --expected-checkpoint "$CKPT" || {
  echo "FAIL 链路自检没过（服务没答 / 动作形状不对 / checkpoint 对不上）"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
R=$LROOT/inference/reports/pi0_replay_cubephase_$STAMP
EXTRA=""
if [ "$MODE" = motion ]; then
  echo "=== [5/5] 真机动作窗口 ${WINDOW}s"
  echo "    流程：preflight -> 相机核对 -> 被动 peek -> 复位到起跑位姿（这步是真动）"
  echo "          -> 跳过 prelift -> 按回放轨迹跑，30 Hz"
  echo "    回放时间线（拉伸后）：方块臂起步、爪合上约 %.1f s、之后抬臂放进抽屉" \
       "$(python3 -c "import json;print(json.load(open('$TRAJ.json'))['pipeline_jaw_close_tick']/30.0)")"
  echo "    演示对照（拉伸前）：方块臂起动 11.6 s、合爪 15.2 s、放下 19.0 s"
  echo "    ⚠ 抽屉臂停在归位点不动，两条臂的相对几何与演示不同 —— 手放急停上。"
  echo "    想先只看下探+抓取就小窗口：bash run_replay.sh motion 12"
  COUNT=${PI0_COUNTDOWN:-0}
  if [ "$COUNT" -gt 0 ] 2>/dev/null; then
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
elif [ "$MODE" = dry-run ]; then
  echo "ARMED = 否（干跑：电机不使能，手臂没归位，服务端本来就不该 arm —— 正常）"
else
  echo "ARMED = 否 —— 服务端从没见到匹配起跑位姿的请求，回放没跑（手臂应当没动）"
fi
if grep -q REPLAY_DONE "$SERVLOG"; then echo "轨迹放完 = 是"
elif [ "$MODE" = dry-run ]; then echo "轨迹放完 = n/a（干跑）"
else echo "轨迹放完 = 否（窗口不够或提前中止）"; fi
python3 "$LDIR/check_replay_run.py" --report "$R.json" --meta "$TRAJ.json" 2>/dev/null \
  || echo "（check_replay_run.py 读不了这份报告，手工看 $R.json）"
exit "$TRIAL_EXIT"
