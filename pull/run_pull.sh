#!/usr/bin/env bash
# 拉抽屉单测启动器 -- 用**时间重参数化过的录制轨迹**顶替 π0 推理服务。
#
# 目的：先单独验一件事——**机器自己能不能把抽屉拉开**。走的是演示里同一段动作
# （today99 / episode_000041 第 48..300 拍，就是抽屉臂伸手、咬住抽屉正面左上那个
# 黑色拉环、把抽屉拉出约 11 cm、松开的那一段），不是策略、不是模型。
# 那个拉环是**抽屉自己的把手**（不是另摆的圈），所以操作步骤里没有"摆布圈"这件事。
#
# 与 cubephase 那份的区别只有一处：那份把抽屉臂钉在归位点上（它要动方块臂），
# 这份两条臂都由轨迹供行 —— 因为这里动的就是抽屉臂。**客户端一个字节都不改**：
# TRIAL 指向本包 shared/ 里那份 pi0_real_trial_v8.py（与活文件逐字节相同），全部闸门、限速、
# 跟踪校验、归位、报告、trace 都还是原版；只把 PI0_HOME_JSON 指到本目录的
# home_target_pull.json（复位目标 = 演示起跑那一拍手臂真正在的地方）。
#
# 三条硬约束（脚本会自己挡住）：
#   1) PI0_JOINT_RATE / PI0_GRIPPER_RATE 必须等于轨迹构建时的值。
#   2) arming 容差必须比客户端 settle_home 的 0.035 rad 松、又比"上一轮停下来的
#      位姿离起跑位姿的距离"紧得多，否则要么 arm 不上、要么被被动 peek 误触发。
#   3) 端口上若已有监听者，只有确认是**本脚本起的那个 replay 服务**才复用。
set -uo pipefail

PORT=${PI0_PORT:-18313}
ARMTOL=${PI0_ARM_TOL:-0.10}
LDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CPHOME=$LDIR/../shared/home_target_cubephase.json                 # 只读它来打印"复位要跨多少"
LROOT=/home/databall/pi0_drawer_mixed_74_v1
PY=${PI0_PY:-/home/databall/pi0_infer_3090/.venv/bin/python}
SPY=/home/databall/miniconda3/envs/lerobot_v30/bin/python
TRIAL=$LDIR/../shared/pi0_real_trial_v8.py
LINKCHK=$LDIR/../shared/check_pi0_link_v8.py
HOME_JSON=$LDIR/home_target_pull.json
SERVER=$LDIR/replay_server.py
BUILDER=$LDIR/make_pull_traj.py
TRAJ=$LDIR/pull_traj
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
  *) echo "usage: bash run_pull.sh [dry-run|motion] [seconds|auto]"; exit 2 ;;
esac

for f in "$SERVER" "$BUILDER" "$TRIAL" "$LINKCHK" "$HOME_JSON" "$CPHOME"; do
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
  echo "    构建轨迹（只读语料 $PI0_DATA，并且重写 $HOME_JSON）..."
  PI0_DATA=${PI0_DATA:-/home/databall/drawer_today99_v1/dataset_train}
  "$SPY" -I -u "$BUILDER" --data "$PI0_DATA" --home-out "$HOME_JSON" --out "$TRAJ" || exit 1
fi
[ -f "$TRAJ.npz" ] || { echo "FAIL 没建成轨迹 $TRAJ.npz"; exit 1; }
[ -f "$TRAJ.json" ] || { echo "FAIL 没建成轨迹 $TRAJ.json"; exit 1; }

# 从 meta 读长度/拉伸倍数/构建时的速率帽/咬住拍，核对运行时的旋钮与复位目标
python3 - "$TRAJ.json" "$HOME_JSON" "$PI0_JOINT_RATE" "$PI0_GRIPPER_RATE" "$ARMTOL" "$CPHOME" <<'EOF' || exit 1
import json, math, sys
m = json.load(open(sys.argv[1]))
home = json.load(open(sys.argv[2]))
jr, gr, tol = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cp = json.load(open(sys.argv[6]))
FPS = 30.0
meta_jr = m['action_rates']['joint_cap_rad_per_tick'] * FPS
meta_gr = m['action_rates']['jaw_cap_m_per_tick'] * FPS
print('    replay_id        = %s' % m['replay_id'])
print('    源演示           = %s 第 %s 拍（含）= %d 拍 = %.2f s'
      % (m['source_file'], m['segment_ticks'], m['horizon_raw'], m['horizon_raw'] / FPS))
print('    协调拉伸后       = %d 拍 = %.2f s（%.2f 倍）+ 前置 %d 拍保持不动'
      % (m['horizon_stretched_walk'], m['horizon_stretched_walk'] / FPS,
         m['stretch_factor_walk'], m['lead_hold_rows']))
print('    总量             = %d 拍 = %.1f s ⇒ 建议窗口 --rollout-seconds %d'
      % (m['horizon'], m['horizon_seconds'], int(math.ceil(m['horizon_seconds'])) + 6))
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
print('    拉多少           = lJ2 %.4f → 峰 %.4f → 末 %.4f（净 %.4f rad，≈11 cm）'
      % (m['drawer_arm_lJ2_min'], m['drawer_arm_lJ2_max'], m['drawer_arm_lJ2_end'],
         abs(m['drawer_arm_lJ2_end'] - m['drawer_arm_lJ2_min'])))
print('    这一段只动抽屉臂 = 方块臂偏离首行 %.5f rad（塔里就是常量）'
      % m['cube_arm_drift_in_traj'])
tgt = list(home['target'])
rt = list(m['reset_target'])
d_home = max(abs(a - b) for a, b in zip(tgt, rt))
print('    复位目标         = home_target_pull.json 与轨迹 reset_target 的 L∞ = %.3g（必须 0）'
      % d_home)
# 复位要从"上一轮停的地方"（≈ cubephase 停放位）走到起跑位姿，跨多少
cph = list(cp['target'])
d_cp = [abs(a - b) for a, b in zip(tgt, cph)]
dim = max(range(14), key=lambda i: d_cp[i])
print('    复位要跨         = 相对 cubephase 停放位 L∞ %.4f rad（最大那维 dim%d；'
      '客户端复位是关节空间直线插值，限速 0.0133 rad/拍 ⇒ 至少 %.0f 拍 ≈ %.1f s）'
      % (max(d_cp), dim, max(d_cp) / (jr / FPS), max(d_cp) / jr))
d_end = float(m['end_row'][dim]) - float(m['start_row'][dim])
print('    peek 拦得住吗    = 上一轮跑完的末行离起跑位姿（dim%d）%.4f rad（必须 >> %.2f）'
      % (dim, abs(d_end), tol))
ok = True
if m['pipeline_limiter_cut_max'] > 1e-12:
    print('FAIL limited_target 实切量 %.6g（应 <1e-12）—— 轨迹没压进帽里，时序保不住'
          % m['pipeline_limiter_cut_max']); ok = False
if m['pipeline_grip_arm_dev_rad'] > 0.02:
    print('FAIL 咬住那拍抽屉臂离演示位 %.4f rad（>0.02）—— 时序没保住'
          % m['pipeline_grip_arm_dev_rad']); ok = False
if m['cube_arm_drift_in_traj'] > 0.01:
    print('FAIL 轨迹里方块臂动了 %.5f rad —— 这一段本该只动抽屉臂' % m['cube_arm_drift_in_traj'])
    ok = False
dg = m['driver_guard']
if dg['joint_step_max_rad'] >= dg['joint_allow_rad_per_tick'] or dg['jaw_step_max_m'] >= dg['jaw_allow_m_per_tick']:
    print('FAIL 有拍会撞驱动护栏 verified_hardware.py:474/476，整跑会被打断'); ok = False
if d_home > 1e-9:
    print('FAIL 复位目标与轨迹起跑状态不一致（%.3g）：客户端会复位到 A 点、'
          '而 arming 参照是 B 点，永远 arm 不上' % d_home); ok = False
if not (0.05 <= tol <= 0.15):
    print('FAIL PI0_ARM_TOL=%.3f 不在 [0.05,0.15] 内。客户端 settle_home 只保证 '
          '0.035 rad，松到 0.25 会让被动 peek 有误触发的余地。' % tol); ok = False
if abs(d_end) <= 2 * tol:
    print('FAIL 末行离起跑位姿只有 %.4f rad，不足 2×arm_tol（%.2f）：'
          '第二次跑同一个测试时 peek 可能直接 arm（会被前置 10 拍吸收，但别靠它）'
          % (d_end, tol)); ok = False
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
echo "    两条臂策略       = 都按轨迹发；这一段里方块臂是常量，等价于不动"
echo "    arming 容差      = $ARMTOL（全 14 维 L∞）"
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
       if [ -z "$OURCMD" ]; then
         echo "     注：这个监听者在**别的用户**下跑（进程命令行读不到，多半上次是 root 起的），"
         echo "        本用户既不能确认它、也不能直接结束它。"
         echo "        要腾出这个端口：ss -ltnp 看清 pid 后 sudo kill <pid>，或换个端口跑。"
       fi
       echo "     本套服务应当是：$SERVER"
       echo "     处理：换一个空端口重跑，例如 PI0_PORT=18314 bash run_pull.sh $MODE"
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
[ -n "$CKPT" ] || { echo "FAIL 读不到 checkpoint（replay_id）"; exit 1; }

echo "=== [4/5] 链路自检（合成灰图问一次，核对 checkpoint，不动电机）"
cd "$LROOT" || exit 1
export PYTHONPATH=$LDIR/../shared
"$PY" "$LINKCHK" --url "http://127.0.0.1:$PORT/" --expected-checkpoint "$CKPT" || {
  echo "FAIL 链路自检没过（服务没答 / 动作形状不对 / checkpoint 对不上）"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
R=$LROOT/inference/reports/pi0_pull_drawer_$STAMP
EXTRA=""
if [ "$MODE" = motion ]; then
  echo "=== [5/5] 真机动作窗口 ${WINDOW}s"
  echo "    流程：preflight -> 相机核对 -> 被动 peek -> 复位到起跑位姿（这步是真动："
  echo "          两条臂都会开到演示起跑那一拍的位姿，抽屉臂主要跨 j2/L∞ 约 1.0 rad，"
  echo "          限速 0.0133 rad/拍、要连续稳 0.5 s 才算到位）-> 跳过 prelift -> 按回放轨迹跑，30 Hz"
  echo "    回放时间线（拉伸后）：伸手 -> 咬住抽屉拉环（约 %.1f s）-> 拉 -> 松开 -> 停" \
       "$(python3 -c "import json;print(json.load(open('$TRAJ.json'))['pipeline_grip_tick']/30.0)")"
  echo "    演示对照（拉伸前）：段第 0..252 拍，咬住第 166 拍、拉在第 172..222 拍、松开第 232..242 拍"
  echo "    抽帧核对过的事实：爪在 202-220 拍合到 0.0029 m 并保持到 279 拍"
  echo "      （两指间夹着约 3 mm 厚的东西 = 真咬住拉环，不是空合）；抽屉约 230 拍开始动、"
  echo "      290-300 拍拉到位置并一直开着；臂在 282-294 拍松开、停在抽屉前不动。"
  echo "    ⚠ 复位是一条**关节空间直线**（客户端 hardware.home 的实现），演示里没有这条路线"
  echo "      （演示只有'从起跑位姿往前走'那一条），所以离线证不了这条直线中途不碰抽屉。"
  echo "      第一次跑：手放急停上；可以先 dry-run 一遍看复位目标对不对。"
  echo "    ⚠ 方块臂在这一段里不动是**设计**（轨迹里它的 7 维是常量）。"
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
  echo "        复位精度看上面那个 dist_all：< $ARMTOL 才 arm，越接近 0 越好"
elif [ "$MODE" = dry-run ]; then
  echo "ARMED = 否（干跑：电机不使能，手臂没归位，服务端本来就不该 arm —— 正常）"
else
  echo "ARMED = 否 —— 服务端从没见到匹配起跑位姿的请求，回放没跑（手臂应当没动）"
fi
if grep -q REPLAY_DONE "$SERVLOG"; then echo "轨迹放完 = 是"
elif [ "$MODE" = dry-run ]; then echo "轨迹放完 = n/a（干跑）"
else echo "轨迹放完 = 否（窗口不够或提前中止）"; fi
python3 "$LDIR/check_pull_run.py" --report "$R.json" --meta "$TRAJ.json" 2>/dev/null \
  || echo "（check_pull_run.py 读不了这份报告，手工看 $R.json）"
exit "$TRIAL_EXIT"
