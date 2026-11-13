#!/usr/bin/env bash
# Samples a phone over adb every few seconds while an on-device benchmark runs (engine/Bench.kt):
# GPU busy % and clock (Samsung /sys/kernel/gpu), thermal-service temperatures (AP = SoC, SKIN, BAT), the app's CPU
# share from top, and the system-wide CPU load. Writes CSV to stdout.
# usage: SERIAL=<adb serial> scripts/phone_sampler.sh SECONDS [INTERVAL]
set -u
export MSYS_NO_PATHCONV=1
ADB="${ADB:-adb} -s ${SERIAL:?set SERIAL}"
secs=$1; every=${2:-5}
echo "t_s,gpu_busy_pct,gpu_clock_mhz,ap_c,skin_c,bat_c,app_cpu_pct,load1"
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt "$secs" ]; do
  now=$(( $(date +%s) - t0 ))
  out=$($ADB shell '
    pid=$(pidof ai.echogesture.app)
    echo "gpu $(cat /sys/kernel/gpu/gpu_busy 2>/dev/null) $(cat /sys/kernel/gpu/gpu_clock 2>/dev/null)"
    dumpsys thermalservice | sed -n "/Current temperatures from HAL/,/Current cooling/p" | grep -E "mName=(AP|SKIN|BAT)," | head -3
    [ -n "$pid" ] && top -b -n 1 -p $pid | tail -1
    cat /proc/loadavg' 2>/dev/null | tr -d '\r')
  gpu=$(echo "$out" | awk '/^gpu /{print $2","$3}')
  temp() { echo "$out" | grep "mName=$1," | head -1 | sed -E 's/.*mValue=([0-9.]+).*/\1/'; }
  cpu=$(echo "$out" | grep "ai.echogesture" | awk '{print $9}')
  load=$(echo "$out" | tail -1 | awk '{print $1}')
  echo "$now,${gpu:-,},$(temp AP),$(temp SKIN),$(temp BAT),${cpu:-},${load:-}"
  sleep "$every"
done
