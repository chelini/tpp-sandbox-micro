#!/bin/bash

for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
  echo performance > $i
done

for cpunum in $(cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | cut -s -d, -f2- | tr ',' '\n' | sort -un); do
    echo 0 >/sys/devices/system/cpu/cpu$cpunum/online
done

echo 0 > tee /proc/sys/kernel/randomize_va_space
echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

cat /sys/devices/system/cpu/intel_pstate/no_turbo
