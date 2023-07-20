#!/bin/bash

echo 0 > tee /proc/sys/kernel/randomize_va_space
echo 1 > tee /sys/devices/system/cpu/intel_pstate/no_turbo

for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
  echo performance > $i
done
