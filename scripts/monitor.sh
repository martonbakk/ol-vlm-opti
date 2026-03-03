#!/bin/bash
# GPU monitor - requires nvidia-smi. Linux: use 'free' for RAM. Windows: RAM not available.

LOG_FILE="gpu_metrics_$(date +%Y%m%d_%H%M%S).csv"
echo "Timestamp,RAM_Used_MB,GPU_Util_%,VRAM_Used_MB,Temp_C,Power_W" > "$LOG_FILE"

trap 'echo -e "\n[+] Stopped. Data: $LOG_FILE"; tput cnorm 2>/dev/null; exit 0' SIGINT
tput civis 2>/dev/null

while true; do
    tput cup 0 0 2>/dev/null || clear
    echo "============== VLM GPU Monitor (Ctrl+C to stop) ==============="
    nvidia-smi
    echo "---------------------------------------------------------------"
    echo "[Logging to $LOG_FILE]"
    TIMESTAMP=$(date +%H:%M:%S)
    RAM_RAW=$(free -m 2>/dev/null | awk '/Mem:/ {print $3}' || echo "0")
    GPU_RAW=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "$TIMESTAMP,$RAM_RAW,$GPU_RAW" >> "$LOG_FILE"
    sleep 1
done
