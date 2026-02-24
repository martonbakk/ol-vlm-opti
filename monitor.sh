#!/bin/bash

LOG_FILE="gpu_metrics_$(date +%Y%m%d_%H%M%S).csv"

echo "Timestamp,RAM_Used_MB,GPU_Util_%,VRAM_Used_MB,Temp_C,Power_W" > "$LOG_FILE"

trap 'echo -e "\n\033[1;32m[+] Monitoring stopped. Data successfully saved to: $LOG_FILE\033[0m"; tput cnorm; exit 0' SIGINT

tput civis
clear

while true; do
    tput cup 0 0
    
    echo -e "\033[1;36m=======================================================\033[0m"
    echo -e "\033[1;33m  VLM HARDWARE MONITOR (RTX 5080) - Exit: Ctrl+C \033[0m"
    echo -e "\033[1;36m=======================================================\033[0m"
    echo ""
    
    echo -e "\033[1;32m---  SYSTEM MEMORY (Linux RAM) ---\033[0m"
    free -h | grep Mem | awk '{print "Used: " $3 " / Total: " $2 " | Free: " $4}'
    echo ""
    
    echo -e "\033[1;32m--- 🎮 GRAPHICS CARD (GPU & VRAM) ---\033[0m"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | awk -F", " '{
        print "GPU Utilization: " $1
        print "VRAM Usage:      " $2 " / " $3
        print "Temperature:     " $4 " °C"
        print "Power Draw:      " $5
    }'
    echo ""
    echo -e "\033[1;36m=======================================================\033[0m"
    echo -e "\033[1;90m[Logging data to $LOG_FILE...]\033[0m"
    
    TIMESTAMP=$(date +%H:%M:%S)
    RAM_RAW=$(free -m | grep Mem | awk '{print $3}')
    GPU_RAW=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')
    
    echo "$TIMESTAMP,$RAM_RAW,$GPU_RAW" >> "$LOG_FILE"
    
    sleep 1
done