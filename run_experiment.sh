#!/bin/bash

if [ -z "$1" ]; then
    echo -e "\033[1;31mError: No Python script provided!\033[0m"
    echo -e "Usage: ./run_experiment.sh <script_name.py>"
    exit 1
fi

TARGET_SCRIPT=$1
BASENAME=$(basename "$TARGET_SCRIPT" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LOG_FILE="gpu_metrics_${BASENAME}_${TIMESTAMP}.csv"
NSYS_OUT="nsys_report_${BASENAME}_${TIMESTAMP}"

echo -e "\033[1;32m[+] Starting experiment: $TARGET_SCRIPT\033[0m"
echo -e "\033[1;32m[+] CSV Log: $LOG_FILE\033[0m"
echo -e "\033[1;32m[+] Nsight Report: $NSYS_OUT.nsys-rep\033[0m"
echo "---------------------------------------------------------"

monitor_process() {
    echo "Timestamp,RAM_Used_MB,GPU_Util_%,VRAM_Used_MB,Temp_C,Power_W" > "$LOG_FILE"
    
    while true; do
        NOW=$(date +%H:%M:%S)
        RAM_RAW=$(free -m | awk '/Mem:/ {print $3}')
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')
        
        # Tiszta adatsor a CSV fájlba
        echo "$NOW,$RAM_RAW,$GPU_INFO" >> "$LOG_FILE"
        
        VRAM=$(echo $GPU_INFO | cut -d',' -f2)
        UTIL=$(echo $GPU_INFO | cut -d',' -f1)
        TEMP=$(echo $GPU_INFO | cut -d',' -f3)
        PWR=$(echo $GPU_INFO | cut -d',' -f4)
        
        echo -e "\033[0;36m[MONITOR $NOW]\033[0m VRAM: \033[1;33m${VRAM} MB\033[0m | GPU: ${UTIL}% | Temp: ${TEMP}°C | Pwr: ${PWR}W"
        
        sleep 1
    done
}

monitor_process &
MONITOR_PID=$!

echo -e "\033[1;35m[+] Launching Nsight Systems profiling...\033[0m"

PYTHON_EXEC=$(uv run which python)

sudo env PATH="$PATH" HF_HOME="$HOME/.cache/huggingface" PYTHONUNBUFFERED=1 nsys profile -t cuda,osrt -o "$NSYS_OUT" "$PYTHON_EXEC" "$TARGET_SCRIPT"

EXIT_STATUS=$?

echo "---------------------------------------------------------"
echo -e "\033[1;32m[+] Target script finished. Stopping monitor...\033[0m"

kill $MONITOR_PID
wait $MONITOR_PID 2>/dev/null

if [ $EXIT_STATUS -eq 0 ]; then
    echo -e "\033[1;32m[+] Done! You can now open $NSYS_OUT.nsys-rep in Nsight Systems.\033[0m"
else
    echo -e "\033[1;31m[-] Profiling ended with an error (Exit code: $EXIT_STATUS)!\033[0m"
fi