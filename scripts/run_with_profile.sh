#!/bin/bash
# Run training with Nsight Systems profiling. Usage: ./scripts/run_with_profile.sh [script.py]

SCRIPT="${1:-train.py}"
BASENAME=$(basename "$SCRIPT" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="gpu_metrics_${BASENAME}_${TIMESTAMP}.csv"
NSYS_OUT="nsys_report_${BASENAME}_${TIMESTAMP}"

echo -e "\033[1;32m[+] Running: $SCRIPT\033[0m"
echo -e "\033[1;32m[+] CSV: $LOG_FILE | Nsight: $NSYS_OUT.nsys-rep\033[0m"
echo "---------------------------------------------------------"

monitor_process() {
    echo "Timestamp,RAM_Used_MB,GPU_Util_%,VRAM_Used_MB,Temp_C,Power_W" > "$LOG_FILE"
    while true; do
        NOW=$(date +%H:%M:%S)
        RAM_RAW=$(free -m | awk '/Mem:/ {print $3}' 2>/dev/null || echo "0")
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        echo "$NOW,$RAM_RAW,$GPU_INFO" >> "$LOG_FILE"
        VRAM=$(echo "$GPU_INFO" | cut -d',' -f2)
        UTIL=$(echo "$GPU_INFO" | cut -d',' -f1)
        echo -e "\033[0;36m[MONITOR $NOW]\033[0m VRAM: \033[1;33m${VRAM} MB\033[0m | GPU: ${UTIL}%"
        sleep 1
    done
}

monitor_process &
MONITOR_PID=$!

cd "$(dirname "$0")/.." || exit 1
PYTHON_EXEC=$(uv run which python)
sudo env PATH="$PATH" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONUNBUFFERED=1 \
    nsys profile -t cuda,osrt -o "$NSYS_OUT" "$PYTHON_EXEC" "$SCRIPT"

EXIT_STATUS=$?
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

[ $EXIT_STATUS -eq 0 ] && echo -e "\033[1;32m[+] Done: $NSYS_OUT.nsys-rep\033[0m" || echo -e "\033[1;31m[-] Error (exit $EXIT_STATUS)\033[0m"
