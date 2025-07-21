#!/bin/bash

# 设置错误处理：任何命令失败则退出
set -e

HOST="0.0.0.0"
PORT=8000
BASE_ROOT=$(dirname "$(readlink -f "$0")")
IS_INSTALLED="$BASE_ROOT/installed"
PID_FILE="$BASE_ROOT/run.pid"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# 检查 requirements.txt 是否存在
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi
# 检查关键依赖是否安装（以 fastapi 为例）
if [ ! -f "$IS_INSTALLED" ]; then
    pip3 install -r requirements.txt > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies from requirements.txt."
        exit 1
    fi
    touch installed
fi
function is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # 已运行
        else
            # PID 文件存在但进程没了，清理
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # 未运行
}

function start() {
    if is_running; then
        echo "服务已运行 (PID $(cat "$PID_FILE"))"
        exit 0
    fi
    gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind $HOST:$PORT --pid "$PID_FILE" --daemon
    sleep 1
    if is_running; then
        echo "服务启动成功"
        exit 0
    else
        rm -fr "$PID_FILE"
        echo "服务启动失败"
        exit 1
    fi
}

function stop() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "Stopping service (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "服务已停止"
    else
        echo "服务未运行"
    fi
}

# 平滑重载
reload() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "Reloading service (PID $PID)..."
        kill -HUP "$PID"
        echo "Service reloaded."
    else
        echo "Service not running, starting..."
        start
    fi
}

# 重启服务
restart() {
    echo "Restarting service..."
    stop
    sleep 1
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    reload)
        reload
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|reload|restart}"
        exit 1
esac