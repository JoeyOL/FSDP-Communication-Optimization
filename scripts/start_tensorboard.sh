# 检查端口是否被占用
PORT=6006
if lsof -i :$PORT | grep LISTEN; then
    echo "⚠️ 端口 $PORT 已被占用，正在释放..."
    lsof -ti :$PORT | xargs -r kill -9
    sleep 2
    echo "✅ 端口 $PORT 已释放。"
fi

# 在后台启动 TensorBoard
tensorboard --logdir /root/llama-7b/fsdp_output/logs --host 0.0.0.0 --port $PORT &
TENSORBOARD_PID=$!
echo "📊 TensorBoard 已在后台启动 (PID: ${TENSORBOARD_PID})。访问 http://<your-ip>:${PORT}"