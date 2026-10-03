#!/bin/bash
# 一键启动：vLLM(8000) + RAG 控制台(6006)
# 用法：bash start_all.sh
set -u

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Qwen3.8-27B-FP8}"
RAG_DIR="${RAG_DIR:-/root/autodl-tmp/rag}"
PY="${PY:-/root/autodl-tmp/rag_env/bin/python}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp}"

export VLLM_USE_FLASHINFER_SAMPLER=0      # 旧 CUDA 工具链下绕开 FlashInfer JIT 编译
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
export SMART_THRESHOLD="${SMART_THRESHOLD:-0.48}"   # 由 eval_rag.py 校准得出
mkdir -p "$TMPDIR"

pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::EngineCore"; pkill -9 -f "[a]pp.py"
sleep 3

echo "① 启动 vLLM（8000）…"
nohup vllm serve "$MODEL_DIR" \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  > "$LOG_DIR/qwen38.log" 2>&1 &

for i in $(seq 1 60); do
  sleep 10
  if curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "   ✅ vLLM 就绪（等了 $((i * 10)) 秒）"; break
  fi
  echo "   …$((i * 10))s"
done

echo "② 启动 RAG 控制台（6006）…"
cd "$RAG_DIR" || exit 1
nohup "$PY" app.py > "$LOG_DIR/rag_app.log" 2>&1 &
sleep 6

echo ""
bash "$(dirname "$0")/health.sh" 2>/dev/null || true
echo ""
echo "👉 打开云平台控制台的「自定义服务」访问网页"
