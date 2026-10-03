#!/bin/bash
# 服务健康检查：bash health.sh
RAG_DIR="${RAG_DIR:-/root/autodl-tmp/rag}"

echo "════ 服务健康检查 ════"

echo -n "vLLM  (8000): "
if curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "✅ 正常"
else
  echo "❌ 没在跑"
fi

echo -n "控制台(6006): "
code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:6006/api/status 2>/dev/null)
if [ "$code" = "200" ]; then echo "✅ 正常"; else echo "❌ 没在跑 (HTTP $code)"; fi

echo -n "vLLM 进程   : "
pgrep -f "[v]llm serve" >/dev/null && echo "存在" || echo "不存在"

echo -n "控制台进程  : "
pgrep -f "[a]pp.py" >/dev/null && echo "存在" || echo "不存在"

echo -n "索引片段数  : "
python -c "import json;print(len(json.load(open('$RAG_DIR/index.json',encoding='utf-8'))))" 2>/dev/null || echo "无"
