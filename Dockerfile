# RAG 问答应用的容器镜像
# 构建：docker build -t my-rag-assistant:latest .
# 运行：docker run -p 6006:6006 -e VLLM_URL=... my-rag-assistant:latest

FROM python:3.12-slim

# Python 行为：不缓冲输出（日志实时可见）、不缓存 pip 包
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# 系统依赖：curl 供健康检查用；build-essential 供少数无预编译包的依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- 先装依赖，再拷代码 ----
# 这样改代码时不会让依赖层缓存失效，重建只需几秒
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- 拷业务代码 ----
COPY rag.py app.py eval_rag.py ./

# 文档目录（实际使用时用 -v 挂载宿主机目录进来）
RUN mkdir -p /app/pdfs

EXPOSE 6006

# 容器健康检查：Docker 会定期调用，失败则标记为 unhealthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:6006/api/status || exit 1

CMD ["python", "app.py"]
