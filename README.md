# 私有化大模型问答系统（vLLM + RAG）

在单张 96GB 显卡上部署 27B 混合注意力模型，并基于它搭建一个带**检索路由**、**引用溯源**、**流式对话**的文档问答应用。

## 演示

| 纯对话（流式 + 思考过程） | 文档问答（带出处引用） |
|---|---|
| ![对话](screenshots/01-chat.png) | ![文档问答](screenshots/02-rag-citation.png) |

| 检索来源与相似度 | 未命中时路由到模型自身知识 |
|---|---|
| ![来源](screenshots/03-sources.png) | ![兜底](screenshots/04-fallback.png) |

**检索质量评估结果**

![评估](screenshots/05-eval.png)

## 功能

| 模块 | 说明 |
|---|---|
| **纯对话** | 流式输出（SSE）、多轮上下文、可开关「思考过程」 |
| **文档问答** | 上传 PDF → 自动切块向量化 → 检索 Top-K → 带出处回答 |
| **检索路由** | 相似度 ≥ 阈值 → 按文档回答；否则用模型自身知识兜底，并**显式标注来源** |
| **反幻觉护栏** | Prompt 强制「只依据资料」，找不到就回「资料中未提及」；**二次检查**失败则触发兜底 |
| **会话持久化** | localStorage 保存历史，刷新不丢 |

## 架构

```
浏览器（单页 HTML，零外部依赖）
  ├─ /api/chat    → SSE 流式对话（多轮上下文）
  ├─ /api/smart   → 检索 + 路由 + 兜底（文档问答）
  ├─ /api/upload  → 上传 PDF
  └─ /api/build   → 切块 + 向量化 + 建索引
        ↓
  FastAPI 后端
        ├─ 向量检索（numpy 余弦，25 片段 / 512 维 / **3ms**）
        └─ vLLM OpenAI API @ 127.0.0.1:8000
                ↓
        Qwen3.8-27B-FP8 @ RTX PRO 6000 96GB
```

## 技术栈

- **推理**：vLLM 0.30.0 · PyTorch 2.13+cu130 · CUDA 12.8 · MTP 投机解码
- **后端**：FastAPI · uvicorn · urllib（零重依赖）
- **检索**：sentence-transformers (`BAAI/bge-small-zh-v1.5`, CPU) · numpy · pypdf
- **前端**：原生 HTML/CSS/JS（无框架、不连 CDN）

---

## 一、模型部署与容量规划

**硬件**：1 × RTX PRO 6000 Blackwell 96GB（AutoDL 容器实例）

**显存账（实测）**

| 组成 | 占用 |
|---|---|
| 权重 + non-torch | 28.08 GiB |
| KV cache | 53.40 GiB |
| 激活峰值 + CUDA graph | 4.92 GiB |
| **合计** | **86.4 GiB = 96 × 0.90**（`--gpu-memory-utilization`） |

**KV cache 建模**

模型为**混合注意力**架构：64 层中仅 **16 层全注意力**（`full_attention_interval=4`），其余 48 层为线性注意力（Gated DeltaNet）。

```
理论（仅全注意力）: 2 × 16层 × 4 KV头 × 256 dim × 2B = 64 KiB/token
实测              :                              93 KiB/token  (1.45×)
```

**差额 45% 反推**：`29 KiB/token × 602,112 tokens ÷ 752 块 ÷ 48 层 ≈ 507 KiB/层/块` ≈ `4 × 256 × 256 × 2B = 512 KiB` —— 即**线性注意力层的循环状态矩阵**，它按 800-token 的块被统一 KV 池分摊到每个 token。

> **结论**：混合注意力模型的显存规划**不能只算 attention KV**，必须计入线性层的固定循环状态，否则会低估 45%。

**容量**

```
KV 池总量: 602,112 tokens
  ├─ 32K  上下文 → 18.4 并发
  ├─ 131K 上下文 →  4.6 并发
  └─ 262K 上下文 →  2.3 并发
```

---

## 二、性能压测

`vllm bench serve`，1024 in / 256 out，32 请求，单卡 TP1。

| 并发 | 输出吞吐 | TTFT(均值) | TPOT(均值) | E2EL(均值) |
|---|---|---|---|---|
| **1** | 80 tok/s | 134 ms | **12.01 ms** | 3.2 s |
| **8** | **413 tok/s** | 339 ms | 15.68 ms | 4.3 s |
| **32** | 515 tok/s | **2395 ms** | 35.50 ms | 11.4 s |

**结论**：
- 吞吐拐点在 **8 并发**：8→32 吞吐仅 +24%，但 TPOT 涨 126%、TTFT 涨 7 倍
- 若追求低延迟，**多副本（每副本 8 并发）优于单实例堆高并发**

**MTP 投机解码**（`num_speculative_tokens=3`）

| 并发 | 接受率 | 接受长度 | P0 | P1 | P2 |
|---|---|---|---|---|---|
| 1 | 52.8% | 2.58 | 65.0% | 51.1% | 42.3% |
| 32 | 48.4% | 2.45 | 61.2% | 46.0% | 38.1% |

**Per-position 接受率衰减很陡（65%→51%→42%）**，定量解释了 `num_speculative_tokens=3` 是最优值：第 4 位收益已低于 draft 成本。

---

## 三、RAG 检索质量评估

自建 12 条评测集（10 正样本 + 2 负样本），自动化评估脚本 `eval_rag.py`。

| 指标 | 结果 |
|---|---|
| **Recall@4** | **10/10 = 100%** |
| **MRR** | **0.900** |
| 正样本最高分 | min 0.503 / 平均 0.607 |
| 负样本最高分 | max 0.462 / 平均 0.401 |
| **检索延迟** | **3 ms** |

**阈值校准（用数据修复的缺陷）**

```
正样本最低分 0.503  |  负样本最高分 0.462   → 分界线清晰
初始阈值 0.35 → 校准为 0.48
```

**修复的问题**：原阈值 0.35 时，负样本「vLLM 的连续批处理怎么实现」（最高分 0.462）会被**误判为命中文档**，导致系统拿无关的制度文档去回答技术问题。

**容量判断的实证**：25 片段 / 512 维暴力算余弦仅 **3ms**，验证了该规模**无需向量数据库**；触发换库的信号是片段量到 10 万级、需要增量更新或元数据过滤时。

---

## 四、关键实现决策

| 决策 | 理由 |
|---|---|
| 切块 800 字 / 重叠 150 字 | 太小丢上下文，太大稀释语义；重叠防止答案被切断 |
| embedding 强制跑 CPU | GPU 上已有 27B 占 86GiB，不抢显存；几千片段 CPU 几秒完成 |
| **向量库用 numpy 文件** | 实测 3ms，零依赖零运维；规模上去再换 FAISS/pgvector |
| **阈值粗筛 + 二次检查** | 相似度绝对值不可靠（相关 0.5 / 半相关 0.37），双保险 |
| **兜底回答显式标注** | 用户会默认"能答就是有依据"，不标注在企业场景是合规问题 |
| **用代码判定是否用资料** | 若让模型自己决定，它会假装资料里有（幻觉来源） |
| 上下文只带最近 12 条 | 滑动窗口，防止 token 爆炸、成本失控、超上下文 |

---

## 五、排障记录

**问题**：vLLM 启动时 `EngineCore failed to start`，报
`nvcc fatal : Unknown option '--compress-mode=size'`

**根因**：镜像自带 nvcc 12.4，而新版本 vLLM 的 FlashInfer 按 CUDA 12.9+ 生成 JIT 编译参数，导致采样 kernel 编译失败。

**定位路径**：启动日志 → `warmup_kernels → sample_tokens` → FlashInfer JIT → nvcc 版本不匹配。
同时发现 `VLLM_USE_V1` / `VLLM_ATTENTION_BACKEND` / `VLLM_DISABLE_FLASHINFER` 三个环境变量在新版 vLLM 中**已被移除**（日志报 `Unknown vLLM environment variable`），照抄旧教程无效。

**解决**：
1. 短期：`VLLM_USE_FLASHINFER_SAMPLER=0` 绕开 FlashInfer 采样器
2. 根本：选用 CUDA ≥12.8 的镜像 + 让 vLLM 自带 CUDA 13 依赖（pip 装的 nvcc 13.4 满足 ≥12.9）
3. 验证：`torch.cuda.get_device_capability()` 返回 `(12, 0)` 确认 Blackwell 支持

---

## 六、运行

```bash
# 1) 启动 vLLM（8000）
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve /path/to/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

# 2) 启动 Web 应用（6006）
python -m venv --system-site-packages rag_env && source rag_env/bin/activate
pip install sentence-transformers pypdf python-multipart

python app.py                    # 对话 + 文档问答
python eval_rag.py -k 4          # 检索质量评估
```

## 依赖

```
fastapi
uvicorn
python-multipart
sentence-transformers
pypdf
numpy
```

## 后续规划

- [ ] Rerank 重排（`bge-reranker-base`）：召回 20 → 精排 4，目标 MRR 0.900 → 0.95+
- [ ] 混合检索（向量 + BM25）：改善专有名词召回
- [ ] 生成层评估（LLM-as-judge）：答案正确性、忠实度、幻觉率
- [ ] 扩充评测集至 30~50 条，增加跨文档检索场景
- [ ] Excel/CSV 表格文档支持（结构化查询优先于向量检索）
