#!/usr/bin/env python3
"""RAG 核心：文档解析 → 切块 → 向量化 → 检索 → 生成

既可作为库被 app.py / eval_rag.py 引用，也可单独跑：
    python rag.py build ./pdfs       # 建索引
    python rag.py ask "你的问题"      # 命令行提问
"""
import argparse, glob, json, os, sys, urllib.request

import numpy as np

WORK  = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(WORK, "index.npz")
META  = os.path.join(WORK, "index.json")

# embedding 模型：优先用本地目录（离线可用），否则从 HuggingFace 拉取
LOCAL_EMB = os.environ.get("LOCAL_EMB", os.path.expanduser("~/autodl-tmp/bge-small-zh"))
EMB_MODEL = os.environ.get("EMB_MODEL") or (
    LOCAL_EMB if os.path.isdir(LOCAL_EMB) else "BAAI/bge-small-zh-v1.5")

VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b")
VLLM_URL   = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")

CHUNK, OVERLAP = 800, 150          # 每块 800 字，重叠 150 字

_emb = None


# ---------------------------------------------------------------- 文本处理
def chunk_text(text, size=CHUNK, overlap=OVERLAP):
    """按字符滑窗切块。重叠区避免答案正好被切在边界上。"""
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + size]
        if len(piece) > 30:
            out.append(piece)
        if i + size >= len(text):
            break
        i += size - overlap
    return out


def read_pdf(path):
    """返回 [(页码, 文本), ...]；页码从 1 开始"""
    from pypdf import PdfReader
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]


def collect_chunks(doc_dir):
    """扫描目录，返回 (文件列表, [{source, page, text}])"""
    files = []
    for pat in ("*.pdf", "*.txt", "*.md"):
        files += glob.glob(os.path.join(doc_dir, pat))
    files.sort()

    chunks = []
    for f in files:
        name = os.path.basename(f)
        if f.lower().endswith(".pdf"):
            for page, txt in read_pdf(f):
                for c in chunk_text(txt):
                    chunks.append({"source": name, "page": page, "text": c})
        else:
            txt = open(f, encoding="utf-8", errors="ignore").read()
            for c in chunk_text(txt):
                chunks.append({"source": name, "page": None, "text": c})
    return files, chunks


# ---------------------------------------------------------------- 向量化
def embed(texts):
    """向量化。强制 CPU —— GPU 上跑着大模型，不抢显存。"""
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer(EMB_MODEL, device="cpu")
    v = _emb.encode(texts, batch_size=32, normalize_embeddings=True,
                    show_progress_bar=len(texts) > 64)
    return np.asarray(v, dtype=np.float32)


# ---------------------------------------------------------------- 索引
def build_index(doc_dir):
    """建索引，返回 (文件数, 片段数)"""
    files, chunks = collect_chunks(doc_dir)
    if not chunks:
        return len(files), 0
    vecs = embed([c["text"] for c in chunks])
    np.savez_compressed(INDEX, vecs=vecs)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    return len(files), len(chunks)


def load_index():
    """返回 (向量矩阵, 片段列表)；没有索引时返回 (None, [])"""
    if not (os.path.exists(INDEX) and os.path.exists(META)):
        return None, []
    return np.load(INDEX)["vecs"], json.load(open(META, encoding="utf-8"))


def retrieve(question, chunks, vecs, k=4):
    """向量检索。归一化后点积 = 余弦相似度。返回 [(chunk, score), ...] 降序"""
    qv = embed([question])[0]
    scores = vecs @ qv
    order = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def format_context(hits):
    out = []
    for c, _ in hits:
        loc = c["source"] + (f" p.{c['page']}" if c["page"] else "")
        out.append(f"[{loc}]\n{c['text']}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- 生成
def call_llm(messages, temperature=0.2, max_tokens=1024, thinking=False):
    """调用 vLLM 的 OpenAI 兼容接口（非流式）"""
    body = {"model": VLLM_MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


RAG_PROMPT = ("你是文档问答助手。只依据下面的【资料】回答问题，不要编造。"
              "如果资料里找不到答案，就直接说「资料中未提及」。"
              "回答末尾用 [文件名 p.页码] 标注依据。\n\n"
              "【资料】\n{ctx}\n\n【问题】{q}")


# ---------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("path")
    a = sub.add_parser("ask")
    a.add_argument("question"); a.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "build":
        nf, nc = build_index(args.path)
        print(f"✅ {nf} 个文件 → {nc} 个片段")
        return

    vecs, chunks = load_index()
    if vecs is None:
        sys.exit("还没建索引：python rag.py build ./pdfs")
    hits = retrieve(args.question, chunks, vecs, args.k)
    print("\n=== 检索片段 ===")
    for c, s in hits:
        print(f"  {s:.3f}  {c['source']} p.{c['page']}")
    print("\n=== 回答 ===")
    print(call_llm([{"role": "user",
                     "content": RAG_PROMPT.format(ctx=format_context(hits),
                                                  q=args.question)}]).strip())


if __name__ == "__main__":
    _cli()
