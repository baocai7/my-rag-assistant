#!/usr/bin/env python3
"""Qwen 控制台：纯对话（流式）+ 文档问答（RAG，带检索路由与兜底）

启动：
    export VLLM_URL=http://127.0.0.1:8000/v1/chat/completions
    python app.py                 # 默认监听 0.0.0.0:6006

接口：
    GET  /            单页界面
    GET  /api/status  索引状态
    POST /api/upload  上传 pdf/txt/md
    POST /api/build   重建索引
    POST /api/chat    纯对话（SSE 流式）
    POST /api/smart   文档问答（检索路由 + 兜底）
"""
import glob, json, os, shutil, urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag import (VLLM_MODEL, VLLM_URL, build_index, call_llm, embed,
                 format_context, load_index, retrieve)

HERE    = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(HERE, "pdfs")
PORT    = int(os.environ.get("RAG_PORT", "6006"))

# 检索路由阈值：由 eval_rag.py 在自建评测集上校准
#   正样本最低分 0.503 / 负样本最高分 0.462  →  取中间 0.48
SMART_THRESHOLD = float(os.environ.get("SMART_THRESHOLD", "0.48"))

os.makedirs(DOC_DIR, exist_ok=True)
STATE = {"vecs": None, "chunks": []}

HTML = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Qwen 控制台</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0;font:15px/1.65 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
     background:#f5f6f8;color:#1f2328;display:flex;flex-direction:column;height:100vh}
.top{display:flex;align-items:center;gap:8px;padding:10px 16px;background:#fff;border-bottom:1px solid #e5e7eb}
.brand{font-weight:600;margin-right:12px}
.tab{padding:7px 16px;border-radius:8px;cursor:pointer;font-size:14px;color:#4b5563}
.tab.on{background:#4f7cff;color:#fff}
.spacer{flex:1}
.mini{border:1px solid #d7dbe3;background:#fff;border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
.mini.on{background:#eef3ff;border-color:#4f7cff;color:#2b4fd0}
.wrap{flex:1;display:flex;min-height:0}
.side{width:290px;background:#fff;border-right:1px solid #e5e7eb;padding:16px;
      display:none;flex-direction:column;gap:12px;overflow:auto}
.side.show{display:flex}
.main{flex:1;display:flex;flex-direction:column;min-width:0}
.chat{flex:1;overflow:auto;padding:24px 32px;display:flex;flex-direction:column;gap:16px}
.empty{color:#9aa3af;text-align:center;margin-top:14vh}
.q{align-self:flex-end;background:#4f7cff;color:#fff;padding:10px 15px;
   border-radius:14px 14px 3px 14px;max-width:70%;white-space:pre-wrap}
.a{align-self:flex-start;background:#fff;border:1px solid #e5e7eb;
   border-radius:14px 14px 14px 3px;padding:14px 18px;max-width:80%;white-space:pre-wrap}
.a .cite{background:#fff3cd;border-radius:4px;padding:0 4px;font-size:13px;color:#92400e}
.think{background:#f7f8fa;border-left:3px solid #cbd5e1;padding:8px 12px;margin-bottom:10px;
       font-size:13px;color:#6b7280;white-space:pre-wrap}
.think summary{cursor:pointer;color:#4b5563}
.tag{display:inline-block;border-radius:6px;padding:2px 8px;font-size:12px;margin-bottom:8px}
.tagdoc{background:#e7f6ec;color:#116329}
.tagmodel{background:#fff3cd;color:#92400e}
.tagnone{background:#eef1f6;color:#4b5563}
.src{margin-top:12px;border-top:1px dashed #e5e7eb;padding-top:10px}
.src summary{cursor:pointer;color:#6b7280;font-size:13px}
.sitem{background:#fafbfc;border:1px solid #eef0f3;border-radius:8px;padding:8px 10px;
       margin-top:8px;font-size:13px;white-space:normal}
.sitem b{color:#0f766e}
.muted{color:#6b7280;font-size:13px}
.drop{border:2px dashed #cbd5e1;border-radius:10px;padding:20px 12px;text-align:center;
      cursor:pointer;background:#fafbfc}
.drop:hover,.drop.on{border-color:#4f7cff;background:#eef3ff}
.btn{border:0;border-radius:8px;padding:9px 14px;background:#4f7cff;color:#fff;
     font-size:14px;cursor:pointer;width:100%}
.btn:disabled{background:#c7d0e0}
.btn.ghost{background:#eef1f6;color:#374151}
.flist{font-size:13px;display:flex;flex-direction:column;gap:6px}
.fitem{background:#f7f8fa;border-radius:6px;padding:6px 9px;word-break:break-all}
.log{font-size:12px;color:#0f766e;min-height:18px}
.bar{border-top:1px solid #e5e7eb;background:#fff;padding:14px 22px;display:flex;gap:10px}
.bar input{flex:1;border:1px solid #d7dbe3;border-radius:9px;padding:11px 14px;font-size:15px;outline:0}
.bar input:focus{border-color:#4f7cff}
.bar select{border:1px solid #d7dbe3;border-radius:9px;padding:0 8px;background:#fff}
.bar button{width:auto;padding:0 22px;font-size:15px}
</style></head><body>

<div class="top">
  <span class="brand">🧠 Qwen 控制台</span>
  <div class="tab on" data-m="chat">💬 对话</div>
  <div class="tab" data-m="rag">📚 文档问答</div>
  <div class="spacer"></div>
  <button class="mini" id="think">思考：关</button>
  <button class="mini on" id="fb">兜底：开</button>
  <button class="mini" id="clear">新对话</button>
</div>

<div class="wrap">
  <div class="side" id="side">
    <div class="muted">上传 PDF → 建索引 → 提问（答案只来自你的文档）</div>
    <div class="drop" id="drop">＋ 点击/拖拽 PDF 到这里
      <input type="file" id="file" accept=".pdf,.txt,.md" multiple hidden></div>
    <button class="btn ghost" id="build">重建索引</button>
    <div class="log" id="log"></div>
    <div class="muted" id="stat">索引：加载中…</div>
    <div class="flist" id="flist"></div>
  </div>

  <div class="main">
    <div class="chat" id="chat"></div>
    <div class="bar">
      <input id="q" placeholder="说点什么…" autocomplete="off">
      <select id="k"><option value="4">Top 4</option><option value="6">Top 6</option><option value="8">Top 8</option></select>
      <button class="btn" id="send">发送</button>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const chatEl = $('#chat');
let mode = 'chat', thinking = false, fallback = true;
const hist = {chat: [], rag: []};

const esc  = s => (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const cite = s => s.replace(/\\[([^\\[\\]]{0,80}?p\\.\\d+)\\]/g, '<span class="cite">[$1]</span>');
const bottom = () => chatEl.scrollTop = chatEl.scrollHeight;

async function toJSON(r){
  const txt = await r.text();
  try { return JSON.parse(txt); }
  catch(e){
    throw new Error('HTTP ' + r.status + ' 返回了非 JSON（后端挂了或代理超时）。前 150 字：'
                    + txt.slice(0, 150));
  }
}

/* ---------- 本地持久化 ---------- */
const LSKEY = 'qwen_hist_v1';
function saveHist(){
  try {
    hist.chat = hist.chat.slice(-40);
    hist.rag  = hist.rag.slice(-40);
    localStorage.setItem(LSKEY, JSON.stringify(hist));
  } catch(e){}
}
function loadHist(){
  try {
    const d = JSON.parse(localStorage.getItem(LSKEY) || 'null');
    if (d && d.chat) hist.chat = d.chat;
    if (d && d.rag)  hist.rag  = d.rag;
  } catch(e){}
}
setInterval(saveHist, 3000);
window.addEventListener('beforeunload', saveHist);

/* ---------- 渲染 ---------- */
function bubble(cls, html){
  const d = document.createElement('div');
  d.className = cls; d.innerHTML = html;
  chatEl.appendChild(d); bottom(); return d;
}
function renderAll(){
  chatEl.innerHTML = '';
  const h = hist[mode];
  if (!h.length){
    chatEl.innerHTML = '<div class="empty">'
      + (mode === 'chat' ? '直接开始对话 👇' : '先在左侧上传 PDF 并重建索引 👈') + '</div>';
    return;
  }
  h.forEach(m => {
    if (m.role === 'user'){ bubble('q', esc(m.content)); return; }
    bubble('a', (m.reasoning
        ? '<details class="think"><summary>思考过程</summary>' + esc(m.reasoning) + '</details>' : '')
      + cite(esc(m.content)) + (m.src || ''));
  });
}
function setMode(m){
  mode = m;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.m === m));
  $('#side').classList.toggle('show', m === 'rag');
  $('#k').style.display = m === 'rag' ? '' : 'none';
  $('#q').placeholder = m === 'rag' ? '问文档里的内容…' : '说点什么…';
  renderAll();
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => setMode(t.dataset.m));

$('#think').onclick = () => {
  thinking = !thinking;
  $('#think').textContent = '思考：' + (thinking ? '开' : '关');
  $('#think').classList.toggle('on', thinking);
};
$('#fb').onclick = () => {
  fallback = !fallback;
  $('#fb').textContent = '兜底：' + (fallback ? '开' : '关');
  $('#fb').classList.toggle('on', fallback);
};
$('#clear').onclick = () => {
  if (hist[mode].length && !confirm('清空当前模式的对话记录？')) return;
  hist[mode] = []; saveHist(); renderAll();
};

/* ---------- 纯对话（SSE 流式） ---------- */
async function sendChat(text){
  hist.chat.push({role:'user', content:text});
  if (chatEl.querySelector('.empty')) chatEl.innerHTML = '';
  bubble('q', esc(text));
  const holder = bubble('a', '<span class="muted">思考中…</span>');
  let think = '', content = '';
  const draw = () => {
    holder.innerHTML = (think
        ? '<details class="think" open><summary>思考过程</summary>' + esc(think) + '</details>' : '')
      + cite(esc(content || '…'));
    bottom();
  };
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages: hist.chat.slice(-12), thinking})});
    if (!r.ok){ holder.innerHTML = '<span style="color:#b91c1c">HTTP ' + r.status + '</span>'; return; }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    while (true){
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\\n'); buf = lines.pop();
      for (const line of lines){
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        let j; try { j = JSON.parse(data); } catch(e){ continue; }
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const t = d.reasoning_content || d.reasoning;
        if (t) { think += t; draw(); }
        if (d.content) { content += d.content; draw(); }
      }
    }
  } catch(e){ holder.innerHTML = '<span style="color:#b91c1c">' + esc(e.message) + '</span>'; }
  hist.chat.push({role:'assistant', content, reasoning:think});
  saveHist(); draw();
}

/* ---------- 文档问答（检索路由 + 兜底） ---------- */
async function sendRag(text){
  bubble('q', esc(text));
  const holder = bubble('a', '<span class="muted">检索中…</span>');
  try {
    const r = await fetch('/api/smart', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:text, k:+$('#k').value, fallback})});
    const d = await toJSON(r);
    if (!r.ok) throw new Error(d.detail || '失败');

    const tag = d.mode === 'doc'
      ? '<span class="tag tagdoc">📚 来自文档 · 最高相似度 ' + d.best_score.toFixed(3) + '</span>'
      : (d.mode === 'model'
         ? '<span class="tag tagmodel">🧠 来自模型自身知识 · 未引用你的文档</span>'
         : '<span class="tag tagnone">📭 文档中未找到</span>');

    const src = d.hits.map(h => '<div class="sitem"><b>' + h.score.toFixed(3) + '</b> · '
      + esc(h.source) + (h.page ? ' p.' + h.page : '')
      + '<div class="muted">' + esc(h.text) + '…</div></div>').join('');
    const srcHtml = '<details class="src"><summary>查看 ' + d.hits.length
      + ' 个检索片段（阈值 ' + d.threshold + '）</summary>' + src + '</details>';

    holder.innerHTML = tag + '<div>' + cite(esc(d.answer)) + '</div>' + srcHtml;
    hist.rag.push({role:'user', content:text},
                  {role:'assistant', content:d.answer, src:tag + srcHtml});
    saveHist();
  } catch(e){ holder.innerHTML = '<span style="color:#b91c1c">' + esc(e.message) + '</span>'; }
}

function send(){
  const q = $('#q').value.trim(); if (!q) return;
  $('#q').value = '';
  mode === 'chat' ? sendChat(q) : sendRag(q);
}
$('#send').onclick = send;
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

/* ---------- 索引管理 ---------- */
async function refresh(){
  try {
    const d = await toJSON(await fetch('/api/status'));
    $('#stat').textContent = '索引：' + d.chunks + ' 个片段 / ' + d.files.length + ' 个文件';
    $('#flist').innerHTML = d.files.map(f => '<div class="fitem">' + esc(f) + '</div>').join('');
  } catch(e){ $('#stat').textContent = '索引：读取失败'; }
}
async function build(){
  $('#build').disabled = true; $('#log').textContent = '正在切块 + 向量化（CPU）…';
  try {
    const d = await toJSON(await fetch('/api/build', {method:'POST'}));
    $('#log').textContent = '✅ 索引完成：' + d.files + ' 个文件 → ' + d.chunks + ' 个片段';
  } catch(e){ $('#log').textContent = '❌ ' + e.message; }
  $('#build').disabled = false; refresh();
}
async function upload(files){
  for (const f of files){
    $('#log').textContent = '上传中：' + f.name + ' …';
    const fd = new FormData(); fd.append('file', f);
    const d = await toJSON(await fetch('/api/upload', {method:'POST', body: fd}));
    if (d.detail){ $('#log').textContent = '上传失败：' + d.detail; return; }
  }
  $('#log').textContent = '上传完成，重建索引中…';
  build();
}
$('#drop').onclick = () => $('#file').click();
$('#file').onchange = e => upload([...e.target.files]);
$('#drop').ondragover = e => { e.preventDefault(); $('#drop').classList.add('on'); };
$('#drop').ondragleave = () => $('#drop').classList.remove('on');
$('#drop').ondrop = e => {
  e.preventDefault(); $('#drop').classList.remove('on');
  upload([...e.dataTransfer.files]);
};
$('#build').onclick = build;

loadHist(); setMode('chat'); refresh();
</script></body></html>
"""


# ---------------------------------------------------------------- 生命周期
@asynccontextmanager
async def lifespan(_app):
    STATE["vecs"], STATE["chunks"] = load_index()
    print(f"[启动] 索引：{len(STATE['chunks'])} 个片段")
    try:
        embed(["预热"]); print("[启动] embedding 模型就绪")
    except Exception as e:
        print("[启动] embedding 预热失败：", e)
    print(f"[启动] 检索阈值：{SMART_THRESHOLD}")
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- 模型
class ChatIn(BaseModel):
    messages: list
    thinking: bool = False
    max_tokens: int = 2048
    temperature: float = 0.7


class AskIn(BaseModel):
    question: str
    k: int = 4
    fallback: bool = True


# ---------------------------------------------------------------- 路由
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/status")
def status():
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DOC_DIR, "*")))
    return {"chunks": len(STATE["chunks"]), "files": files,
            "threshold": SMART_THRESHOLD}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "只支持 pdf / txt / md")
    dest = os.path.join(DOC_DIR, os.path.basename(file.filename))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "file": os.path.basename(dest)}


@app.post("/api/build")
def build():
    nf, nc = build_index(DOC_DIR)
    if nc == 0:
        raise HTTPException(400, "pdfs/ 目录里没有可解析的文件")
    STATE["vecs"], STATE["chunks"] = load_index()
    return {"files": nf, "chunks": nc}


@app.post("/api/chat")
def chat(inp: ChatIn):
    """纯对话：把 vLLM 的 SSE 流原样转发给浏览器"""
    body = {"model": VLLM_MODEL, "messages": inp.messages,
            "max_tokens": inp.max_tokens, "temperature": inp.temperature, "stream": True}
    if not inp.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(VLLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    def gen():
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    yield line
        except Exception as e:
            payload = {"choices": [{"delta": {"content": f"\n[错误] {e}"}}]}
            yield ("data: " + json.dumps(payload) + "\n\n").encode()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/smart")
def smart(inp: AskIn):
    """命中文档 → 按文档答；没命中（或模型说资料不足）→ 用模型自身知识兜底"""
    if STATE["vecs"] is None or not STATE["chunks"]:
        raise HTTPException(400, "还没有索引，请先上传文档并重建索引")

    hits = retrieve(inp.question, STATE["chunks"], STATE["vecs"], inp.k)
    best = hits[0][1] if hits else -1.0
    thr = SMART_THRESHOLD

    detail = [{"source": c["source"], "page": c["page"],
               "score": s, "text": c["text"][:300]} for c, s in hits]

    # ---- 路线 1：命中文档 ----
    if best >= thr:
        prompt = ("你是文档问答助手。只依据下面的【资料】回答问题，不要编造。"
                  "如果资料里确实找不到答案，就只回复：资料中未提及。\n\n"
                  f"【资料】\n{format_context(hits)}\n\n【问题】{inp.question}")
        try:
            ans = call_llm([{"role": "user", "content": prompt}], temperature=0.2).strip()
        except Exception as e:
            raise HTTPException(502, f"调用 vLLM 失败：{e}")
        # 二次检查：模型自己说资料不够 → 也走兜底
        if (not inp.fallback) or ("未提及" not in ans):
            return {"mode": "doc", "answer": ans, "hits": detail,
                    "best_score": best, "threshold": thr}

    # ---- 路线 2：兜底到模型自身知识 ----
    if not inp.fallback:
        return {"mode": "none", "hits": detail, "best_score": best, "threshold": thr,
                "answer": f"文档中没有找到相关内容（最高相似度 {best:.3f}，阈值 {thr}）。"
                          "可以换个说法，或把相关文档加进知识库。"}

    sysmsg = ("你是通用 AI 助手。注意：以下回答不来自用户的文档，"
              "请基于你自己的知识回答，并在结尾明确提示这一点。")
    try:
        ans = call_llm([{"role": "system", "content": sysmsg},
                        {"role": "user", "content": inp.question}], temperature=0.7)
    except Exception as e:
        raise HTTPException(502, f"调用 vLLM 失败：{e}")
    return {"mode": "model", "answer": ans.strip(), "hits": detail,
            "best_score": best, "threshold": thr}


if __name__ == "__main__":
    import uvicorn
    print(f"控制台: http://0.0.0.0:{PORT}   → vLLM: {VLLM_URL}")
    print(f"检索阈值: {SMART_THRESHOLD}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
