#!/usr/bin/env python3
"""RAG 检索层评估：Recall@k / MRR / 相似度分布 / 阈值建议

用法：
    1. 把下面 CASES 换成你自己文档的问题
    2. python eval_rag.py -k 4
    3. 结果同时打印到终端并保存为 eval_result.json

为什么只评检索层：RAG 的质量上限由**检索**决定；检索层评估全自动、客观、不需要人工标注答案，
而生成层需要 LLM-as-judge 或人工评分。先保证检索，再谈生成。
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rag import embed, load_index            # noqa: E402

# ==================== 测试集（换成你自己文档的问题）====================
# q    : 问题
# file : 正确答案所在文件名中的关键词；None 表示"文档里没有"的负样本
# page : 正确答案页码，不确定就写 None（只要文件名对就算命中）
CASES = [
    # ---- 正样本：文档里能查到 ----
    {"q": "员工年假有多少天？",           "file": "员工手册", "page": 12},
    {"q": "试用期最长是多久？",           "file": "员工手册", "page": 8},
    {"q": "报销流程需要哪几级审批？",     "file": "员工手册", "page": 21},
    {"q": "加班费怎么计算？",             "file": "员工手册", "page": 15},
    {"q": "离职需要提前多久提出？",       "file": "员工手册", "page": 30},
    {"q": "公司的考勤制度是怎样的？",     "file": "员工手册", "page": 10},
    {"q": "差旅住宿标准是多少？",         "file": "员工手册", "page": 23},
    {"q": "保密协议的期限是多久？",       "file": "员工手册", "page": 35},
    # ---- 负样本：文档里没有（验证兜底路由是否该触发）----
    {"q": "什么是 PagedAttention？",      "file": None, "page": None},
    {"q": "vLLM 的连续批处理怎么实现？",  "file": None, "page": None},
]
# =====================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=4, help="检索 Top-K")
    ap.add_argument("--threshold", type=float, default=0.48,
                    help="当前线上阈值，用于对比")
    args = ap.parse_args()

    vecs, chunks = load_index()
    if vecs is None:
        sys.exit("还没有索引，先跑：python rag.py build ./pdfs")
    embed(["预热"])                                  # 预热 embedding 模型

    k = args.k
    hit, rr = 0, []
    pos_best, neg_best, rows = [], [], []

    print(f"\n评估开始：{len(CASES)} 条用例，k={k}\n" + "─" * 72)
    for c in CASES:
        t0 = time.time()
        qv = embed([c["q"]])[0]
        scores = vecs @ qv
        order = np.argsort(-scores)[:k]
        ms = (time.time() - t0) * 1000

        best = float(scores[order[0]])
        got = [(chunks[i]["source"], chunks[i]["page"], float(scores[i])) for i in order]

        if c["file"] is None:                        # 负样本
            ok = best < args.threshold
            neg_best.append(best)
            rank = None
            tag = f"负样本({'正确挡住' if ok else '误判为命中'})"
        else:                                        # 正样本
            pos_best.append(best)
            rank = next((r for r, (s, p, _) in enumerate(got, start=1)
                         if c["file"] in s and (c["page"] is None or p == c["page"])), None)
            ok = rank is not None
            rr.append(1.0 / rank if rank else 0.0)
            tag = f"命中 rank={rank}" if rank else "未命中"

        hit += 1 if ok else 0
        rows.append({"q": c["q"], "ok": ok, "best": best, "rank": rank, "ms": ms})

        print(f"{'✅' if ok else '❌'} {c['q'][:26]:<28} 最高分={best:.3f}  {tag}  ({ms:.0f}ms)")
        print("     top%d: " % k + " | ".join(
            f"{s[-22:]} p.{p}" for s, p, _ in got[:3]))

    n_pos, n_neg = len(pos_best), len(neg_best)
    print("─" * 72)
    print("\n【汇总】")
    print(f"  用例总数        : {len(CASES)}  (正样本 {n_pos} / 负样本 {n_neg})")
    print(f"  命中率 @ {k}       : {hit}/{len(CASES)} = {hit / len(CASES) * 100:.1f}%")
    if n_pos:
        print(f"  正样本 Recall@{k} : {sum(1 for r in rr if r > 0)}/{n_pos} = "
              f"{sum(1 for r in rr if r > 0) / n_pos * 100:.1f}%")
        print(f"  MRR             : {np.mean(rr):.3f}")
        print(f"  正样本最高分    : min={min(pos_best):.3f}  平均={np.mean(pos_best):.3f}")
    if n_neg:
        print(f"  负样本最高分    : max={max(neg_best):.3f}  平均={np.mean(neg_best):.3f}")
    print(f"  检索平均耗时    : {np.mean([r['ms'] for r in rows]):.0f} ms")

    if n_pos and n_neg:
        lo, hi = min(pos_best), max(neg_best)
        print("\n【阈值建议】")
        print(f"  当前线上阈值    : {args.threshold}")
        print(f"  正样本最低分    : {lo:.3f}   负样本最高分: {hi:.3f}")
        if lo > hi:
            print(f"  ✅ 存在清晰分界线 → 建议阈值取中间: {(lo + hi) / 2:.3f}")
        else:
            print(f"  ⚠️ 分数有重叠（{hi:.3f} > {lo:.3f}）→ 单靠阈值分不开，"
                  f"需要 Rerank 或混合检索改善")

    out = os.path.join(HERE, "eval_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"k": k, "threshold": args.threshold,
                   "hit_rate": hit / len(CASES),
                   "mrr": float(np.mean(rr)) if rr else None,
                   "pos_best": pos_best, "neg_best": neg_best, "detail": rows},
                  f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out}")
    if rr:
        print("\n📌 可以写进简历的一句话：")
        print(f"   「在 {len(CASES)} 条自建评测集上，检索 Recall@{k} = "
              f"{sum(1 for r in rr if r > 0)}/{n_pos}，MRR = {np.mean(rr):.3f}」")


if __name__ == "__main__":
    main()
