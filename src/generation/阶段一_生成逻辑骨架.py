# -*- coding: utf-8 -*-
"""
阶段一 · 约束内生成逻辑骨架（LLM + MCTS 接口层）
==================================================================
依赖：阶段零_聚宽基建骨架.py（须在同一研究环境 Notebook 中先执行，提供
      BaseAlphaFactor / evaluate_factor / FIELD_WHITELIST / OPERATOR_WHITELIST 等）
方法学来源：AAAI 2026《Navigating the Alpha Jungle》LLM-MCTS 框架
功能：把 BaseAlphaFactor 接上 "LLM提案 → MCTS选择 → 回测反馈" 的迭代循环

⚠️ 运行环境：聚宽【研究环境】(Notebook)。本地会因缺 jqfactor 报错。
⚠️ LLM 调用为【占位接口】，需接入你自己的模型(API/本地)，见 LLMInterface。
==================================================================
"""

import json
import os
import math
import random
from collections import defaultdict, Counter
from datetime import datetime

# 若阶段零与本文件在同一 Notebook 中，可直接使用其中的类与函数。
# 为解耦，这里做"软依赖"：优先从全局命名空间取，取不到则给出明确报错。
def _require(name):
    obj = globals().get(name) or __import__("builtins").__dict__.get(name)
    if obj is None:
        raise RuntimeError(
            f"未找到 {name}。请先在 Notebook 中执行《阶段零_聚宽基建骨架.py》再运行本文件。"
        )
    return obj


# ===================================================================
# 0. 知识库加载（对应 SOP 阶段一"白名单知识库 = 料库输入"）
# ===================================================================
class KnowledgeBase:
    """加载 knowledge_base.json，作为 LLM 与 MCTS 的唯一可引用料库。"""

    def __init__(self, path):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def fields(self):
        return [d["field"] for d in self.data["field_dictionary"] if d.get("whitelisted")]

    def operators(self):
        return list(self.data["operator_whitelist"])

    def effective_factors(self):
        return self.data.get("effective_factor_library", [])

    def failed_lessons(self):
        return [x["lesson"] for x in self.data.get("failed_experiments", [])]

    def fsa_genes(self, top_k=3):
        """FSA 频繁子树规避：返回需避开的 top-k 根基因（自动化的失败/饱和复盘）。"""
        genes = sorted(self.data.get("frequent_subtrees_fsa", []),
                       key=lambda g: g.get("support", 0), reverse=True)
        return [g["root_gene"] for g in genes[:top_k]]

    def add_factor(self, factor_meta):
        self.data.setdefault("effective_factor_library", []).append(factor_meta)
        self._save()

    def add_failure(self, attempt, result, lesson):
        self.data.setdefault("failed_experiments", []).append({
            "id": "X%04d" % (len(self.data["failed_experiments"]) + 1),
            "attempt": attempt, "result": result, "lesson": lesson,
            "date": datetime.now().strftime("%Y-%m-%d"),
        })
        self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


# ===================================================================
# 1. LLM 接口层（占位 — 接入你自己的模型）
# ===================================================================
class LLMInterface:
    """
    论文中 LLM 扮演三个角色：
      (a) propose_improvement：基于低分维度提出定向改进建议
          （论文：Softmax 按分数差采样，专攻弱项而非泛泛而谈）
      (b) translate_to_formula：把建议翻译成表达式树 / 因子类代码
      (c) summarize_history：回传播时生成改进历史摘要，防后续节点重复建议
    下面均为占位实现，请替换为真实调用（OpenAI / 通义 / 本地 Ollama 等）。
    """

    def __init__(self, model="gpt-4.1", kb=None):
        self.model = model
        self.kb = kb

    def propose_improvement(self, node, eval_report, kb):
        """
        入参:
          node         : MCTSNode（当前因子子树）
          eval_report  : evaluate_factor 返回的字典（ic/rankic/ir/turnover/diversity/overfit_risk）
          kb           : KnowledgeBase
        返回: 自然语言改进建议字符串（例如"当前 Turnover 偏高，建议改用低频 MA 差并用 Rank 平滑"）
        """
        # ===== TODO: 接入真实 LLM =====
        # 提示词要点（论文做法）：
        #   1) 把 eval_report 五维分数列出，找出最低 1~2 维
        #   2) 用 Softmax(分数差) 加权，让模型聚焦最弱维度
        #   3) 注入 kb.fsa_genes() 作为"禁止结构"硬约束
        #   4) 注入 kb.failed_lessons() 作为"别再踩的坑"
        #   5) 只允许使用 kb.fields()/kb.operators() 中的符号
        return ("[占位]建议：针对 %s 维度改进，使用白名单内算子重新组合，"
                "并避开根基因 %s。" % (
                    self._weak_dims(eval_report),
                    ",".join(kb.fsa_genes()) if kb else "",
                ))

    def translate_to_formula(self, suggestion, kb):
        """
        入参: suggestion 自然语言建议
        返回: (expr字符串, 算子列表) 或直接返回一个 BaseAlphaFactor 子类源码字符串
        论文对应：把建议翻译成新公式，语法非法则反馈 LLM 迭代修正。
        """
        # ===== TODO: 接入真实 LLM，并做白名单校验 =====
        expr = "MA(close,5) - MA(close,20)"   # 占位：实际应由 LLM 生成
        ops = ["MA", "Sub"]
        if kb is not None:
            # 越权算子即时拦截（机构流程：死权限边界）
            bad = [o for o in ops if o not in kb.operators()]
            if bad:
                raise PermissionError("LLM 生成了非白名单算子: %s" % bad)
        return expr, ops

    def summarize_history(self, trajectory):
        """
        入参: trajectory — 本次搜索至今的 (建议, 评分) 轨迹列表
        返回: 一句话摘要，注入后续节点的 prompt 以防重复建议（论文回传播机制）
        """
        # ===== TODO: 接入真实 LLM，把轨迹压缩为摘要 =====
        return "[占位]已尝试 %d 次改进，需避免重复方向。" % len(trajectory)

    @staticmethod
    def _weak_dims(report):
        dims = {"ic": report.get("ic", 0), "rankic": report.get("rankic", 0),
                "ir": report.get("ir", 0), "turnover": -report.get("turnover", 1),
                "diversity": report.get("diversity", 0),
                "overfit_risk": -report.get("overfit_risk", 0)}
        s = sorted(dims.items(), key=lambda kv: kv[1])[:2]
        return ",".join(k for k, _ in s)


# ===================================================================
# 2. MCTS 节点
# ===================================================================
class MCTSNode:
    def __init__(self, factor=None, expr=None, parent=None):
        self.factor = factor          # BaseAlphaFactor 实例或类
        self.expr = expr              # 表达式字符串
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q = 0.0                  # Q 值 = 子树最大评分（论文：回传播用 max 而非 mean）
        self.eval_report = None       # 五维评估字典
        self.history_summary = ""     # LLM 生成的改进历史摘要

    def is_leaf(self):
        return len(self.children) == 0

    def uct(self, c=1.0, total_visits=1):
        if self.visits == 0:
            return float("inf")
        return self.q + c * math.sqrt(math.log(total_visits + 1) / (self.visits + 1e-9))


# ===================================================================
# 3. 频繁子树规避 FSA（= 自动化的失败/饱和复盘，对应 SOP 阶段一）
# ===================================================================
class FrequentSubtreeAvoider:
    """
    论文：从【有效因子库】挖掘频繁闭合根基因（忽略参数只看结构，如 MA(vwap,t)），
    取 top-3 明确指令 LLM 生成时避开，防止公式结构同质化。
    这里提供：给定一批表达式，统计根基因频次并输出需规避清单。
    """

    @staticmethod
    def extract_root_gene(expr):
        """极简根式提取：取表达式最外层算子(形如 OP(...))。生产环境应做真正的 AST 解析。"""
        import re
        m = re.match(r"\s*([A-Za-z]+)\s*\(", expr)
        return m.group(1) + "(...)" if m else expr

    def mine(self, expr_list, top_k=3):
        genes = [self.extract_root_gene(e) for e in expr_list]
        cnt = Counter(genes)
        return [g for g, _ in cnt.most_common(top_k)]


# ===================================================================
# 4. 搜索循环（论文四阶段：选择 / 扩展 / 评估 / 回传播）
# ===================================================================
class AlphaMiner:
    def __init__(self, kb, llm, start, end, universe,
                 c=1.0, budget_init=3, verbose=True):
        self.kb = kb
        self.llm = llm
        self.start, self.end, self.universe = start, end, universe
        self.c = c
        self.budget = budget_init
        self.verbose = verbose
        self.trajectory = []          # (建议, 评分) 轨迹，供 LLM summarize
        self.root = MCTSNode(expr="SEED")   # 种子：可放人工基线因子
        self.best = None

    # ---- 4.1 选择：UCT 准则，任意节点可被选扩展（论文去除独立模拟阶段）----
    def select(self):
        node, total = self.root, self._total_visits()
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct(self.c, total))
            total = self._total_visits()
        return node

    # ---- 4.2 扩展：LLM 双角色（定向建议 + 翻译为公式），带参数回测择优 ----
    def expand(self, node):
        report = node.eval_report or {"ic": 0, "rankic": 0, "ir": 0,
                                      "turnover": 1, "diversity": 0, "overfit_risk": 0}
        suggestion = self.llm.propose_improvement(node, report, self.kb)
        self.trajectory.append((suggestion, report.get("q", 0)))
        node.history_summary = self.llm.summarize_history(self.trajectory)

        expr, ops = self.llm.translate_to_formula(suggestion, self.kb)
        child = MCTSNode(expr=expr, parent=node)
        node.children.append(child)
        return child

    # ---- 4.3 评估：跳过模拟，直接金融回测（复用阶段零 evaluate_factor）----
    def evaluate(self, node):
        evaluate_factor = _require("evaluate_factor")
        BaseAlphaFactor = _require("BaseAlphaFactor")
        try:
            fac = BaseAlphaFactor.from_expr(node.expr) if hasattr(BaseAlphaFactor, "from_expr") \
                  else BaseAlphaFactor(node.expr)
            rep = evaluate_factor(fac, self.start, self.end, self.universe)
        except Exception as e:
            # 语法非法/越权：反馈 LLM 迭代修正；此处先记录失败
            rep = {"ic": 0, "rankic": 0, "ir": 0, "turnover": 1,
                   "diversity": 0, "overfit_risk": 1, "error": str(e)}
            self.kb.add_failure("LLM生成表达式执行失败: %s" % node.expr,
                                str(e), "需把报错回灌 LLM 修正语法/白名单。")
        node.eval_report = rep
        node.visits += 1
        # 综合评分 = 五维加权（论文回测多维评分）；此处给一个可改权重示例
        node.q = (0.3 * rep.get("rankic", 0) + 0.2 * rep.get("ir", 0)
                  - 0.2 * rep.get("turnover", 1) + 0.15 * rep.get("diversity", 0)
                  - 0.15 * rep.get("overfit_risk", 0))
        return rep

    # ---- 4.4 回传播：Q = 子树最大评分，并写入 LLM 历史摘要 ----
    def backpropagate(self, node):
        best_q = node.q
        p = node.parent
        while p is not None:
            best_q = max(best_q, p.q)
            p.q = best_q
            p.visits += 1
            p.children_visited = getattr(p, "children_visited", 0) + 1
            p = p.parent
        # 动态预算：出现新纪录 +1（论文机制）
        if self.best is None or node.q > self.best.q:
            self.best = node
            self.budget += 1
            if self.verbose:
                print("[新纪录] q=%.4f expr=%s" % (node.q, node.expr))

    # ---- 主循环 ----
    def run(self, n_iter):
        for i in range(n_iter):
            node = self.select()
            if node is self.root or node.eval_report is None:
                child = self.expand(node)
            else:
                child = node
            rep = self.evaluate(child)
            self.backpropagate(child)
            if self.verbose and i % 10 == 0:
                print("[iter %d/%d] budget=%d best_q=%.4f" % (
                    i + 1, n_iter, self.budget, self.best.q if self.best else 0))
        return self.best

    def _total_visits(self):
        cnt = [0]
        def walk(n):
            cnt[0] += n.visits
            for c in n.children:
                walk(c)
        walk(self.root)
        return max(cnt[0], 1)


# ===================================================================
# 5. Demo：跑通"约束内生成"最小闭环
# ===================================================================
def demo_mine():
    kb = KnowledgeBase(os.path.join(os.path.dirname(__file__), "知识库", "knowledge_base.json"))
    llm = LLMInterface(kb=kb)
    # 注意：start/end/universe 需与阶段零 CONFIG 一致
    miner = AlphaMiner(kb, llm, start="2021-01-01", end="2023-12-31",
                       universe="CSI300", n_iter=20)
    best = miner.run(20)
    print("\n=== 搜索完成 ===")
    print("最优表达式:", best.expr if best else None)
    print("五维评估:", best.eval_report if best else None)
    return best


if __name__ == "__main__":
    demo_mine()
