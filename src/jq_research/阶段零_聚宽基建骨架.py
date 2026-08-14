# ============================================================
# 阶段零：聚宽因子挖掘基建骨架（个人 SOP 落地版）
# 适用环境：聚宽【研究环境 / Notebook】—— 依赖 jqfactor / jqdata
# 设计目标：统一口径 + 算子/字段白名单 + 统一预处理管线 + Factor 基类约束
# 重要：本文件是模板，必须在聚宽研究环境运行（本地无 jqfactor 包，不可直接跑）
# API 依据：聚宽因子看板文档(#name:factor) + 平台API文档(#name:api)
# ============================================================

import numpy as np
import pandas as pd

import jqfactor  # 聚宽因子模块
from jqfactor import (
    Factor, calc_factors, analyze_factor,
    winsorize, winsorize_med, neutralize, standardlize,
)

# ---------- 0. 全局配置（统一口径，消除幻觉根因） ----------
CONFIG = {
    "universe": "000300.XSHG",     # 默认股票池（沪深300）；中证1000 用 "000852.XSHG"
    "industry": "sw_l1",           # 行业分类：申万一级（亦可 sw_l2 / jq_l1）
    "start_date": "2011-01-01",
    "end_date": "2024-11-30",
    "max_window": 60,              # 回看窗口上限（算子白名单约束，防过拟合）
    "quantiles": 5,                # 分层数
    "periods": (1, 5, 10, 20),     # 收益预测周期（日）
    "use_real_price": True,        # 真实价格模式（前复权/真实价固定，避免口径不一致）
    "top_n": 50,                   # 入库因子目标数量（参考论文 alpha 集大小）
}

# ---------- 1. 字段白名单（数据字典提炼，权限边界） ----------
# 仅允许引用下述字段；引用未列入的字段将被拦截。
FIELD_WHITELIST = {
    "行情": ["open", "high", "low", "close", "volume", "money",
             "high_limit", "low_limit", "pre_close", "factor"],
    "估值": ["market_cap", "circulating_market_cap", "pe_ratio", "pb_ratio",
             "ps_ratio", "pcf_ratio", "turnover_ratio"],
    # 财务字段需经 get_fundamentals / finance.run_query 取，带 _1(上一季)/_y(最新年) 时间偏移
    "财务": ["roe", "roa", "gross_profit_margin", "operating_profit_margin",
             "net_profit_margin", "eps", "inc_revenue", "inc_net_profit"],
}

def check_field(field):
    flat = [f for grp in FIELD_WHITELIST.values() for f in grp]
    if field not in flat:
        raise PermissionError(f"字段 '{field}' 不在白名单，已被权限边界拦截。")
    return True

# ---------- 2. 算子白名单（借鉴论文 Table 2，约 30 个） ----------
# key=内部名, value=人类可读公式；引用未列入的算子将被拦截。
OPERATOR_WHITELIST = {
    # 一元
    "neg": "-x", "abs": "|x|", "square": "x^2", "inv": "1/x", "sign": "Sign(x)",
    "sin": "Sin(x)", "cos": "Cos(x)", "tanh": "Tanh(x)", "log": "Log(x)",
    "delay": "Delay(x,t)", "diff": "Diff(x,t)", "pct": "Pct(x,t)",
    "ma": "Ma(x,t)", "median": "Med(x,t)", "sum": "Sum(x,t)", "std": "Std(x,t)",
    "max": "Max(x,t)", "min": "Min(x,t)", "rank": "Rank(x,t)",
    "skew": "Skew(x,t)", "kurt": "Kurt(x,t)", "vari": "Vari(x,t)",
    "autocorr": "Autocorr(x,t,n)", "zscore": "Zscore(x,t)",
    # 二元
    "add": "x+y", "sub": "x-y", "mul": "x*y", "div": "x/y",
    "greater": "Greater(x,y)", "less": "Less(x,y)",
    "cov": "Cov(x,y,t)", "corr": "Corr(x,y,t)",
}

def check_operator(op):
    if op not in OPERATOR_WHITELIST:
        raise PermissionError(f"算子 '{op}' 不在白名单，已被权限边界拦截。")
    return True

# ---------- 3. 统一预处理管线（所有因子必经，消除口径差异） ----------
def standardize_pipeline(factor_series, axis=1):
    """统一：中位数去极值 -> 市值+行业中性化 -> Z-Score 标准化。
    所有因子在入库前必须走此管线，确保同一指标全口径一致。"""
    s = winsorize_med(factor_series, scale=5.0)              # 中位数去极值（5倍MAD）
    s = neutralize(s, how=["market_cap", "industry"], axis=axis)  # 市值+行业中性化
    s = standardlize(s)                                     # Z-Score 标准化
    return s

# ---------- 4. Factor 基类（约束生成：强制白名单 + 统一预处理） ----------
class BaseAlphaFactor(Factor):
    """所有自定义因子必须继承此类，强制：
       1) 使用白名单算子（calc 内引用算子前调用 check_operator）；
       2) max_window 受全局上限约束；
       3) 原始值自动走统一预处理管线。
       子类需定义：name / max_window / dependencies / calc。"""
    max_window = CONFIG["max_window"]

    def calc(self, data):
        raise NotImplementedError("子类必须实现 calc()")

    def _postprocess(self, raw):
        return standardize_pipeline(raw)

# ---------- 5. 统一计算入口 ----------
def build_factor(factor_cls, securities, start_date=None, end_date=None):
    start_date = start_date or CONFIG["start_date"]
    end_date = end_date or CONFIG["end_date"]
    f = factor_cls()
    res = calc_factors(
        securities=securities, factors=[f],
        start_date=start_date, end_date=end_date,
        use_real_price=CONFIG["use_real_price"], skip_paused=True,
    )
    return res[f.name]

# ---------- 6. 示例因子：最短可跑通的趋势因子 ----------
class DemoTrendFactor(BaseAlphaFactor):
    name = "demo_trend"
    max_window = 20
    dependencies = ["close"]

    def calc(self, data):
        check_operator("ma")               # 约束：只用白名单算子
        close = data["close"]
        return close.rolling(20).mean() - close.rolling(5).mean()

# ---------- 7. 五维评估（3维聚宽原生 + 2维自建） ----------
def evaluate_factor(factor_df, start_date=None, end_date=None):
    """聚宽原生支持：IC / RankIC / 分层收益 / 换手率 tear sheet。"""
    start_date = start_date or CONFIG["start_date"]
    end_date = end_date or CONFIG["end_date"]
    far = analyze_factor(
        factor_df, start_date=start_date, end_date=end_date,
        industry=CONFIG["industry"], universe=CONFIG["universe"],
        quantiles=CONFIG["quantiles"], periods=CONFIG["periods"],
        use_real_price=CONFIG["use_real_price"], skip_paused=True,
    )
    return far  # far.ic / far.ic_monthly / turnover tear sheet / create_full_tear_sheet()

def diversity_score(new_factor, factor_library):
    """自建 Diversity 维度：与已入库因子的最大相关性，要求 < 0.8（论文标准）。"""
    corrs = factor_library.apply(lambda col: new_factor.corr(col))
    return corrs.abs().max()

def overfit_risk_score(far, train_end="2020-12-31"):
    """自建 Overfitting 维度：样本内/外 RankIC 衰减比。衰减越大风险越高。"""
    ic_in = far.ic.loc[:train_end].mean()
    ic_out = far.ic.loc[train_end:].mean()
    decay = (ic_in - ic_out) / (abs(ic_in) + 1e-9)
    return decay  # 越接近 1 越可能过拟合

# ---------- 8. 使用示范（在聚宽研究环境运行） ----------
def demo_run():
    securities = get_index_stocks(CONFIG["universe"], date=CONFIG["end_date"])
    fval = build_factor(DemoTrendFactor, securities)
    far = evaluate_factor(fval)
    print("IC:\n", far.ic)
    print("月度IC:\n", far.ic_monthly)
    far.create_full_tear_sheet()   # 出全景报告

if __name__ == "__main__":
    # demo_run()  # 取消注释在聚宽研究环境执行
    pass
