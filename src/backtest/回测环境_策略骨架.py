# ============================================================
# 回测环境策略骨架（下游：消费阶段零~二挖出的因子）
# 适用环境：聚宽【策略编辑器 / 回测】
# 功能：选股 -> 调仓 -> 下单 -> 风控；平台自动输出 夏普/回撤/Alpha/Beta
# 重要：与研究环境 API 不互通，因子类需在本环境重新定义（或从阶段零复制）
# ============================================================

from jqfactor import Factor, get_factor_values
# 注：set_benchmark / set_option / set_order_cost / set_slippage / order_target_value
#     等为回测环境内置，无需 import。

# ---------- 0. 因子定义（与阶段零保持一致；此处粘贴你通过质控的因子类） ----------
class DemoTrendFactor(Factor):
    name = "demo_trend"
    max_window = 20
    dependencies = ["close"]

    def calc(self, data):
        close = data["close"]
        return close.rolling(20).mean() - close.rolling(5).mean()

# ---------- 1. 策略配置 ----------
CONFIG = {
    "universe": "000300.XSHG",        # 股票池（与阶段零一致）
    "hold_num": 30,                   # 持仓数量
    "rebalance_freq": "weekly",       # 调仓频率：weekly / daily
    "max_weight_per_stock": 0.05,     # 单票权重上限 5%（风控）
    "max_position_ratio": 0.95,       # 最大总仓位（留 5% 现金）
    "min_market_cap": 5e10,           # 市值下限，剔除小盘（可选）
}

# ---------- 2. 初始化（回测环境入口） ----------
def initialize(context):
    set_benchmark(CONFIG["universe"])
    set_option('use_real_price', True)                              # 真实价模式
    set_order_cost(Commission(buy_cost=0.0003, sell_cost=0.0013, min_cost=5), type='stock')
    set_slippage(FixedSlippage(0.002))                              # 双边千二滑点
    g.factor = DemoTrendFactor()
    g.target_stocks = []

    if CONFIG["rebalance_freq"] == "weekly":
        run_weekly(rebalance, 1, 'open')        # 每周一开盘调仓
    else:
        run_daily(rebalance, 'open')           # 每日开盘调仓

# ---------- 3. 盘前：刷新股票池 + 过滤 ----------
def before_trading_start(context):
    stocks = get_index_stocks(CONFIG["universe"], date=context.current_dt)
    g.universe_stocks = filter_specials(context, stocks)

# ---------- 4. 调仓主逻辑 ----------
def rebalance(context):
    # 1) 取因子最新值（回测环境专用 get_factor_values）
    factor_data = get_factor_values(
        g.factor, g.universe_stocks,
        end_date=context.previous_date, count=1,
    )[g.factor.name]
    latest = factor_data.iloc[-1].dropna()

    # 2) 打分选股：因子值降序取 Top N
    ranked = latest.sort_values(ascending=False)
    target = ranked.head(CONFIG["hold_num"]).index.tolist()

    # 3) 风控：单票上限 + 总仓位上限
    total_value = context.portfolio.total_value
    max_total = total_value * CONFIG["max_position_ratio"]
    per_stock = min(total_value * CONFIG["max_weight_per_stock"],
                    max_total / max(len(target), 1))

    # 4) 调仓：买入目标，卖出非目标
    for stock in target:
        order_target_value(stock, per_stock)
    for stock in context.portfolio.positions:
        if stock not in target:
            order_target_value(stock, 0)

    g.target_stocks = target
    log.info("调仓完成，持仓 %d 只，单票目标市值 %.0f" % (len(target), per_stock))

# ---------- 5. 工具：过滤 ST / 停牌 ----------
def filter_specials(context, stocks):
    # ⚠️ 必须用 get_current_data() 函数，回测环境没有全局变量 current_data
    cd = get_current_data()
    return [s for s in stocks
            if not cd[s].paused          # 非停牌
            and not cd[s].is_st]         # 非ST（已退市股票不在指数成分内，无需额外过滤）

# ---------- 说明 ----------
# 风险指标（Sharpe / MaxDrawdown / Alpha / Beta / 收益）由回测引擎自动计算，
# 在回测结果页直接查看，无需手写。本骨架只负责"因子 -> 选股 -> 下单 -> 基础风控"。
# 后续可叠加：止损线、行业中性约束、多因子合成（把多个通过质控的因子加权）。
