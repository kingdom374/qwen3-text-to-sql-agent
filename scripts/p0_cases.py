"""Fixed questions and independent reference queries for the P0 demo database."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    question: str
    reference_sql: str | None
    answer_in_yuan: bool = False


@dataclass(frozen=True)
class Case:
    turns: tuple[Turn, ...]
    repair_on_error: bool = False
    injected_sql: tuple[str, ...] = ()


Q2_2026 = "order_date >= '2026-04-01' AND order_date < '2026-07-01'"
Q2_2025 = "order_date >= '2025-04-01' AND order_date < '2025-07-01'"
EAST = "region = '华东'"
NORTH = "region = '华北'"
COUNT_ORDERS = "SELECT COUNT(*) FROM orders WHERE "
PRODUCT_JOIN = "FROM orders o JOIN products p ON o.product_id = p.product_id "

CASES = {
    "add-region": Case((
        Turn("2026 年第二季度共有多少笔订单？", COUNT_ORDERS + Q2_2026),
        Turn("只看华东呢？", COUNT_ORDERS + f"{Q2_2026} AND {EAST}"),
    )),
    "replace-region": Case((
        Turn("2026 年第二季度，华东地区共有多少笔订单？", COUNT_ORDERS + f"{Q2_2026} AND {EAST}"),
        Turn("改看华北呢？", COUNT_ORDERS + f"{Q2_2026} AND {NORTH}"),
    )),
    "sales-repair": Case((
        Turn(
            "2026 年第二季度华东地区的销售额是多少？请以人民币元回答。",
            f"SELECT SUM(quantity * unit_price_cents) FROM orders WHERE {Q2_2026} AND {EAST}",
            answer_in_yuan=True,
        ),
    ), repair_on_error=True),
    "join-category": Case((
        Turn(
            "2026 年第二季度零食品类共有多少笔订单？请用订单表关联商品表后统计。",
            f"SELECT COUNT(*) {PRODUCT_JOIN}WHERE {Q2_2026} AND p.category = '零食'",
        ),
    )),
    "empty-orders": Case((
        Turn(
            "2026 年第一季度华东有哪些订单编号？按编号升序列出，若没有就明确返回空列表。",
            "SELECT order_id FROM orders WHERE order_date >= '2026-01-01' "
            f"AND order_date < '2026-04-01' AND {EAST} ORDER BY order_id",
        ),
    )),
    "count-by-category": Case((
        Turn(
            "按品类名称升序列出 2026 年第二季度各品类的订单笔数，每行先给品类再给笔数。",
            f"SELECT p.category, COUNT(*) {PRODUCT_JOIN}WHERE {Q2_2026} "
            "GROUP BY p.category ORDER BY p.category",
        ),
    )),
    "replace-year": Case((
        Turn(
            "2026 年第二季度华东地区售出了多少件商品？",
            f"SELECT SUM(quantity) FROM orders WHERE {Q2_2026} AND {EAST}",
        ),
        Turn("改成 2025 年同期呢？", f"SELECT SUM(quantity) FROM orders WHERE {Q2_2025} AND {EAST}"),
    )),
    "retry-exhausted": Case(
        (Turn("2026 年第二季度共有多少笔订单？", None),),
        repair_on_error=True,
        injected_sql=(
            "SELECT p0_injected_missing_column FROM orders",
            "SELECT p0_injected_missing_column_retry FROM orders",
        ),
    ),
}
