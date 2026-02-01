import re
from collections import defaultdict

def parse_log(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    # 匹配一个用例块
    pattern = re.compile(
        r"Test Case #(\d+).*?Data Type:\s+\x1b\[36m(\w+)\x1b\[0m.*?Avg Time:\s+([0-9.]+)\s+ms",
        re.S
    )
    out = {}
    for case_id, dtype, avg in pattern.findall(text):
        out[(int(case_id), dtype)] = float(avg)
    return out

naive = parse_log("naive.log")
reduce = parse_log("reduce.log")

all_keys = sorted(set(naive) | set(reduce))
print(f"{'Case':>4} {'Type':>6} {'naive(ms)':>10} {'reduce(ms)':>11} {'diff(ms)':>9} {'speedup':>8}")
for (case_id, dtype) in all_keys:
    n = naive.get((case_id, dtype))
    r = reduce.get((case_id, dtype))
    if n is None or r is None:
        continue
    diff = r - n
    speedup = (n / r) if r != 0 else float("inf")
    print(f"{case_id:>4} {dtype:>6} {n:>10.6f} {r:>11.6f} {diff:>9.6f} {speedup:>8.3f}")