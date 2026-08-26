# 严格按列分核验证记录

日期：2026-08-13

## 命令

```powershell
python .\non_deter_swizzle.py --full-test --demo
```

## 结果

```text
PASS dense: cases=4968, coordinates=6574396, columns=172735,
            max_load_skew/m=1.00 <= 1.00
PASS causal: cases=920, coordinates=4065476, columns=107593,
             max_load_skew/m=1.00 <= 1.00
PASS strict column ownership: every (batch, s2_idx) has exactly one owner core
PASS relaxed constraint: same-row tasks in one local_step are allowed and observed
dense: tasks=384, max_local_steps=48, same_column_transition=95.74%
causal: tasks=408, max_local_steps=58, same_column_transition=90.00%
```

## 核心断言

- 10,639,872 个有效坐标无重复、无遗漏、无越界；
- 280,328 个实际 `(batch,s2_idx)` 列的 owner 集合大小全部为 1；
- 每个坐标所在核与 `dense_column_owner/causal_column_owner` 的解析结果一致；
- Causal 坐标全部满足 `s2_idx<=s1_idx`；
- 奇数 batch 尾三角没有 padding 任务；
- 每核负载差不超过一个最长列 `m`；
- 同 local-step 同行任务允许出现，未误保留确定性约束；
- Causal 负载公式额外穷举 `k=1..64、m=1..256、B=1..9`；
- 尾三角整数反解抽查到 `m=1,000,000`。

## 代表性列 owner 审计

```text
Dense  k=8,m=16,n=12,B=2: columns=24, split_columns=0, max_owners=1
Causal k=8,m=16,B=2:      columns=32, split_columns=0, max_owners=1
Causal k=8,m=16,B=3:      columns=48, split_columns=0, max_owners=1
```

说明：这是 Python 坐标算法验证，不代表 NPU 精度或性能结论。正式 C++ 迁移后仍需验证 Kernel 写回冲突机制及端到端性能。
