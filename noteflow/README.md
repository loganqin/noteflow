# NoteFlow

NoteFlow 是论文 *NoteFlow: Ranking Deposit--Withdrawal Links in Ethereum
Mixers* 的独立开源实现。它接收任意上游模型产生的固定候选配对分数，输出：

1. 满足存款单次使用约束的全局最优首选；
2. 由跨提款竞争估计得到的非负存款价格；
3. 保持可行首选不变、同时保留全部合法候选的完整排名。

本目录是纯代码发行包。**不含交易数据、训练标签、模型权重、候选分数、
缓存或论文实验结果。** 数据接口及隔离方式见 [DATA.md](DATA.md)。

## 方法

对提款 `i`、存款 `j` 的固定分数 `s_ij`，NoteFlow 分三步运行：

- **Full-Graph Assignment**：在所有合法边上求最大权重一对一匹配，得到
  `g_i`。这一阶段不做 Top-L 裁剪。
- **Deposit-Price Estimation**：在每行分数最高的 `L` 条边和一组公开可行
  骨架边上，求熵正则容量松弛的有限迭代近似，得到价格 `p_j >= 0`。
- **Fixed-First Ranking**：计算 `a_ij = s_ij - lambda * p_j`，输出
  `[g_i] + sort_desc(C_i \\ {g_i}, a_ij)`。

因此，第一名组成联合可行匹配；第二名以后是每个查询的候选解释，不表示
多套同时可行的完整匹配。价格估计可以是稀疏且未在迭代预算内收敛的近似，
相关状态会明确写入诊断信息。

## 安装

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

开发环境：

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/check_release.py
```

## Python API

输入采用压缩的一维表示。`counts[i]` 是第 `i` 个查询的候选数；该查询对应
`scores` 的下一段连续区间，候选列为公共候选序列的前缀
`0, ..., counts[i]-1`。

```python
import numpy as np
from noteflow import NoteFlowConfig, run_noteflow

counts = np.array([3, 4, 4])
scores = np.concatenate([
    [2.0, 1.9, 0.1],
    [2.1, 1.2, 0.4, 0.0],
    [1.8, 1.7, 1.6, 0.3],
])

result = run_noteflow(
    scores,
    counts,
    row_ids=np.array(["w-a", "w-b", "w-c"]),
    config=NoteFlowConfig(),
)

print(result.assignment.columns)  # 联合可行的第一名
print(result.ranking_for(0))      # 查询 0 的完整候选排名
print(result.diagnostics())       # 近似与收敛状态
```

默认参数与论文锁定设置一致：`T=0.3`、`lambda=1.0`、`L=64`、最多
1000 次迭代、容差 `0.001`。完整可运行示例见
[examples/quickstart.py](examples/quickstart.py)。

## 命令行

准备一个外部 `.npz` 文件，其中包含 `scores`、`counts`，以及可选的
`row_ids`：

```bash
noteflow \
  --input ../noteflow-data/scores.npz \
  --output ../noteflow-data/noteflow-output.npz \
  --config configs/paper.json
```

输出仍保存在外部数据目录，不应提交到代码仓库。输出文件包含 `assignment`、
`prices`、`column_mass`、`adjusted_scores`、压缩的 `rankings`、
`ranking_offsets` 和 JSON `metadata`。所有 NumPy 文件均以
`allow_pickle=False` 读取。

如需在全部候选边上估计价格，可使用 `--full-support`；这不会改变硬匹配
始终使用完整候选图的事实。

## 规模与内存

硬匹配是精确的矩形线性分配，当前参考实现会分配
`query_count * max(counts)` 个 `float64` 单元。默认内存保护阈值为一亿单元
（约 0.75 GiB）。超过阈值时实现会显式失败，避免意外耗尽内存；在确认机器
容量后可通过 `max_dense_elements` 调整。价格估计在默认设置下使用至多约
`m * (L + 1)` 条支持边，但最终排名仍覆盖全部输入候选。

## 测试与发布检查

```bash
pytest -q
python scripts/check_release.py
python examples/quickstart.py
python -m build
```

测试覆盖精确匹配与穷举解的一致性、软价格与可解对称情形的一致性、
稀疏支持的行置换等变性、固定首选与候选完整性、未收敛诊断，以及 CLI 的
无 pickle 往返。

`scripts/check_release.py` 会拒绝常见数据、权重、缓存和结果文件；构建配置
也不会把这些内容收入源码包或 wheel。

## 适用边界

- 候选集必须是同一个稳定存款序列的前缀，并存在覆盖所有查询的一对一匹配。
- 实现对应离线窗口内的联合推理，不应描述为严格在线预测。
- 单次使用假设不直接覆盖拆分、多对多、缺失真值或候选集外真值。
- 预测链接只是候选假设，不构成身份、意图或违法行为的证据。
- 上游评分器训练和论文实验复现需要另行提供的数据与模型，不属于此代码包。

## License

MIT，见 [LICENSE](LICENSE)。发布前如需使用机构指定许可证或实名版权信息，
可直接替换该文件及 `pyproject.toml` 中的对应元数据。
