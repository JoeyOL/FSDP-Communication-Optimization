## 1.3 梯度通信的稀疏化方法

### 1.3.1 研究背景与动机

与量化“降低每元素比特数”不同，**稀疏化**通过减少参与通信的梯度分量数量来降低通信量：仅传输选中的坐标及其取值（及必要的索引），未选中的分量在解码端视为零。根据现有综述，稀疏化可按选取规则分为**按幅值选取**（如 Top-k、阈值法）与**按概率选取**（如 Random-k），以及基于** Sketch 数据结构**的 holistic 方法（如 Sketched-SGD），在压缩比、无偏性、计算开销与收敛性之间形成不同权衡。

丢弃绝对值较小的分量对收敛的影响通常小于丢弃大分量，因此按幅值保留“重要”坐标是常见设计。这类方法多为有偏压缩，需配合**误差反馈**（将未发送的信息累积到残差并在下一轮加入梯度后再稀疏化）以在理论上保证收敛。按概率选取则易于满足无偏性（如 Random-k 配合缩放 $d/k$），计算复杂度低但方差较大。Sketch 类方法将梯度映射到低维结构后聚合，再利用线性性从合并后的 sketch 恢复近似 top-k 或 heavy hitters，通信量可做到与维度对数相关，适合大规模 worker 场景。

本节在 FSDP 的反向传播梯度同步路径下，讨论本课题所实现的稀疏化方法的形式化定义与实现要点，包括：Top-k、Random-k、Threshold-v（双阈值）以及 Sketched-SGD（Count Sketch + HEAVYMIX）。各算法的表述以原论文与综述为准。

---

### 1.3.2 统一框架与所实现的稀疏化方法

**误差反馈与通信形式。** 设第 $t$ 步待同步的梯度向量为 $g_t \in \mathbb{R}^d$，稀疏化算子为 $C(\cdot)$，输出为 $k$ 稀疏向量（或等价的索引–取值对）。带误差反馈时，记残差为 $e_t$，则
$$
\widetilde{g}_t = g_t + e_t,\quad \widehat{g}_t = C(\widetilde{g}_t),\quad e_{t+1} = \widetilde{g}_t - \widehat{g}_t.
$$
在 FSDP 中，若采用与量化相同的 reduce-scatter(sum) 语义，则各 rank 将稀疏向量补零为 $d$ 维后参与 reduce-scatter，解码后除以 world size $M$ 得到平均梯度分片。 Alternatively，可采用**稀疏通信**：各 rank 仅发送选中的索引与取值，经 all-gather 后在本地合并为完整梯度和，再按 rank 切分并除以 $M$。后者在 $k \ll d$ 时能显著减少通信字节数，但需要统一的索引编码与合并协议。下文均假定解码端得到的是“各 rank 稀疏梯度之和”的 $d$ 维向量，再取本 rank 分片并除以 $M$。

**Top-k。** Top-k 保留梯度中**绝对值最大的 $k$ 个**分量，其余置零。记 $|g|_{(1)} \ge \cdots \ge |g|_{(d)}$ 为 $|g|$ 的非增排列，则
$$
(\mathrm{topk}(g))_i = \begin{cases} g_i, & \text{若 } i \in \mathrm{argmax}_{|S|=k} \sum_{j \in S} |g_j| \text{（即前 $k$ 大坐标）}, \\ 0, & \text{否则}. \end{cases}
$$
实现上常取 $k = \lfloor \rho d \rfloor$（$\rho$ 为稀疏率）。Top-k 为确定性、有偏压缩，需误差反馈以保证收敛；计算复杂度为 $O(d \log k)$（如用部分排序）。在 FSDP 中可对补零后的 $d$ 维向量做 reduce-scatter(sum)，或采用稀疏 all-gather 索引与取值再在本地合并。

**Random-k。** Random-k 从 $[d]$ 中**均匀随机**选取 $k$ 个坐标并保留其取值，其余置零。为保持无偏，对选中分量乘以缩放因子 $d/k$，即
$$
(\mathrm{randk}(g))_i = \begin{cases} (d/k) \cdot g_i, & \text{若 } i \in \omega, \\ 0, & \text{否则}, \end{cases}
\quad \omega \sim_{\mathrm{u.a.r.}} \binom{[d]}{k}.
$$
因此 $\mathbb{E}[\mathrm{randk}(g)] = g$，方差与 $d/k$ 相关。计算上仅需一次随机抽样，复杂度为 $O(k)$，但随机数生成在实际中可能带来额外开销。在 FSDP 中的聚合与解码方式同 Top-k（稠密 reduce-scatter 或稀疏 all-gather 合并）。

**Threshold-v（双阈值）。** Threshold-v 使用正、负两个阈值 $v_+ \ge 0$、$v_- \ge 0$，仅保留满足 $g_i \ge v_+$ 或 $g_i \le -v_-$ 的分量，其余置零：
$$
(\mathrm{thresh}(g))_i = \begin{cases} g_i, & g_i \ge v_+ \text{ 或 } g_i \le -v_-, \\ 0, & \text{否则}. \end{cases}
$$
若 $v_+ = v_-$ 则为对称阈值。阈值可由用户给定，或根据目标稀疏率从当前梯度估计（如取满足稀疏率 $\rho$ 的分位数）。计算复杂度为 $O(d)$，无需排序；缺点是对阈值敏感，不同层/不同迭代最优阈值可能不同。在 FSDP 中同样可走稠密补零 reduce-scatter 或稀疏索引–取值通信；若采用固定 $k$ 的稀疏通信，需在超过阈值的候选者中再按幅值取 top-k，不足 $k$ 则以零填充以保持各 rank 发送长度一致。

**Sketched-SGD（Count Sketch + HEAVYMIX）。** Sketched-SGD 利用 **Count Sketch** 将梯度 $g \in \mathbb{R}^d$ 压缩为 $O(\log d)$ 量级的 sketch $S(g)$，并利用 sketch 的**线性性** $S(g_1) + S(g_2) = S(g_1 + g_2)$ 在参数服务器或 all-reduce 语义下先合并各 worker 的 sketch，再从合并后的 sketch 恢复近似 top-k 坐标（heavy hitters）。Count Sketch 可对每个坐标给出 $\ell_2$ 误差界，从而识别满足 $g_i^2 \gtrsim \|g\|_2^2/k$ 的分量。**HEAVYMIX**（见 Ivkin 等）从合并 sketch 中查询 $\hat{\ell}_2^2 \approx \|g\|_2^2$ 与各 $\hat{g}_i^2$，取 $H = \{i : \hat{g}_i^2 \ge \hat{\ell}_2^2/k\}$，再从剩余坐标中随机补足至 $k$ 个得到 Topk，最后通过**第二轮通信**向各 worker 收集 Topk 位置上的精确梯度值并求和。因此每轮包含：一轮 sketch 通信与聚合、一次 HEAVYMIX 恢复 Topk 索引、一轮精确值 all-gather 或按索引收集。误差反馈将本轮未进入更新向量的部分写入残差，下一轮与梯度相加后再做 sketch。本课题实现中，两轮可简化为：各 rank 发送局部 sketch，合并后得到全局 top-k 索引（或近似），再 all-gather 各 rank 在 these 索引上的取值并合并为完整梯度和，最后按 FSDP 分片取本地部分并除以 $M$。

**实现要点与小结。** 在 FSDP 中，上述稀疏化方法均通过通信 hook 在 reduce-scatter（或等价的稀疏 all-gather 合并）前接入：输入为未分片的扁平梯度 $g$，输出为当前 rank 应写入的平均梯度分片。若采用稠密表示，则先得到 $k$ 稀疏向量，补零后参与 reduce-scatter(sum)，再除以 $M$；若采用稀疏表示，则发送 (indices, values) 并在接收端合并为完整和再切分、除以 $M$。误差反馈若启用，残差需与当前 $g$ 同形并按参数组正确维护。Top-k / Random-k / Threshold-v 为逐元素稀疏化，Sketched-SGD 为基于 sketch 的 holistic 方法，计算与通信形态不同，但均可接入同一 FSDP hook 接口。后续将在此基础上进行实现与实验对比，评估各方法在压缩比、通信耗时、step time 与收敛性上的权衡。
