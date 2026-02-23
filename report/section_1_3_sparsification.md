## 1.3 梯度通信的稀疏化方法

### 1.3.1 研究背景与动机

与量化“降低每元素比特数”不同，**稀疏化**通过减少参与通信的梯度分量数量来降低通信量：仅传输选中的坐标及其取值（及必要的索引），未选中的分量在解码端视为零。根据现有综述，稀疏化可按选取规则分为**按幅值选取**（如 Top-k、阈值法）与**按概率选取**（如 Random-k），以及基于** Sketch 数据结构**的 holistic 方法（如 Sketched-SGD），在压缩比、无偏性、计算开销与收敛性之间形成不同权衡。

丢弃绝对值较小的分量对收敛的影响通常小于丢弃大分量，因此按幅值保留“重要”坐标是常见设计。这类方法多为有偏压缩，需配合**误差反馈**（将未发送的信息累积到残差并在下一轮加入梯度后再稀疏化）以在理论上保证收敛。按概率选取则易于满足无偏性（如 Random-k 配合缩放 $d/k$），计算复杂度低但方差较大。Sketch 类方法将梯度映射到低维结构后聚合，再利用线性性从合并后的 sketch 恢复近似 top-k 或 heavy hitters，通信量可做到与维度对数相关，适合大规模 worker 场景。

本节在 FSDP 的反向传播梯度同步路径下，讨论本课题所实现的稀疏化方法的形式化定义与实现要点，包括：Top-k、Random-k、Threshold-v（双阈值）以及 Sketched-SGD（Count Sketch + HEAVYMIX）。各算法的表述以原论文与综述为准。

---

### 1.3.2 统一框架与所实现的稀疏化方法

量化方法主要通过控制每个坐标的比特数来降低通信开销，而稀疏化则从另一个角度出发：并非对所有坐标一视同仁地发消息，而是仅对一小部分被认为“重要”的坐标进行通信。本节在统一残差框架的基础上，依次介绍 Top-k、Random-k、Threshold-v 和 Sketched-SGD，并说明它们在 FSDP 下的具体通信形式。

先给出稀疏化版的误差反馈与通信形式。设第 $t$ 步待同步的梯度向量为 $g_t \in \mathbb{R}^d$，稀疏化算子为 $C(\cdot)$，输出可以看作 $k$ 稀疏向量（或者一组索引–取值对）。带误差反馈时，记残差为 $e_t$，有
$$
\widetilde{g}_t = g_t + e_t,\quad \widehat{g}_t = C(\widetilde{g}_t),\quad e_{t+1} = \widetilde{g}_t - \widehat{g}_t.
$$
在 FSDP 中，若采用与量化相同的 reduce-scatter(sum) 语义，则各 rank 将稀疏向量补零为 $d$ 维后参与 reduce-scatter，解码后除以 world size $M$ 得到平均梯度分片。 Alternatively，可采用**稀疏通信**：各 rank 仅发送选中的索引与取值，经 all-gather 后在本地合并为完整梯度和，再按 rank 切分并除以 $M$。后者在 $k \ll d$ 时能显著减少通信字节数，但需要统一的索引编码与合并协议。下文均假定解码端得到的是“各 rank 稀疏梯度之和”的 $d$ 维向量，再取本 rank 分片并除以 $M$。

在这一框架下，最直接的一类方法是按幅值选取最大的若干坐标，即 Top-k。Top-k 保留梯度中绝对值最大的 $k$ 个分量，其余置零。记 $|g|_{(1)} \ge \cdots \ge |g|_{(d)}$ 为 $|g|$ 的非增排列，则
$$
(\mathrm{topk}(g))_i = \begin{cases} g_i, & \text{若 } i \in \mathrm{argmax}_{|S|=k} \sum_{j \in S} |g_j| \text{（即前 $k$ 大坐标）}, \\ 0, & \text{否则}. \end{cases}
$$
实现上常取 $k = \lfloor \rho d \rfloor$（$\rho$ 为稀疏率）。Top-k 是确定性的有偏压缩，误差反馈在这里尤为重要，用来“填平”那些每一步被抛弃的小梯度。计算复杂度大致为 $O(d \log k)$（可以通过部分排序实现）。在 FSDP 中，本实现支持两种通信形式：其一是对补零后的 $d$ 维向量做 reduce-scatter(sum)（稠密通信），其二是仅发送选中坐标的索引与取值，通过 all-gather 汇总各 rank 的 $(\text{indices}, \text{values})$ 后在本地合并为完整梯度和再切分（稀疏通信，发送量约为 $O(k)$）。代码里用 `sparse_comm` 标志在两种形式之间切换，默认开启稀疏通信，以便在 $k \ll d$ 时显著减少通信字节数。

Top-k 强调“优先传输大梯度”，但从无偏性的角度看仍属有偏压缩。若更关注无偏性，可以考虑采用随机稀疏化，即 Random-k。Random-k 从 $[d]$ 中均匀随机选取 $k$ 个坐标并保留其取值，其余置零。为保持无偏，对选中分量乘以缩放因子 $d/k$，即
$$
(\mathrm{randk}(g))_i = \begin{cases} (d/k) \cdot g_i, & \text{若 } i \in \omega, \\ 0, & \text{否则}, \end{cases}
\quad \omega \sim_{\mathrm{u.a.r.}} \binom{[d]}{k}.
$$
因此 $\mathbb{E}[\mathrm{randk}(g)] = g$，但方差与 $d/k$ 直接相关，$k$ 越小方差越大。计算上，Random-k 仅需一次随机抽样，复杂度为 $O(k)$；在 FSDP 中，其聚合与解码方式与 Top-k 完全一致，本实现同样通过 `sparse_comm` 切换是走稠密 reduce-scatter，还是走稀疏 all-gather 合并。

在“按幅值选取的 Top-k”和“按概率选取的 Random-k”之间，还可以通过显式阈值给出一种折中，即 Threshold-v。Threshold-v 使用正、负两个阈值 $v_+ \ge 0$、$v_- \ge 0$，仅保留满足 $g_i \ge v_+$ 或 $g_i \le -v_-$ 的分量，其余置零：
$$
(\mathrm{thresh}(g))_i = \begin{cases} g_i, & g_i \ge v_+ \text{ 或 } g_i \le -v_-, \\ 0, & \text{否则}. \end{cases}
$$
若 $v_+ = v_-$ 则为对称阈值。阈值既可以由用户显式给定，也可以根据目标稀疏率从当前梯度自适应估计（例如取满足稀疏率 $\rho$ 的分位数）。在本实现中，当用户未显式指定时，会在每步根据当前 $|g|$ 的经验分位数近似求得“约保留前 $\rho d$ 个最大绝对值”的对称阈值。Threshold-v 的计算复杂度为 $O(d)$，不需要完整排序，比 Top-k 更轻量；代价是对阈值更敏感，不同层或不同迭代间最优阈值可能相差较大。在 FSDP 中，它也可以走稠密补零 reduce-scatter，或者走稀疏索引–取值通信；如果外层强制固定发送长度为 $k$，就需要在超过阈值的候选者中再按幅值取 top-k，不足 $k$ 的部分则用零填充，让每个 rank 的发送形状一致。

上述三种方法都是逐坐标做决策：要么选前 $k$ 大，要么随机抽 $k$ 个，要么依据阈值判断是否保留。对于更大规模的分布式场景，还可以采用基于 sketch 的整体编码方法，即 Sketched-SGD。

Sketched-SGD（Count Sketch + HEAVYMIX）利用 Count Sketch 将梯度 $g \in \mathbb{R}^d$ 压缩为 $O(\log d)$ 量级的 sketch $S(g)$，并利用 sketch 的线性性 $S(g_1) + S(g_2) = S(g_1 + g_2)$ 在参数服务器或 all-reduce 语义下先合并各 worker 的 sketch，再从合并后的 sketch 恢复近似 top-k 坐标（heavy hitters）。Count Sketch 可以给出每个坐标的 $\ell_2$ 误差界，从而识别满足 $g_i^2 \gtrsim \|g\|_2^2/k$ 的分量。HEAVYMIX（见 Ivkin 等）在此基础上，从合并 sketch 中估计 $\hat{\ell}_2^2 \approx \|g\|_2^2$ 和各 $\hat{g}_i^2$，取 $H = \{i : \hat{g}_i^2 \ge \hat{\ell}_2^2/k\}$，再从剩余坐标随机补足到 $k$ 个得到 Topk，最后通过第二轮通信向各 worker 收集这些位置上的精确梯度值并求和。因此，每一轮 Sketched-SGD 通常包含：一轮 sketch 通信与聚合、一次 HEAVYMIX 恢复 Topk 索引、一轮精确值的 all-gather 或按索引收集。

在本课题的实现中，用一个开关参数控制“单轮 sketch”与“两轮 Sketch+HEAVYMIX”两种模式：单轮模式只使用 Count Sketch 恢复近似 top-k 稀疏向量，再走稀疏通信即可；两轮模式则更加贴近论文描述，各 rank 先发送局部 sketch 并 all-reduce 合并得到全局 top-k 索引，然后对这些索引上的精确梯度值做第二轮 all-gather 并合并为完整梯度和，最后按 FSDP 分片取本地部分并除以 $M$。

在 FSDP 中，上述稀疏化方法都是通过通信 hook 在 reduce-scatter（或者等价的稀疏 all-gather 合并）前接入：输入是未分片的扁平梯度 $g$，输出是当前 rank 应写入的平均梯度分片。采用稠密表示时，通常的做法是先得到 $k$ 稀疏向量，补零后直接参与 reduce-scatter(sum)，再除以 $M$；采用稀疏表示时，则发送 (indices, values)，在接收端把所有 rank 的稀疏更新合并为完整和，再切分、除以 $M$。误差反馈如果启用，残差需要与当前 $g$ 同形并按参数组正确维护。Top-k、Random-k、Threshold-v 可以看作逐坐标的稀疏化，而 Sketched-SGD 则是基于 sketch 的整体方法，计算和通信形态都略有不同，但在 FSDP 的 hook 接口下可以统一处理。后续实验会在这些方法之间比较压缩比、通信耗时、step time 和收敛性的差异，为不同资源与精度约束下的方案选择提供依据。
