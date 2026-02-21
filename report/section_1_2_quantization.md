## 1.2 梯度通信的量化压缩方法

### 1.2.1 研究背景与动机

在数据并行的分布式深度学习中，各计算节点在每轮迭代中需同步梯度以更新全局模型。当模型参数量与节点数较大时，梯度通信的带宽与延迟往往成为训练时间的主要瓶颈。通信压缩通过减少每轮传输的数据量或比特数来缓解该瓶颈；其中**量化**一类方法不改变参与通信的梯度维度，而是将每个梯度分量用更少的比特表示，从而在保持聚合语义（如求和或平均）的前提下降低通信量。

根据现有综述，通信量化可大致分为**截断表示**（truncated expression）与**码本映射**（code-book mapping）两类。量化在理论上带来**通信–方差权衡**：比特数越少，通信越省，但量化误差通常增大，导致梯度估计的方差上升，可能增加达到目标精度所需的迭代次数。因此，量化方法的设计目标是在给定通信预算下，使量化后的梯度估计尽可能无偏且方差可控，从而保证收敛性与收敛速度。

本节在 Fully Sharded Data Parallel（FSDP）的反向传播梯度同步路径下，讨论本课题所实现的梯度量化方法的形式化定义、统计性质及实现要点，包括：INT8 对称量化（含动态树变体）、QSGD 随机量化、Natural Compression（NC）、以及 1-bit 类方法（含误差反馈与 Seide 形式）。各算法的公式与性质均以原论文或综述中的表述为准。

---

### 1.2.2 统一框架与所实现的量化方法

**误差反馈与接口约束。** 设第 $t$ 步待同步的梯度向量为 $g_t \in \mathbb{R}^d$，量化算子为 $C(\cdot)$，解码（反量化）为 $D(\cdot)$。理想情况下希望 $D(C(g_t))$ 接近 $g_t$。若量化引入偏置，可通过**误差反馈**（Error Feedback, EF）在本地维护残差并参与下一轮量化。记残差为 $e_t$，则带误差反馈的流程为
$$
\widetilde{g}_t = g_t + e_t,\quad q_t = C(\widetilde{g}_t),\quad e_{t+1} = \widetilde{g}_t - D(q_t).
$$
在 FSDP 中，梯度同步采用 reduce-scatter(sum) 的聚合方式：各 rank 对量化后的梯度做按元素求和，再按 rank 切分得到本地分片。因此解码后需对分片结果除以 world size $M$ 才得到“平均梯度”分片。若量化依赖尺度或范数等标量，这些量需在参与聚合的各 rank 间一致（或通过一次轻量级 all-reduce 取得），否则会引入系统性偏差。下文各方法均在此约定下讨论。

**INT8 对称量化。** 8 比特量化在保持较高数值精度的同时，将每元素通信量降为全精度的 1/4。线性对称量化以全局最大绝对值作为尺度：设 $g \in \mathbb{R}^d$，取 $\alpha = \max_i |g_i|$，量化区间半宽为 $Q_r$（例如 $Q_r = \lfloor 127/M \rfloor$ 以兼顾 reduce-scatter 求和后不溢出），尺度 $s = Q_r / \max(\alpha, \epsilon)$，量化与反量化为 $q_i = \mathrm{clip}(\mathrm{round}(g_i \cdot s), -Q_r, Q_r)$，$\widehat{g}_i = q_i/s$。通信时传输 int8 的 $q$，在接收端按 $s$ 反量化后除以 $M$ 即得平均梯度分片。**动态树 8-bit** 采用非均匀表示：将归一化后的梯度用 7 比特随机舍入到若干级，并与符号一起编码为 uint8，在相同通信位宽下可减小大值分量的相对误差；在 FSDP 下需对量化后的向量做 all-gather 再在浮点域解码求和，或等价地保证各 rank 使用一致尺度与编码规则。

**QSGD：随机均匀量化。** QSGD 通过随机舍入将梯度映射到 $s$ 个离散级，在期望意义下保持无偏。记 $\|g\|_2$ 为梯度向量的 $\ell_2$ 范数，对 $g \neq 0$ 有（Alistarh 等，NIPS 2017）
$$
C_{\mathrm{QSGD}}(g_i) = \|g\|_2 \cdot \mathrm{sign}(g_i) \cdot \xi_i(g, s),
$$
其中 $\xi_i(g,s)$ 取值于 $\{0, 1/s, \ldots, 1\}$：设 $|g_i|/\|g\|_2 \in [\ell/s, (\ell+1)/s]$，$0 \le \ell < s$，则 $\xi_i$ 以概率 $p_i = s \cdot |g_i|/\|g\|_2 - \ell$ 取 $(\ell+1)/s$，否则取 $\ell/s$。有 $\mathbb{E}[C_{\mathrm{QSGD}}(g)] = g$ 且 $\mathbb{E}[\|C_{\mathrm{QSGD}}(g) - g\|_2^2] \le \min(n/s^2, \sqrt{n}/s)\|g\|_2^2$。实践中可采用分桶 QSGD：将 $g$ 分成若干桶，每桶独立按 $\ell_2$ 范数缩放并做 $s$ 级随机量化。在 FSDP 中需先就 $\|g\|_2$（或每桶范数）做一次标量同步，再对量化后的向量做 reduce-scatter(sum)，解码时用同一尺度反量化并除以 $M$。

**Natural Compression（NC）。** NC 采用非均匀码本：将每个标量随机舍入到与其最近的两个二次幂之一，在无偏的前提下使二阶矩增长不超过常数因子（$\omega = 1/8$）。对 $t \neq 0$，记 $\alpha = \log_2 |t|$，NC 定义为（Horváth 等，MSML 2022）
$$
C_{\mathrm{nat}}(t) =
\begin{cases}
\mathrm{sign}(t) \cdot 2^{\lfloor \alpha \rfloor} & \text{以概率 } p(t), \\
\mathrm{sign}(t) \cdot 2^{\lceil \alpha \rceil} & \text{以概率 } 1 - p(t),
\end{cases}
\quad
p(t) = \frac{2^{\lceil \alpha \rceil} - |t|}{2^{\lceil \alpha \rceil} - 2^{\lfloor \alpha \rfloor}},\quad C_{\mathrm{nat}}(0) = 0.
$$
NC 不需要全局尺度标量，仅依赖各分量的幅值，适合分布式场景；实现可近似为“忽略尾数、按指数随机舍入”，与浮点表示兼容，计算开销小。在 FSDP 下，对 $g$ 逐分量做 $C_{\mathrm{nat}}$ 后做 reduce-scatter(sum)，再除以 $M$ 即得平均梯度分片。

**1-bit 量化与误差反馈。** 1-bit SGD（Seide 等）将每个梯度分量量化为 1 比特（符号），并配合误差反馈以补偿量化偏差：$q_i = \mathrm{sign}(\widetilde{g}_i)$，$\widehat{g}_i = \mu \cdot q_i$，其中 $\mu$ 为尺度（如 $\|\widetilde{g}\|_1/d$ 或按桶估计的局部均值），误差反馈按上式更新残差 $e_t$。**1-bit Seide** 在每列（或每块）上维护两个重构值 $a$、$b$（正、负分量的均值），只传输符号与 $a,b$，解码时按符号在 $a$ 与 $b$ 之间选择并求和，在相同 1-bit 通信下可减小重构误差。在 FSDP 中，1-bit 类方法需在 hook 内维护与当前梯度分片同形状的残差状态，显存与实现复杂度较高，适用于通信绝对占优且能承受额外状态的场景。

**实现要点与小结。** 在 FSDP 中，上述量化方法均通过通信 hook 在 reduce-scatter 前接入：输入为未分片的扁平梯度 $g \in \mathbb{R}^d$，输出为当前 rank 应写入的平均梯度分片 $g^{(r)} \in \mathbb{R}^{d/M}$。Hook 内部需完成“量化 → 聚合（由 reduce-scatter 完成）→ 反量化 → 除以 $M$”的流程，并保证尺度、码本等元数据在各 rank 一致。误差反馈若启用，需为每个被压缩的张量维护与 $g$ 同形的残差，并注意 FSDP 下不同参数组对应不同 $d$，残差需按调用时的张量形状正确初始化或重置。综上所述，本节给出了梯度通信量化在 FSDP 场景下的统一表述，并形式化了 INT8、QSGD、NC、1-bit（含 Seide）的量化公式与统计性质；后续将在此基础上进行实现与实验对比，评估各方法在收敛速度、通信节省与端到端训练时间上的权衡。
