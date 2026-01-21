# Natural Compression for Distributed Deep Learning

原PDF为多栏版式（已线性化）。

Samuel Horváth, Chen-Yu Ho, L’udovít Horváth, Atal Narayan Sahu, Marco Canini, Peter Richtárik  
Proceedings of Machine Learning Research, vol. 145: 1–40, 3rd Annual Conference on Mathematical and Scientific Machine Learning (MSML) 2022

*Majority of work done while SH was a Ph.D. student at KAUST.*  
*Majority of work done while LH was an MS student at Comenius University and a research intern at KAUST.*

---

## Abstract

Modern deep learning models are often trained in parallel over a collection of distributed machines to reduce training time. In such settings, communication of model updates among machines becomes a significant performance bottleneck, and various lossy update compression techniques have been proposed to alleviate this problem. In this work, we introduce a new, simple yet theoretically and practically effective compression technique: **natural compression** ($C_{\mathrm{nat}}$). Our technique is applied individually to all entries of the to-be-compressed update vector. It works by randomized rounding to the nearest (negative or positive) power of two, which can be computed in a “natural” way by ignoring the mantissa.

We show that compared to no compression, $C_{\mathrm{nat}}$ increases the second moment of the compressed vector by not more than the tiny factor $9/8$, which means that the effect of $C_{\mathrm{nat}}$ on the convergence speed of popular training algorithms, such as distributed SGD, is negligible. However, the communication savings enabled by $C_{\mathrm{nat}}$ are substantial, leading to $3$–$4\times$ improvement in overall theoretical running time. For applications requiring more aggressive compression, we generalize $C_{\mathrm{nat}}$ to **natural dithering**, which we prove is exponentially better than the common random dithering technique. Our compression operators can be used on their own or in combination with existing operators for a more aggressive combined effect while offering new state-of-the-art theoretical and practical performance.

**Keywords:** Distributed Optimization, Stochastic Optimization, Non-convex Optimization, Gradient Compression

---

## 1. Introduction

Modern deep learning models (He et al., 2016) are almost invariably trained in parallel or distributed environments, which is necessitated by the enormous size of the data sets and the dimension and complexity of the models required to obtain state-of-the-art performance. Our work focuses on the **data-parallel** paradigm, in which the training data is split across several workers capable of operating in parallel (Bekkerman et al., 2011; Recht et al., 2011).

Formally, we consider optimization problems of the form

$$
\min_{x \in \mathbb{R}^d} f(x) := \frac{1}{n} \sum_{i=1}^n f_i(x),
\tag{1}
$$

where $x \in \mathbb{R}^d$ represents the parameters of the model, $n$ is the number of workers, and $f_i : \mathbb{R}^d \to \mathbb{R}$ is a loss function composed of data stored on worker $i$. Typically, $f_i$ is modeled as $f_i(x) := \mathbb{E}_{\zeta \sim \mathcal{D}_i}[f_\zeta(x)]$, where $\mathcal{D}_i$ is the distribution of data stored on worker $i$, and $f_\zeta$ is the loss of model $x$ on data point $\zeta$. The distributions $\mathcal{D}_1, \ldots, \mathcal{D}_n$ can be different, which means that the functions $f_1, \ldots, f_n$ may have different minimizers.

### Distributed learning

Problem (1) is commonly solved by **distributed stochastic gradient descent (SGD)** (Robbins and Monro, 1951). In each iteration, every worker computes a stochastic gradient $g_i(x_k)$ locally and sends it to a master node, which aggregates

$$
 g_k = \sum_{i=1}^n g_i(x_k)
$$

and broadcasts $g_k$ back to the workers. Each worker then performs the update

$$
 x_{k+1} = x_k - \eta_k \frac{1}{n} g_k,
$$

where $\eta_k > 0$ is the stepsize.

A key bottleneck of this algorithm, and its many variants (e.g., mini-batching (Goyal et al., 2017), importance sampling (Horváth and Richtárik, 2019), momentum (Nesterov, 2013), variance reduction (Johnson and Zhang, 2013)), is the **communication cost** of the typically dense gradient vectors $g_i(x_k)$ (and in a parameter-server implementation, also of $g_k$). As $d$ is very large in modern deep learning, communicating $d$-dimensional float vectors dominates the training time in many practical systems (Seide et al., 2014; Alistarh et al., 2017; Zhang et al., 2017; Lin et al., 2018; Lim et al., 2018).

### Communication reduction

Two largely orthogonal directions have been explored to mitigate this bottleneck:

1. **More local computation per communication round**, e.g. by
   - large mini-batches (Goyal et al., 2017),
   - local subproblems (Shamir et al., 2014),
   - reduced communication frequency (McDonald et al., 2009; Stich, 2018).

2. **Reduction of message size** via lossy compression (quantization or sparsification):
   - lower-precision floating point (Gupta et al., 2015; Na et al., 2017),
   - random dithering and quantization (Seide et al., 2014; Alistarh et al., 2017; Wen et al., 2017),
   - random sparsification (Suresh et al., 2017; Konečný and Richtárik, 2018),
   - structured sparsity via coordinate/block descent (Fercoq et al., 2014).

A fundamental issue is the **compression–variance trade-off**: aggressive compression reduces the number of bits communicated but increases the variance of the compressed gradient, which slows convergence. The key is therefore to design compressors that reduce communication **without** significantly increasing the variance.

Outside of distributed optimization, such compression operators are also important in **quantization theory** and **control theory** (Elia and Mitter, 2001; Sun and Goyal, 2011; Sun et al., 2012).

### Summary of contributions

We summarize our main contributions informally:

- **New compression operators.** We propose **natural compression** $C_{\mathrm{nat}}$ and its generalization **natural dithering** $D^{p,s}_{\mathrm{nat}}$. Natural compression rounds each scalar independently to a nearby power of two in an unbiased fashion. We show that $C_{\mathrm{nat}}$ belongs to a class $\mathcal{B}(\omega)$ of unbiased compressors with bounded second moment, with a remarkably small variance parameter $\omega = 1/8$ (Theorem 3). This means it has negligible effect on the convergence of SGD-like methods, but yields $3.56\times$ and $5.82\times$ communication savings for float32 and float64 respectively. Natural dithering is shown to be **exponentially better** than standard random dithering (Theorem 8).

- **State-of-the-art compression under second-moment budget.** Given a budget on the second moment parameter $\omega + 1$ (see Equation (3)), our operators achieve the **largest compression factor**, i.e., fewest transmitted bits, compared to prior sparsification and dithering schemes (Figure&nbsp;1).

- **Lightweight implementation.** Implementing $C_{\mathrm{nat}}$ on top of IEEE 754 floating point is simple: it essentially amounts to ignoring the mantissa and possibly increasing the exponent by one. The compression can be implemented via bit manipulations with negligible computational cost.

- **Compatibility with in-network aggregation (INA).** In the SwitchML system (Sapio et al., 2021), programmable switches can only perform integer additions. Our compressors naturally output integer-encoded values (sign bits and exponent bits), making them the first provably sound compressors compatible with SwitchML-style INA.

- **Bidirectional compression for distributed SGD.** We give the first convergence analysis of non-convex distributed SGD with compression **both at workers and at the master** (Algorithm&nbsp;1). The analysis holds for any compressors in $\mathcal{B}(\omega)$ and yields convergence rates with linear speedup in the number of nodes.

- **Improved total complexity.** We show that for our compressors the increase in iteration count due to compression is more than offset by the per-iteration communication savings. As a result, the **overall theoretical training time decreases** (Theorem&nbsp;9 and Table&nbsp;1). Among previously known unbiased compressors, only standard dithering (QSGD, Alistarh et al., 2017) had such guarantees; natural dithering is exponentially better.

- **Experiments.** Empirical results confirm the theory: $C_{\mathrm{nat}}$ matches the convergence of uncompressed training while significantly reducing wall-clock time and total transmitted data across a range of models and datasets. Composing $C_{\mathrm{nat}}$ with other compressors such as TopK or standard dithering further improves the communication–accuracy trade-off.

---

## 2. Natural Compression

We now define our basic compression operator $C_{\mathrm{nat}}$ and show its main properties.

### 2.1 Definition

Natural compression is first defined for scalars and then applied element-wise to vectors.

Given $t \in \mathbb{R}$, $C_{\mathrm{nat}}(t)$ is a random variable obtained by **randomized logarithmic rounding** of $t$ to the nearest power of two.

For nonzero $t$, let $\alpha = \log_2 |t| \in \mathbb{R}$ so that $|t| = 2^{\alpha}$. Then $2^{\lfloor \alpha \rfloor} \le |t| \le 2^{\lceil \alpha \rceil}$. We round $t$ to either $\operatorname{sign}(t) 2^{\lfloor \alpha \rfloor}$ or $\operatorname{sign}(t) 2^{\lceil \alpha \rceil}$ with appropriate probabilities chosen so that the resulting random variable is **unbiased**.

When $t = 0$ we set $C_{\mathrm{nat}}(0) = 0$.

#### Definition 1 (Natural compression)

The **natural compression** operator $C_{\mathrm{nat}} : \mathbb{R} \to \mathbb{R}$ is defined by

$$
C_{\mathrm{nat}}(0) = 0,
$$

and for $t \neq 0$,

$$
C_{\mathrm{nat}}(t) =
\begin{cases}
\operatorname{sign}(t)\, 2^{\lfloor \log_2 |t| \rfloor}, & \text{with probability } p(t), \\
\operatorname{sign}(t)\, 2^{\lceil \log_2 |t| \rceil}, & \text{with probability } 1 - p(t),
\end{cases}
\tag{2}
$$

where

$$
p(t) := \frac{2^{\lceil \log_2 |t| \rceil} - |t|}{2^{\lceil \log_2 |t| \rceil} - 2^{\lfloor \log_2 |t| \rfloor}}.
$$

Equivalently, we can write

$$
C_{\mathrm{nat}}(t) = \operatorname{sign}(t)\, 2^{\lfloor \log_2 |t| \rfloor} (1 + \lambda(t)),
$$

where $\lambda(t) \sim \operatorname{Bernoulli}(1 - p(t))$.

For vectors $x = (x_1, \ldots, x_d)^\top \in \mathbb{R}^d$ we apply $C_{\mathrm{nat}}$ element-wise:

$$
\bigl(C_{\mathrm{nat}}(x)\bigr)_i := C_{\mathrm{nat}}(x_i), \quad i = 1, \ldots, d.
$$

**Example.** For $t=2.5$, we have $2^1 = 2 \le 2.5 \le 4 = 2^2$. Natural compression rounds 2.5 to $2$ with probability $3/4$ and to $4$ with probability $1/4$, so that $\mathbb{E}[C_{\mathrm{nat}}(2.5)] = 2.5$.

![Figure 2](pdf_to_md/assets/nc/fig2.png)

*Figure 2: Illustration of natural compression applied to $t = 2.5$; the value is rounded to 2 with probability $3/4$ and to 4 with probability $1/4$ so that the operator remains unbiased.*

### 2.2 Compression operators with bounded second moment

We work within a standard class of unbiased compressors with bounded second moment (Jiang and Agrawal, 2018; Khirirat et al., 2018; Horváth et al., 2019).

#### Definition 2 (Compression operators)

A random mapping $C : \mathbb{R}^d \to \mathbb{R}^d$ is called a **compression operator** if, for all deterministic $x \in \mathbb{R}^d$,

$$
\mathbb{E}[C(x)] = x,
\qquad
\mathbb{E}\bigl[\|C(x)\|^2\bigr] \le (\omega + 1) \|x\|^2
\tag{3}
$$

for some finite $\omega \ge 0$. In this case we write $C \in \mathcal{B}(\omega)$.

The condition (3) implies that the variance of $C$ is bounded:

$$
\mathbb{E}\bigl[\|C(x) - x\|^2\bigr] \le \omega \|x\|^2.
$$

Many commonly used compressors (random sparsification, random dithering, etc.) fall into $\mathcal{B}(\omega)$ for suitable $\omega$.

### 2.3 Variance of natural compression

The main result of this section is that $C_{\mathrm{nat}}$ has an extremely small variance.

#### Theorem 3

$$
C_{\mathrm{nat}} \in \mathcal{B}\left(\tfrac{1}{8}\right).
$$

That is, $C_{\mathrm{nat}}$ is unbiased and satisfies

$$
\mathbb{E}\bigl[\|C_{\mathrm{nat}}(x)\|^2\bigr] \le \frac{9}{8} \|x\|^2, \quad \forall x \in \mathbb{R}^d.
$$

The proof proceeds by analyzing the scalar case and then extending to vectors by linearity.

By contrast, a seemingly natural unbiased compressor $C_{\mathrm{int}}$ that rounds to the nearest **integer** (rather than power of two) does **not** belong to $\mathcal{B}(\omega)$ for any finite $\omega$.

#### Theorem 4

There is no finite $\omega \ge 0$ such that $C_{\mathrm{int}} \in \mathcal{B}(\omega)$.

Intuitively, near zero the second moment of $C_{\mathrm{int}}$ blows up.

### 2.4 Implementation on IEEE 754 floats

Natural compression is extremely efficient on standard IEEE 754 floating point representations.

In single precision (binary32), a float is represented by a sign bit $s$, an 8-bit exponent $e$, and a 23-bit mantissa $m$:

$$
 t = (-1)^s \cdot 2^{e-127} \cdot (1 + m),
$$

with $m \in [0,1)$ encoded in fractional bits.

For such a number, natural compression acts as

- with probability $1 - m$, keep the sign and exponent, set mantissa to 0, yielding $(-1)^s 2^{e-127}$;
- with probability $m$, increase exponent by 1, set mantissa to 0, yielding $(-1)^s 2^{e-126}$.

Thus, **apart from drawing a random bit**, $C_{\mathrm{nat}}$ can be implemented entirely by cheap bit operations: clearing the mantissa and possibly incrementing the exponent.

![Figure 4](pdf_to_md/assets/nc/fig4.png)

*Figure 4: IEEE 754 single-precision binary32 representation of $t = -2.75$ showing sign, exponent, and mantissa bits.*

When communicating compressed updates, we only need to send the sign bit and the exponent bits. For binary32 this amounts to 9 bits per scalar (1 sign + 8 exponent bits), a $3.56\times$ reduction compared to 32 bits. For binary64 we need 12 bits (1 sign + 11 exponent bits), a $5.82\times$ reduction compared to 64 bits.

### 2.5 Composition with other compressors

Natural compression can be composed with other compressors and retains a bounded second moment.

#### Theorem 5 (Composition)

If $C_1 \in \mathcal{B}(\omega_1)$ and $C_2 \in \mathcal{B}(\omega_2)$, then the composition

$$
(C_1 \circ C_2)(x) := C_1(C_2(x))
$$

belongs to $\mathcal{B}(\omega_{12})$ with

$$
\omega_{12} = \omega_1\omega_2 + \omega_1 + \omega_2.
$$

In particular, if $C \in \mathcal{B}(\omega)$, then

$$
C_{\mathrm{nat}} \circ C \in \mathcal{B}\left(\frac{9}{8}\,\omega + \frac{1}{8}\right).
$$

Since composing with $C_{\mathrm{nat}}$ reduces each scalar to (sign, exponent) while increasing the variance parameter only slightly, it is attractive to apply $C_{\mathrm{nat}}$ **on top of** other compressors such as sparsifiers or standard dithering.

![Figure 1](pdf_to_md/assets/nc/fig1.png)

*Figure 1: Communication cost (bits per coordinate) vs. second moment factor $\omega + 1$ for several compressors applied to a vector of dimension $d=10^6$. Natural compression ($C_{\mathrm{nat}}$) and natural dithering ($D^{p,s}_{\mathrm{nat}}$) achieve favorable trade-offs compared to prior methods.*

Table&nbsp;1 (reproduced below) summarizes the overall speedups achievable by combining various worker-side compressors $C_{W_i}$ with distributed SGD under equal target accuracy.

**Table 1: Overall worker-to-master speedup for different compressors**

| Approach        | $C_{W_i}$          | No. iterations $T'(\omega_W)$           | Bits per iter. (worker $\to$ master) | Speedup factor (vs. no compression) |
|-----------------|--------------------|-----------------------------------------|--------------------------------------|-------------------------------------|
| Baseline        | identity           | $1$                                     | $32d$                                | $1$                                 |
| New             | $C_{\mathrm{nat}}$ | $(9/8)^\theta$                          | $9d$                                 | $3.2$–$3.6\times$                  |
| Sparsification  | $S_q$              | $(d/q)^\theta$                          | $(33 + \log_2 d)q$                  | $0.6$–$6.0\times$                  |
| New             | $C_{\mathrm{nat}} \circ S_q$ | $(9d/8q)^\theta$              | $(10 + \log_2 d)q$                  | $1.0$–$10.7\times$                 |
| Dithering       | $D^{p,2^{s-1}}_{\mathrm{sta}}$ | $(1 + \kappa d^{1/r} 2^{1-s})^\theta$ | $31 + d(2 + s)$                     | $1.8$–$15.9\times$                 |
| New             | $D^{p,s}_{\mathrm{nat}}$ | $(9/8 + \kappa d^{1/r} 2^{1-s})^\theta$ | $31 + d(2 + \log_2 s)$           | $4.1$–$16.0\times$                 |

Here $\theta \in (0,1]$ reflects dependence on the number of workers, $q$ is the expected sparsity level, $p$ is the norm parameter used in dithering, $r = \min\{p,2\}$, and $\kappa = \min\{1, d^{1/2} 2^{1-s}\}$.

---

## 3. Natural Dithering

Motivated by natural compression, we now introduce **natural dithering**, a new random dithering operator that yields much better variance–compression trade-offs than standard dithering.

### 3.1 General dithering framework

We first define a general family of dithering operators.

Let $1 \le p \le \infty$ and $\|x\|_p$ denote the $p$-norm. Consider levels

$$
0 = \ell_s < \ell_{s-1} < \cdots < \ell_1 < \ell_0 = 1.
$$

#### Definition 6 (General dithering)

The **general dithering operator** with respect to $\|\cdot\|_p$ and levels $\{\ell_j\}_{j=0}^s$, denoted $D^{C,p,s}_{\mathrm{gen}}$, is defined as follows. For $x \in \mathbb{R}^d$:

- If $x = 0$, set $D^{C,p,s}_{\mathrm{gen}}(x) = 0$.
- Otherwise, let $y_i = |x_i| / \|x\|_p$. Find $u \in \{0,1,\ldots,s-1\}$ such that $\ell_{u+1} \le y_i \le \ell_u$. Define a random variable $\xi(y_i)$ that equals $\ell_u$ with probability
  $$
  \frac{y_i - \ell_{u+1}}{\ell_u - \ell_{u+1}}
  $$
  and equals $\ell_{u+1}$ with the complementary probability.

Then the compressed vector is

$$
\bigl(D^{C,p,s}_{\mathrm{gen}}(x)\bigr)_i = C(\|x\|_p) \cdot \operatorname{sign}(x_i) \cdot \xi(y_i),
$$

where $C \in \mathcal{B}(\omega)$ is an outer norm compressor (possibly identity).

Standard random dithering (Goodall, 1951; Roberts, 1962) is recovered by choosing a **linear partition** of $[0,1]$ into $u$ equal subintervals and setting $C$ to the identity. For instance, QSGD (Alistarh et al., 2017) corresponds to $D^{2,s}_{\mathrm{sta}}$ with uniform levels; TernGrad (Wen et al., 2017) corresponds to $D^{\infty,1}_{\mathrm{sta}}$.

### 3.2 Natural dithering

Natural dithering is obtained by using a **geometric** partition of $[0,1]$ compatible with powers of two.

We define the levels

$$
\ell_{s-1} = 2^{1-s}, \; \ell_{s-2} = 2^{2-s}, \; \ldots, \; \ell_1 = 2^{-1}, \; \ell_0 = 1,
$$

and apply the general dithering scheme with $C$ equal to the identity (or to $C_{\mathrm{nat}}$ when we also want to compress the norm).

The resulting operator is denoted $D^{p,s}_{\mathrm{nat}}$. Intuitively, we quantize the **normalized** entries $y_i = |x_i|/\|x\|_p$ to the nearest powers of two, again in an unbiased fashion.

![Figure 3](pdf_to_md/assets/nc/fig3.png)

*Figure 3: Randomized rounding for natural (left) and standard (right) dithering with $s=3$ levels applied to the same scalar $t = 3/8$.*

As with $C_{\mathrm{nat}}$, the mantissa of the floating-point representation is ignored and only exponents (and signs) are communicated.

#### Theorem 7

Natural dithering belongs to $\mathcal{B}(\omega)$ with

$$
D^{p,s}_{\mathrm{nat}} \in \mathcal{B}(\omega), \quad
\omega = \frac{1}{8} + d^{1/r} 2^{1-s} \min\bigl\{1, d^{1/r} 2^{1-s}\bigr\}, \quad r = \min\{p,2\}.
$$

This is dramatically better than standard dithering for the same number of levels.

#### Theorem 8 (Natural vs. standard dithering)

Fix $s$. For a given vector $x$ and norm parameter $p$:

- For the **same number of levels** $s$, natural dithering $D^{p,s}_{\mathrm{nat}}$ has $O(2^{s-1}/s)$ times **smaller variance** than standard dithering $D^{p,s}_{\mathrm{sta}}$.
- For a given variance budget $\omega$, standard dithering must use $u = 2^{s-1}$ levels while natural dithering needs only $s$ levels, an **exponential reduction** in the number of levels.

![Figure 22](pdf_to_md/assets/nc/fig22.png)

*Figure 22: 1D visualization comparing natural dithering $D^{p,s}_{\mathrm{nat}}$ and standard dithering $D^{p,u}_{\mathrm{sta}}$ with $u=2^{s-1}$ for $s=4$. Standard dithering uses 8 levels in $[0,1]$, while natural dithering uses only 4 levels that are a subset of the standard ones.*

(Additional detailed variance comparisons and empirical results are reported in Appendix&nbsp;A.2 and Figures&nbsp;16–18.)

---

## 4. Distributed SGD with Bidirectional Compression

We now incorporate our compressors into distributed SGD, allowing compression both at the workers and at the master.

Assume each worker $i$ has access to unbiased stochastic gradients $g_i(x_k)$ for $f_i$ with bounded variance $\sigma_i^2$ and similarity constants $\zeta_i^2$ (measuring how different $\nabla f_i$ is from the global gradient $\nabla f$). Let

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^n \sigma_i^2, \qquad
\zeta^2 = \frac{1}{n} \sum_{i=1}^n \zeta_i^2.
$$

Assume $f$ is $L$-smooth.

### 4.1 Algorithm

We consider the following distributed SGD algorithm with **bidirectional compression**.

#### Algorithm 1: Distributed SGD with bidirectional compression

- **Input:** learning rates $\{\eta_k\}_{k=0}^{T-1} > 0$, initial vector $x_0$.

For $k = 0,1,\ldots,T-1$:

1. **Worker side (in parallel over $i = 1,\ldots,n$):**
   - Compute a stochastic gradient $g_i(x_k)$ of $f_i$ at $x_k$.
   - Compress it: $\Delta_k^i = C_{W_i}(g_i(x_k))$.

2. **Master side:**
   - Aggregate: $\Delta_k = \sum_{i=1}^n \Delta_k^i$.
   - Compress aggregated update: $g_k = C_M(\Delta_k)$.
   - Broadcast $g_k$ to all workers.

3. **Worker side (in parallel):**
   - Update local copy: $x_{k+1} = x_k - \eta_k \frac{1}{n} g_k$.

We assume $C_M \in \mathcal{B}(\omega_M)$ and $C_{W_i} \in \mathcal{B}(\omega_{W_i})$. Let $\omega_W = \max_i \omega_{W_i}$.

Define

$$
\alpha = \frac{(\omega_M + 1)(\omega_W + 1)}{n} \sigma^2
           + \frac{(\omega_M + 1)\omega_W}{n} \zeta^2,
$$

$$
\beta = 1 + \omega_M + \frac{(\omega_M + 1)\omega_W}{n}.
\tag{4}
$$

### 4.2 Convergence guarantee

#### Theorem 9

Let $C_M \in \mathcal{B}(\omega_M)$ and $C_{W_i} \in \mathcal{B}(\omega_{W_i})$ and assume $f$ is $L$-smooth. Choose constant stepsize $\eta_k = \eta \in (0, 2/(\beta L))$. Let $a$ be a random index chosen uniformly from $\{0,1,\ldots,T-1\}$. Then

$$
\mathbb{E}\bigl[\|\nabla f(x_a)\|^2\bigr]
\le
\frac{2\bigl(f(x_0) - f(x^*)\bigr)}{\eta (2 - \beta L \eta) T}
+ \frac{\alpha L \eta}{2 - \beta L \eta},
\tag{5}
$$

where $x^*$ is an optimal solution of (1).

In particular, for any $\varepsilon > 0$, choosing

$$
\eta = \frac{\varepsilon}{L(\alpha + \varepsilon \beta)}
$$

and

$$
T \ge \frac{2L \bigl(f(x_0) - f(x^*)\bigr)(\alpha + \varepsilon \beta)}{\varepsilon^2}
$$

ensures that $\mathbb{E}\|\nabla f(x_a)\|^2 \le \varepsilon$.

The bound (5) shows $O(1/T)$ convergence of the expected gradient norm towards a plateau controlled by $\alpha$. More compression (larger $\omega_M$, $\omega_W$) increases $\alpha$ and thus the asymptotic gradient norm, but also reduces the per-iteration communication cost. Table&nbsp;1 and related tables in Appendix&nbsp;D.7 quantify how the total runtime (iterations × bits per iteration) improves when using $C_{\mathrm{nat}}$ and $D^{p,s}_{\mathrm{nat}}$.

In the special case of identical data across workers (so $\zeta=0$), the iteration complexity scales as

$$
T(\omega_M, \omega_W)
\propto (\omega_M + 1)(\omega_W + 1) \frac{\sigma^2}{n} + (\omega_M + 1) \varepsilon.
$$

The relative slowdown compared to no compression is bounded by

$$
\frac{T(\omega_M, \omega_W)}{T(0,0)} \in (\omega_M + 1, (\omega_M + 1)(\omega_W + 1)],
$$

with the upper bound approached for small $n$ or very small $\varepsilon$, and the lower bound as $n \to \infty$.

---

## 5. Experiments

We briefly summarize the main experimental findings; detailed plots and additional experiments are given in the Appendices.

### 5.1 CIFAR-10 (ResNet-110 and AlexNet)

The authors train ResNet-110 and AlexNet on CIFAR-10 using a proof-of-concept implementation that integrates natural compression with in-network aggregation (SwitchML-style architecture).

![Figure 5 Placeholder](#)

*Figure 5: Training loss and test accuracy of ResNet-110 and AlexNet on CIFAR-10. Natural compression ($C_{\mathrm{nat}}$) matches the baseline accuracy while significantly reducing wall-clock time.*

Results show that:

- For ResNet-110, $C_{\mathrm{nat}}$ reduces training time by about 26% relative to no compression.
- For AlexNet, the reduction is about 66%.
- Final accuracies match or slightly improve upon the baseline with the same hyperparameters.

![Figure 6 Placeholder](#)

*Figure 6: Training throughput speedup for various CNN architectures on ImageNet. Bars compare In-Network Aggregation, deterministic rounding, and stochastic $C_{\mathrm{nat}}$.*

### 5.2 Composition with natural dithering

![Figure 7 Placeholder](#)

*Figure 7: Train loss and test accuracy of VGG11 on CIFAR-10 comparing standard dithering $D^{2,27}_{\mathrm{sta}}$ with natural dithering $D^{2,8}_{\mathrm{nat}}$ composed with $C_{\mathrm{nat}}$.*

For the same effective variance, natural dithering uses exponentially fewer levels than standard dithering, yielding substantial reductions in communicated bits without harming convergence.

### 5.3 ImageNet and NCF

The paper further reports experiments on:

- **ResNet-50 on ImageNet**, showing that $C_{\mathrm{nat}}$ preserves accuracy while reducing communication and maintaining good weak scaling when increasing the number of workers from 8 to 16.
- **Neural Collaborative Filtering (NCF) on MovieLens-20M**, where $C_{\mathrm{nat}}$ achieves the same hit-rate@10 as the baseline with significantly less communication.

![Figure 10 Placeholder](#)

*Figure 10: ResNet-50 on ImageNet with and without $C_{\mathrm{nat}}$ using 8 and 16 workers.*

![Figure 11 Placeholder](#)

*Figure 11: NCF on MovieLens-20M with and without $C_{\mathrm{nat}}$.*

### 5.4 Combination with TopK and OmniReduce

The compressor **TopK-$C_{\mathrm{nat}}$** applies TopK sparsification and then natural compression to the selected coordinates.

**Table 2: TopK vs. TopK-$C_{\mathrm{nat}}$ on CIFAR-10 (ResNet-18)**

| $K$ sparsity | Without $C_{\mathrm{nat}}$ (bits, accuracy)    | With $C_{\mathrm{nat}}$ (bits, accuracy)         |
|--------------|-----------------------------------------------|-----------------------------------------------|
| 0.39%        | $(0.78\%, 93.72 \pm 0.07\%)$                  | $(0.48\%, 93.93 \pm 0.26\%)$                  |
| 1.56%        | $(3.12\%, 94.21 \pm 0.06\%)$                 | $(1.95\%, 94.47 \pm 0.13\%)$                 |
| 6.25%        | $(12.5\%, 93.93 \pm 0.53\%)$                 | $(7.81\%, 94.13 \pm 0.26\%)$                 |

**Table 3: TopK-$C_{\mathrm{nat}}$ vs. other compressors at matched communication budget**

| Method           | Rel. comm. | Test acc. (SOTA)       | $K$ for TopK-$C_{\mathrm{nat}}$ | TopK-$C_{\mathrm{nat}}$ acc.      |
|------------------|-----------:|------------------------|-------------------------------:|-----------------------------------|
| SGD              |   100%     | $93.69 \pm 0.32\%$    | N/A                            | N/A                               |
| SignSGD          |   3.12%    | $93.47 \pm 0.29\%$    | 2.5%                           | $94.28 \pm 0.12\%$               |
| EF-SignSGD       |   3.12%    | $93.81 \pm 0.06\%$    | 2.5%                           | $94.28 \pm 0.12\%$               |
| PowerSGD rank-2  |   0.74%    | $94.26 \pm 0.22\%$    | 0.6%                           | $94.23 \pm 0.01\%$               |
| PowerSGD rank-7  |   2.36%    | $94.24 \pm 0.09\%$    | 1.9%                           | $94.25 \pm 0.10\%$               |

Composing $C_{\mathrm{nat}}$ with **OmniReduce** (Fei et al., 2020), which performs block-based sparse collective communication, achieves up to $30\times$ speedup over NCCL in micro-benchmarks when gradients are very sparse.

![Figure 12 Placeholder](#)

*Figure 12: Violin plot comparing OmniReduce with and without $C_{\mathrm{nat}}$ and NCCL on synthetic sparse tensors.*

---

## 6. Conclusions and Future Work

The paper introduces two new compression operators—natural compression $C_{\mathrm{nat}}$ and natural dithering $D^{p,s}_{\mathrm{nat}}$—and provides a general convergence theory for distributed SGD with **bidirectional** compression (at both workers and master). These compressors:

- have small variance (bounded second moment with $\omega = 1/8$ for $C_{\mathrm{nat}}$),
- are extremely simple and efficient to implement on IEEE 754 hardware,
- integrate naturally with in-network aggregation architectures,
- can be composed with existing compressors such as sparsifiers and standard dithering to improve overall performance,
- yield provable and empirically validated reductions in **total training time**.

Extensions left for future work include:

- very large-scale experiments with thousands of nodes,
- applications beyond centralized data-parallel SGD,
- using natural compression and dithering in other optimization algorithms (e.g., variance-reduced methods, adaptive methods).

---

## References

(References are reproduced in the same order as in the original paper. For brevity we keep them as plain text citations here.)

Alham Fikri Aji and Kenneth Heafield. Sparse communication for distributed gradient descent. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, 2017.

Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In *Advances in Neural Information Processing Systems*, 2017.

Dan Alistarh, Torsten Hoefler, Mikael Johansson, Nikola Konstantinov, Sarit Khirirat, and Cedric Renggli. The convergence of sparsified gradient methods. In *Advances in Neural Information Processing Systems 31*, 2018.

Ron Bekkerman, Mikhail Bilenko, and John Langford. *Scaling up machine learning: Parallel and distributed approaches*. Cambridge University Press, 2011.

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signSGD: Compressed optimisation for non-convex problems. In *Proceedings of the 35th International Conference on Machine Learning*, 2018.

Sébastien Bubeck. Convex optimization: Algorithms and complexity. *Foundations and Trends in Machine Learning*, 2015.

N. Dryden, T. Moon, S. A. Jacobs, and B. V. Essen. Communication quantization for data-parallel training of deep neural networks. In *MLHPC*, 2016.

Nicola Elia and Sanjoy K. Mitter. Stabilization of linear systems with limited information. *IEEE Transactions on Automatic Control*, 2001.

Jiawei Fei, Chen-Yu Ho, Atal Narayan Sahu, Marco Canini, and Amedeo Sapio. Efficient sparse collective communication and its application to accelerate distributed deep learning. Technical report, KAUST, 2020.

Olivier Fercoq, Zheng Qu, Peter Richtárik, and Martin Takáč. Fast distributed coordinate descent for minimizing non-strongly convex losses. In *IEEE MLSP*, 2014.

Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. *SIAM Journal on Optimization*, 2013.

W. M. Goodall. Television by pulse code modulation. *Bell System Technical Journal*, 1951.

Priya Goyal et al. Accurate, large minibatch SGD: training ImageNet in 1 hour. arXiv:1706.02677, 2017.

Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In *ICML*, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

Xiangnan He et al. Neural collaborative filtering. In *WWW*, 2017.

Samuel Horváth and Peter Richtárik. Nonconvex variance reduced optimization with arbitrary sampling. In *ICML*, 2019.

Samuel Horváth, Dmitry Kovalev, Konstantin Mishchenko, Sebastian Stich, and Peter Richtárik. Stochastic distributed learning with gradient quantization and variance reduction. arXiv:1904.05115, 2019.

Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In *CVPR*, 2017.

Itay Hubara et al. Quantized neural networks: Training neural networks with low precision weights and activations. *JMLR*, 2017.

Peng Jiang and Gagan Agrawal. A linear speedup analysis of distributed deep learning with sparse and quantized communication. In *NeurIPS 31*, 2018.

Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In *NeurIPS*, 2013.

Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian U. Stich, and Martin Jaggi. Error feedback fixes SignSGD and other gradient compression schemes. In *ICML*, 2019.

Sarit Khirirat, Hamid Reza Feyzmahdavian, and Mikael Johansson. Distributed learning with compressed gradients. arXiv:1806.06573, 2018.

D. P. Kingma and J. Ba. ADAM: A Method for Stochastic Optimization. In *ICLR*, 2015.

Jakub Konečný and Peter Richtárik. Randomized distributed mean estimation: accuracy vs communication. *Frontiers in Applied Mathematics and Statistics*, 2018.

Hyeontaek Lim, David G. Andersen, and Michael Kaminsky. 3LC: Lightweight and effective traffic compression for distributed machine learning. arXiv:1802.07389, 2018.

Yujun Lin, Song Han, Huizi Mao, Yu Wang, and Bill Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In *ICLR*, 2018.

Ryan McDonald et al. Efficient large-scale distributed training of conditional maximum entropy models. In *NeurIPS 22*, 2009.

Konstantin Mishchenko, Eduard Gorbunov, Martin Takáč, and Peter Richtárik. Distributed learning with compressed gradient differences. arXiv:1901.09269, 2019.

T. Na et al. On-chip training of recurrent neural networks with limited numerical precision. In *IJCNN*, 2017.

Yurii Nesterov. *Introductory lectures on convex optimization: A basic course*. 2013.

Benjamin Recht, Christopher Ré, Stephen Wright, and Feng Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In *NeurIPS*, 2011.

Herbert Robbins and Sutton Monro. A stochastic approximation method. *Annals of Mathematical Statistics*, 1951.

L. Roberts. Picture coding using pseudo-random noise. *IRE Transactions on Information Theory*, 1962.

Amedeo Sapio et al. Scaling distributed machine learning with in-network aggregation. In *NSDI*, 2021.

Frank Seide et al. 1-bit stochastic gradient descent and application to data-parallel distributed training of speech DNNs. In *Interspeech 2014*, 2014.

Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate Newton-type method. In *ICML*, 2014.

Sebastian U. Stich. Local SGD converges fast and communicates little. arXiv:1805.09767, 2018.

Sebastian U. Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In *NeurIPS 31*, 2018.

John Z. Sun and Vivek K. Goyal. Scalar quantization for relative error. In *Data Compression Conference*, 2011.

John Z. Sun, Grace I. Wang, Vivek K. Goyal, and Lav R. Varshney. A framework for Bayesian optimality of psychophysical laws. *Journal of Mathematical Psychology*, 2012.

Ananda Theertha Suresh, Felix X. Yu, Sanjiv Kumar, and H. Brendan McMahan. Distributed mean estimation with limited communication. In *ICML*, 2017.

Hanlin Tang et al. DoubleSqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression. In *ICML*, 2019.

Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. PowerSGD: Practical low-rank gradient compression for distributed optimization. In *NeurIPS*, 2019.

Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-efficient distributed optimization. In *NeurIPS*, 2018.

Wei Wen et al. TernGrad: Ternary gradients to reduce communication in distributed deep learning. In *NeurIPS*, 2017.

Hantian Zhang et al. ZipML: Training linear models with end-to-end low precision, and a little bit of deep learning. In *ICML*, 2017.

Shuai Zheng, Ziyue Huang, and James Kwok. Communication-efficient distributed blockwise momentum SGD with error-feedback. In *NeurIPS*, 2019.

---

## Appendix

### A. Extra Experiments

#### A.1. Convergence Tests on CIFAR-10

We trained various DNNs on the CIFAR-10 dataset with and without $C_{\mathrm{nat}}$. In all experiments, we used Batch Normalization and no Dropout. The results (Figures 13, 14, 15) show significant speed-up without accuracy loss.

![Figure 13 Placeholder](#)
*Figure 13: DenseNet40 (k=12)*

![Figure 14 Placeholder](#)
*Figure 14: AlexNet (Batch sizes: 256, 512, 1024)*

![Figure 15 Placeholder](#)
*Figure 15: ResNet (#layers: 20, 44, 56)*

#### A.2. $D^{p,s}_{\mathrm{nat}}$ vs. $D^{p,u}_{\mathrm{sta}}$: Empirical Variance

We empirically confirm the theoretical variance properties from Theorem 8 using synthetic data.

![Figure 16 Placeholder](#)
*Figure 16: $D^{p,s}_{\mathrm{nat}}$ vs. $D^{p,u}_{\mathrm{sta}}$ with $u=s$. Natural dithering has dramatically smaller variance.*

![Figure 17 Placeholder](#)
*Figure 17: $D^{p,s}_{\mathrm{nat}}$ vs. $D^{p,u}_{\mathrm{sta}}$ with $u=2^{s-1}$. The empirical variances are nearly identical.*

![Figure 18 Placeholder](#)
*Figure 18: For large $s$, standard dithering can outperform natural dithering, as the former converges to the identity operator (variance 0) while the latter converges to $C_{\mathrm{nat}}$ (variance 1/8).*

#### A.3. Different Compression Operators

We compare our natural compressors with their non-natural counterparts on MNIST and CIFAR-10. Figures 19 and 20 show that natural versions achieve similar convergence in terms of epochs but with a substantial reduction in communicated bits.

![Figure 19 Placeholder](#)
*Figure 19: CIFAR10 with VGG11, comparing `None`, random sparsification, and random dithering with and without natural compression.*

![Figure 20 Placeholder](#)
*Figure 20: MNIST with 2 fully connected layers, comparing `None`, random sparsification, and random dithering with and without natural compression.*

### B. Experimental Setup

The setup details, including system implementation, hyperparameters, and hardware, are provided in the appendix.

![Figure 21 Placeholder](#)
*Figure 21: Histogram of exponents of gradients exchanged during training for ResNet110 (left) and AlexNet (right).*

### C. Details and Proofs for Sections 2 and 3

The appendix provides detailed proofs for:
- Theorem 3 (Variance of $C_{\mathrm{nat}}$)
- Theorem 4 (Unbounded variance of $C_{\mathrm{int}}$)
- Theorem 5 (Composition)
- Theorem 7 (Variance of $D^{p,s}_{\mathrm{nat}}$)
- Theorem 8 (Natural vs. standard dithering)

### D. Details and Proofs for Section 4

The appendix provides:
- Formal assumptions for the convergence analysis.
- Detailed proofs of lemmas used to establish Theorem 9.
- Analysis under different cost models and communication patterns.
- Formal definition of the random sparsification operator $S_q$.

### E. Limitations and Extensions

The paper's focus is on unbiased compressors, but the natural compression mechanism can also be combined with biased techniques like TopK sparsification.


