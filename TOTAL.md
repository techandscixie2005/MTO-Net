下面给出一版我认为最适合作为**第一篇文章最终执行版**的方案。它保留 MTO-Net 最有原创性的部分，同时把动力学模拟、过渡态搜索、波函数/哈密顿量重构、完整量子化学电子结构这些容易拖散主线的内容全部降级为“可扩展方向”或“能力边界说明”。

---

# MTO-Net v6-lite-response

## Valence-Adaptive Molecular Tensor Orbitals for Equivariant Molecular Response Learning

**面向分子响应性质学习的价电子自适应分子张量轨道网络**

## 0. 一句话总纲

[
\boxed{
\textbf{
MTO-Net 将原子级等变张量场重组为一组价电子自适应、无几何中心、带符号组合、具有门控非线性的分子级响应模式，并用这些模式统一预测和解释分子的偶极矩、极化率、IR、Raman 与 UV-Vis 谱学响应。
}
}
]

英文可以写成：

[
\boxed{
\textit{
MTO-Net reorganizes atom-centered equivariant tensor fields into valence-adaptive, center-free, signed and nonlinear molecular tensor orbitals, enabling stable and interpretable learning of molecular tensorial and spectroscopic responses.
}
}
]

这篇文章的核心不是“再做一个更强的等变 GNN”，而是提出一个新的中间表示问题：

[
\boxed{
\textbf{
等变分子模型内部能否形成稳定、可复用、可解释的分子级电子响应模式？
}
}
]

e3nn、Equiformer、MACE、DetaNet 等已有工作已经充分证明了 E(3)/SE(3)-equivariant 表示对 3D 分子建模、张量性质和谱图预测是有效的：e3nn 提供了 TensorProduct、spherical harmonics 等等变操作；Equiformer 将 Transformer 操作替换为等变对应物并引入 MLP attention 与 nonlinear message passing；MACE 通过 higher-body equivariant messages 提升了力场学习效率；DetaNet 展示了 E(3)-equivariance + self-attention 对 IR、Raman、UV-Vis、NMR 和多阶张量性质预测的有效性。([arXiv][1])

你的创新点应该明确定位为：

[
\boxed{
\textbf{
不是如何预测性质，而是如何在预测性质之前形成分子级、轨道式、稳定的响应表示。
}
}
]

---

# 1. 研究边界：主动砍掉“大一统量子化学”叙事

第一篇必须主动声明：

[
\boxed{
\textbf{
本文不重构波函数，不预测 Hamiltonian，不做过渡态搜索，不做分子动力学模拟，不主攻势能面，也不声称 MTO 是真实分子轨道。
}
}
]

推荐论文中的边界声明：

[
\boxed{
\textit{
We do not aim to reconstruct quantum-chemical wavefunctions, Hamiltonians, transition states, or molecular dynamics trajectories. Instead, we test whether atom-centered equivariant tensor fields can be reorganized into stable molecule-level response modes that support tensorial and spectroscopic property prediction.
}
}
]

中文：

[
\boxed{
\textbf{
本文不试图重构量子化学波函数、哈密顿量、过渡态或分子动力学轨迹；我们的目标是验证原子级等变张量场能否被重组为稳定的分子级响应模式，并支撑张量与谱学性质预测。
}
}
]

这样可以避免审稿人把你拉去和 DeepH、OrbNet、NequIP、MACE 在完全不同的任务上正面竞争。你的主线是 **representation and interpretability for molecular response learning**，不是 universal quantum chemistry engine。

---

# 2. 核心科学假设

本文的科学假设是：

[
\boxed{
\textbf{
分子响应性质并不只需要一个黑箱全局向量，而可以由一组稳定的、价电子自适应的分子级张量响应模式来组织。
}
}
]

也就是说，传统模型通常是：

[
{Z_i,\mathbf r_i}
\rightarrow
{h_i^{(l)}}
\rightarrow
\mathrm{pooling/readout}
\rightarrow
y.
]

而 MTO-Net 改为：

[
{Z_i,\mathbf r_i}
\rightarrow
{h_i^{(l)}}
\rightarrow
{\mathcal O_k^{(l)}, n_k}*{k=1}^{K(N*{\mathrm{val}})}
\rightarrow
\mu,\alpha,IR,Raman,UV.
]

其中 (\mathcal O_k) 是 **Molecular Tensor Orbital, MTO**。但必须严格表述为：

[
\boxed{
\textbf{
MTO 是等变隐空间中的分子级轨道式响应模式，而不是真实 Kohn-Sham 轨道或量子化学波函数。
}
}
]

更安全的英文定义：

[
\boxed{
\textit{
MTOs are molecule-level equivariant latent response modes inspired by molecular orbital organization, rather than physical molecular orbitals.
}
}
]

---

# 3. 第一篇主任务

第一篇只聚焦五类响应性质：

[
\boxed{
\mathcal T
==========

{\mu,\alpha,IR,Raman,UV}.
}
]

其中：

[
\mu \sim l=1,
]

[
\alpha \sim l=0\oplus l=2.
]

IR/Raman 通过 (\partial\mu/\partial Q_m)、(\partial\alpha/\partial Q_m) 与响应张量联系起来；UV-Vis 作为电子响应谱图任务，第一版直接预测 broadened spectrum，不强行拆成真实 occupied-to-virtual transitions。DetaNet 的结果也说明，直接预测 broad UV-Vis spectrum 是比从基态结构硬预测 oscillator strength 更稳的路线。([PubMed][2])

暂时不作为主任务：

[
E,\mathbf F,Hessian,\rho(\mathbf r),\psi,H_{\mathrm{DFT}},TS,MD.
]

这些内容可以在 Discussion 里一句话说明：

[
\boxed{
\textbf{
MTO bottleneck 原则上可以接入能量、力和密度等监督，但本文聚焦于响应性质中的稳定分子级模式学习。
}
}
]

---

# 4. 总体模型架构

最终模型分为六个模块：

[
\boxed{
\text{Input}
\rightarrow
\text{Equivariant Atomic Tensor Encoder}
\rightarrow
\text{Valence-Adaptive MTO Bank}
\rightarrow
\text{Center-Free Signed Nonlinear Assembly}
\rightarrow
\text{Charge-Conserving Activity Gate}
\rightarrow
\text{Response Readouts}.
}
]

具体流程：

[
{Z_i,\mathbf r_i}_{i=1}^N
\rightarrow
h_i
===

h_i^{(0)}
\oplus
h_i^{(1)}
\oplus
h_i^{(2)}
]

[
\rightarrow
\mathcal O_k
============

\mathcal O_k^{(0)}
\oplus
\mathcal O_k^{(1)}
\oplus
\mathcal O_k^{(2)}
]

[
\rightarrow
n_k
\rightarrow
\hat\mu,\hat\alpha,\hat I_{\mathrm{IR}},\hat I_{\mathrm{Raman}},\hat I_{\mathrm{UV}}.
]

第一版固定：

[
\boxed{
L_{\max}=2.
}
]

理由很清楚：(\mu) 是一阶向量响应，(\alpha) 可分解为标量各向同性部分与 (l=2) 的各向异性部分；IR/Raman 又依赖 (\mu,\alpha) 对振动坐标的导数。因此 (L_{\max}=2) 足以支撑主线。更高阶 (l=3) 可以留给超极化率、手性响应或二代模型。

---

# 5. 原子级等变张量编码器

输入：

[
\mathcal G={Z_i,\mathbf r_i}_{i=1}^N.
]

Backbone 输出：

[
h_i^{(l)}
\in
\mathbb R^{C_l\times(2l+1)},
\quad
l=0,1,2.
]

这里的 backbone 可以是：

[
\text{e3nn/TFN-like},
\quad
\text{Equiformer-like},
\quad
\text{MACE-like},
\quad
\text{DetaNet-like}.
]

但论文里必须写清楚：

[
\boxed{
\textbf{
MTO-Net 的创新不在 backbone，而在 atom-centered tensor fields 到 molecule-level response modes 的重组。
}
}
]

这可以防止审稿人把你的贡献误解成“又一个 backbone”。

工程建议：

* 第一版选择一个稳定、小型的 e3nn/Equiformer-like backbone；
* 不追求最强 backbone；
* backbone 的输出只需要保证 (l=0,1,2) 通道稳定；
* 所有非标量通道不得加 bias；
* tensor normalization 使用 norm-based normalization；
* 如果涉及响应导数，cutoff 至少使用 (C^2) 平滑函数。

---

# 6. Valence-Adaptive MTO Bank

固定 (K) 的瓶颈有明显问题：小分子容量过剩，大分子容量不足，而且没有电子结构直觉。因此 MTO 数量随价电子数变化。

定义价电子数：

[
N_{\mathrm{val}}
================

\sum_i v(Z_i)-Q.
]

对于第一版 neutral closed-shell 分子，(Q=0)。定义：

[
K_{\mathrm{occ}}
================

\left\lceil
\frac{N_{\mathrm{val}}}{2}
\right\rceil,
]

[
K_{\mathrm{resp}}
=================

\gamma K_{\mathrm{occ}},
]

[
\boxed{
K(N_{\mathrm{val}})
===================

(1+\gamma)
\left\lceil
\frac{N_{\mathrm{val}}}{2}
\right\rceil.
}
]

第一版主模型建议：

[
\gamma=1,
\quad
K\approx N_{\mathrm{val}}.
]

但注意表述：

[
\boxed{
\textbf{
K 不是真实轨道数，也不是电子数，而是随价电子复杂度增长的 response active capacity。
}
}
]

也就是说，(K) 是一种神经活性空间容量：

[
\boxed{
\textbf{
valence-adaptive latent active space.
}
}
]

这是文章最漂亮的设计之一，因为它既有物理直觉，又不把模型绑定到真实 MO。

---

# 7. Slot query 设计

由于 (K) 是 variable-(K)，不能简单为每个 slot 学一个固定 embedding。建议定义归一化 rank：

[
\tau_k=\frac{k}{K}.
]

并定义 slot type：

[
t_k=
\begin{cases}
\mathrm{core/occupied\text{-}like}, & k\le K_{\mathrm{occ}},\
\mathrm{response/virtual\text{-}like}, & k>K_{\mathrm{occ}}.
\end{cases}
]

slot query：

[
\boxed{
q_k
===

\mathrm{MLP}*q
\left[
\mathrm{PE}(\tau_k),
\mathrm{Emb}(t_k),
\mathrm{Emb}(N*{\mathrm{val}})
\right].
}
]

这里的 occupied-like / virtual-like 只是组织隐空间的软标签，不对应真实占据轨道和空轨道。

batch 实现：

[
K_{\max}=\max_b K_b.
]

所有 MTO 张量 padding 到 (K_{\max})，并维护 mask：

[
M_{bk}=\mathbf 1(k\le K_b).
]

所有 readout、diversity loss、activity gate、解释性分析都必须乘 (M_{bk})。

---

# 8. Center-Free Signed Nonlinear MTO Assembly

这是模型的核心。最终版不引入显式几何中心。

原因：

1. 几何信息已经由等变 backbone 编码在 (h_i^{(l)}) 中；
2. 显式 MTO center 会把 MTO 推向局域 blob，不利于表达共轭、离域、D-(\pi)-A 电荷转移等多中心响应；
3. 无中心 assembly 更符合“分子级响应模式”而不是“空间局域 token”的定位。

最终 assembly 分四步：

[
\boxed{
\text{shared invariant routing}
\rightarrow
\text{(l)-specific sign}
\rightarrow
\text{linear equivariant assembly}
\rightarrow
\text{residual gated equivariant relaxation}.
}
]

---

## 8.1 Shared invariant routing

[
u_{ki}
======

f_{\mathrm{route}}
\left(
h_i^{(0)},q_k,\mathrm{Emb}(Z_i)
\right).
]

[
\boxed{
a_{ki}
======

\mathrm{softmax}*i(u*{ki}).
}
]

其中 (a_{ki}\ge 0)，表示第 (k) 个 MTO 从第 (i) 个原子吸收信息的强度。

这个 routing 只使用 (l=0) 标量特征，所以是旋转不变量，不破坏等变性。

---

## 8.2 (l)-specific signed coefficient

为了表达同相/反相、成键/反键、节点结构，引入：

[
s_{ki}^{(l)}
============

\tanh
f_{\mathrm{sign},l}
\left(
h_i^{(0)},q_k,\mathrm{Emb}(Z_i)
\right).
]

然后：

[
\tilde c_{ki}^{(l)}
===================

a_{ki}s_{ki}^{(l)}.
]

归一化：

[
\boxed{
c_{ki}^{(l)}
============

A_k^{(l)}
\frac{
a_{ki}s_{ki}^{(l)}
}{
\sqrt{
\sum_j(a_{kj}s_{kj}^{(l)})^2+\varepsilon
}
}.
}
]

这里：

* (a_{ki})：贡献强度；
* (s_{ki}^{(l)})：相位式符号；
* (A_k^{(l)})：整体幅度；
* 归一化项：防止分子大小、原子数、slot collapse 导致范数不稳。

这个设计是 MTO 和普通 attention pooling 的根本区别：普通 attention 通常是正权重加权平均，而 MTO 允许带符号的节点结构。

---

## 8.3 Linear equivariant assembly

[
\boxed{
P_k^{(l)}
=========

\sum_i
c_{ki}^{(l)}
h_i^{(l)}.
}
]

因为 (c_{ki}^{(l)}) 是旋转不变量，(h_i^{(l)}) 是 (l)-阶等变张量，所以 (P_k^{(l)}) 严格保持等变性。

这一步是神经版 LCAO-like assembly：

[
\text{atomic tensor fields}
\rightarrow
\text{molecular tensor response modes}.
]

---

## 8.4 Residual gated equivariant nonlinearity

只做线性组合会让 MTO 太像 pooling。必须在 assembly 后加入非线性，但非线性不能破坏等变性。

先构造不变 summary：

[
I_k
===

\left[
P_k^{(0)},
|P_k^{(1)}|,
|P_k^{(2)}|,
n_k
\right].
]

对 (l=0)：

[
\boxed{
\mathcal O_k^{(0)}
==================

P_k^{(0)}
+
\alpha_0
\mathrm{MLP}_0(I_k).
}
]

对 (l>0)：

[
g_k^{(l)}
=========

\sigma
\left(
\mathrm{MLP}_{g,l}(I_k)
\right),
]

[
\boxed{
\mathcal O_k^{(l)}
==================

P_k^{(l)}
+
\alpha_l
g_k^{(l)}
\odot
W_lP_k^{(l)}.
}
]

其中 (W_l) 只混合 channel，不混合 (m) 分量，因此不破坏等变性。

初始化：

[
\alpha_l=10^{-3}\sim10^{-2}.
]

这样模型初始接近线性 LCAO-like assembly，训练后再学习非线性修正。

---

# 9. Charge-Conserving Activity Gate

我建议在论文里不要过度叫 occupation gate，而叫：

[
\boxed{
\textbf{
charge-conserving activity gate.
}
}
]

因为 (n_k) 不是物理占据数，而是 MTO slot 的软活性分配。

定义：

[
n_k
===

2\sigma\left(
\frac{\mu-\epsilon_k}{\theta}
\right)
=======

\frac{2}{1+\exp((\epsilon_k-\mu)/\theta)}.
]

约束：

[
\boxed{
\sum_{k=1}^K n_k=N_{\mathrm{val}}.
}
]

其中：

[
\epsilon_k=f_\epsilon(I_k)
]

是 MTO activity score，(\mu) 是每个分子动态求解的 chemical-potential-like scalar。

前向传播用二分法求 (\mu)，但不让 autograd 穿过二分迭代。反向传播使用隐函数求导。

定义：

[
d_k=
\frac{\partial n_k}{\partial \mu}
=================================

\frac{1}{\theta}
n_k
\left(
1-\frac{n_k}{2}
\right).
]

则：

[
\boxed{
\frac{\partial \mu}{\partial \epsilon_i}
========================================

\frac{d_i}{\sum_j d_j}.
}
]

进一步：

[
\boxed{
\frac{\partial n_k}{\partial \epsilon_i}
========================================

-d_k\delta_{ki}
+
\frac{d_kd_i}{\sum_jd_j}.
}
]

如果上游梯度为：

[
g_k=\frac{\partial \mathcal L}{\partial n_k},
]

则：

[
\boxed{
\frac{\partial \mathcal L}{\partial \epsilon_i}
===============================================

d_i
\left[
\frac{\sum_k g_kd_k}{\sum_jd_j}
-------------------------------

g_i
\right].
}
]

这个设计有两个好处：

1. 不需要保存二分迭代计算图；
2. 梯度自动满足电子数守恒方向：

[
\sum_k
\frac{\partial n_k}{\partial \epsilon_i}
========================================

0.

]

工程建议：

* (\theta) 第一版不要设为可学习参数；
* schedule：
  [
  \theta:0.5\rightarrow0.03;
  ]
* 不建议低于 (0.02)，否则梯度容易消失或不稳定；
* 对 ((\mu-\epsilon_k)/\theta) clamp 到 ([-30,30])；
* 第一篇只做 neutral closed-shell 分子；
* radicals 和 spin-separated gate 放第二版。

---

# 10. Response Readouts

## 10.1 偶极矩 (\mu)

偶极矩是向量，即 (l=1) 输出：

[
\boxed{
\hat\mu
=======

\sum_k
M_k
r_{\mu}(I_k,n_k)
\mathcal O_k^{(1)}.
}
]

这里 (r_\mu) 是标量权重，因此整体保持等变。

---

## 10.2 极化率 (\alpha)

极化率分为 isotropic scalar 和 anisotropic rank-2：

[
\alpha
======

\alpha_{\mathrm{iso}}
\oplus
\alpha_{\mathrm{aniso}}^{(2)}.
]

读出：

[
\hat\alpha_{\mathrm{iso}}
=========================

f_{\alpha,0}
\left(
{\mathcal O_k^{(0)},I_k,n_k}
\right),
]

[
\boxed{
\hat\alpha^{(2)}
================

\sum_k
M_k
r_{\alpha}(I_k,n_k)
\mathcal O_k^{(2)}.
}
]

最后从 spherical tensor 转回 Cartesian tensor。

---

## 10.3 IR/Raman

第一篇不要训练 Hessian。直接使用数据集中的 DFT normal modes (Q_m) 和 frequencies (\omega_m)。

IR：

[
I_m^{\mathrm{IR}}
\propto
\left|
\frac{\partial \mu}{\partial Q_m}
\right|^2.
]

Raman：

[
I_m^{\mathrm{Raman}}
\propto
F
\left(
\frac{\partial \alpha}{\partial Q_m},
\omega_m
\right).
]

工程上用 VJP，不要 full Jacobian：

```python
# dmu_dR: [3, N, 3]
# Q:      [M, N, 3]
dmu_dQ = torch.einsum("cna,mna->cm", dmu_dR, Q)

# dalpha_dR: [6, N, 3]
# Q:         [M, N, 3]
dalpha_dQ = torch.einsum("cna,mna->cm", dalpha_dR, Q)
```

这能证明 MTO bottleneck 支撑响应导数，而不需要把势能面、Hessian、MD 拉进主线。

---

## 10.4 UV-Vis

第一版直接预测 broadened spectrum：

[
\boxed{
\hat I_{\mathrm{UV}}(\omega)
============================

f_{\mathrm{UV}}
\left(
{\mathcal O_k,I_k,n_k},\omega
\right).
}
]

或者预测离散 grid：

[
\hat{\mathbf I}_{\mathrm{UV}}
=============================

\mathrm{MLP}_{\mathrm{UV}}
\left(
\mathrm{MTOPool}({\mathcal O_k,I_k,n_k})
\right).
]

不要写成“模型显式学习 occupied-to-virtual transition”。可以说：

[
\boxed{
\textbf{
UV supervision probes whether response-like MTO modes capture electronic excitation-sensitive patterns.
}
}
]

---

# 11. 损失函数

总体损失：

[
\mathcal L
==========

\lambda_\mu\mathcal L_\mu
+
\lambda_\alpha\mathcal L_\alpha
+
\lambda_{\mathrm{IR}}\mathcal L_{\mathrm{IR}}
+
\lambda_{\mathrm{Raman}}\mathcal L_{\mathrm{Raman}}
+
\lambda_{\mathrm{UV}}\mathcal L_{\mathrm{UV}}
+
\lambda_{\mathrm{div}}\mathcal L_{\mathrm{div}}
+
\lambda_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}.
]

张量性质：

[
\mathcal L_\mu
==============

|\hat\mu-\mu|_2^2,
]

[
\mathcal L_\alpha
=================

|\hat\alpha-\alpha|_F^2.
]

谱图：

[
\mathcal L_{\mathrm{spec}}
==========================

\mathrm{MSE}(\hat I,I)
+
\lambda_{\cos}
\left(
1-\cos(\hat I,I)
\right).
]

所有任务 label 做 z-score 标准化。多任务权重可以先用 uncertainty weighting；如果出现任务梯度冲突，再启用 GradNorm 或 PCGrad。

---

# 12. 正则项

## 12.1 Diversity loss

防止 slot collapse：

[
G_{kk'}
=======

\sum_l
\frac{
\langle
\mathcal O_k^{(l)},
\mathcal O_{k'}^{(l)}
\rangle
}{
|\mathcal O_k^{(l)}|
|\mathcal O_{k'}^{(l)}|+\epsilon
}.
]

[
\mathcal L_{\mathrm{div}}
=========================

\sum_{k\ne k'}G_{kk'}^2.
]

权重很小：

[
\lambda_{\mathrm{div}}=10^{-4}\sim10^{-3}.
]

不要强行把 MTO 正交死，因为物理上近简并响应子空间本来可能存在。

## 12.2 Routing entropy

早期轻微鼓励 routing entropy，避免所有 slot collapse 到同一原子区域。后期减弱，让 slot 自然局域或离域。

## 12.3 Activity smoothness

不强制 (n_k) 整数化，只通过 (\theta) 退火控制活性分配从软到较尖锐。

---

# 13. 训练流程

## Stage A：张量响应预训练

[
\mathcal T_1={\mu,\alpha}.
]

目标：

* 验证 (l=1) 和 (l=2) MTO 通道可训练；
* 检查 MTO maps 是否稳定；
* 做最小 ablation：no sign / no gate / fixed (K)。

## Stage B：加入振动响应

[
\mathcal T_2={\mu,\alpha,IR,Raman}.
]

目标：

* 验证加入 IR/Raman 后，(\mu,\alpha) 的 MTO maps 是否保持稳定；
* 用 DFT normal modes 计算响应导数；
* 不训练 Hessian。

## Stage C：加入电子谱图

[
\mathcal T_3={\mu,\alpha,IR,Raman,UV}.
]

目标：

* 观察 UV 是否诱导出新的 response-like MTO；
* 检查原有 (\mu,\alpha) 模式是否被破坏；
* 做 fragment-level UV attribution。

## Stage D：可选辅助任务

可选加入小权重 (E,\mathbf F)，仅作为几何平滑正则。第一篇主文不强调。

---

# 14. 稳定性与解释性验证

这是文章的真正核心，不是附加分析。

## 14.1 Task-stability

训练三组模型：

[
\mathcal T_1={\mu,\alpha},
]

[
\mathcal T_2={\mu,\alpha,IR,Raman},
]

[
\mathcal T_3={\mu,\alpha,IR,Raman,UV}.
]

对同一分子，比较 property-specific atom map 是否稳定。

定义 intrinsic contribution：

[
w_{ki}^{\mathrm{int}}
=====================

\sum_l
|c_{ki}^{(l)}h_i^{(l)}|^2.
]

mask 第 (k) 个 MTO：

[
\mathcal O_k\leftarrow 0.
]

计算属性变化：

[
A_{p,k}
=======

|\hat y_p-\hat y_p^{(-k)}|.
]

property-specific atom map：

[
\boxed{
w_{p,i}
=======

\sum_k
A_{p,k}
\frac{
w_{ki}^{\mathrm{int}}
}{
\sum_iw_{ki}^{\mathrm{int}}+\epsilon
}.
}
]

比较：

[
\mathrm{corr}
\left(
w_{\mu}^{\mathcal T_1},
w_{\mu}^{\mathcal T_2}
\right),
]

[
\mathrm{corr}
\left(
w_{\alpha}^{\mathcal T_1},
w_{\alpha}^{\mathcal T_3}
\right).
]

如果高，说明 MTO 不是某个任务头的偶然 artifact，而是跨任务稳定响应模式。

---

## 14.2 Seed subspace-stability

MTO 存在 gauge freedom：

[
\mathcal O_k\rightarrow-\mathcal O_k,
\quad
r_k\rightarrow-r_k,
]

预测不变。因此不要直接比较单个 slot。

选 top-(r) 个 response-important MTO，形成矩阵 (O_a,O_b)，QR 正交化：

[
Q_a=\mathrm{QR}(O_a),
\quad
Q_b=\mathrm{QR}(O_b).
]

投影矩阵：

[
P_a=Q_aQ_a^\top,
\quad
P_b=Q_bQ_b^\top.
]

定义：

[
\boxed{
S_{\mathrm{sub}}
================

\frac{1}{r}
\mathrm{Tr}(P_aP_b).
}
]

这个指标对 sign flip 和 slot permutation 都不敏感。

---

## 14.3 Slot intervention

对特定 MTO：

[
\mathcal O_k\leftarrow0.
]

观察：

[
\Delta\mu,\Delta\alpha,\Delta IR,\Delta Raman,\Delta UV.
]

如果某些 MTO 主要影响 (\mu/IR)，某些主要影响 (\alpha/Raman)，某些主要影响 UV，就能说明 MTO 具有功能分化。

---

## 14.4 化学验证

做三类轻量但有说服力的验证。

### Donor-acceptor polarity

体系：

[
\text{benzene}
\rightarrow
\text{aniline}
\rightarrow
\text{nitrobenzene}
\rightarrow
\text{p-nitroaniline}.
]

比较：

[
w_{\mu,i}
]

与：

[
\Delta q_i^{\mathrm{NPA}},
\quad
\Delta q_i^{\mathrm{ESP}},
\quad
\Delta q_i^{\mathrm{Mulliken}}.
]

报告 Spearman 相关和 (R^2)。

### Conjugation / polarizability

体系：

[
\text{alkane}
\rightarrow
\text{alkene}
\rightarrow
\text{polyene}
\rightarrow
\text{D-}\pi\text{-A}.
]

比较：

[
w_{\alpha,i}
]

与共轭长度、bond order alternation、NBO delocalization descriptor。

### UV charge-transfer direction

选 20–50 个典型 D-(\pi)-A 分子，做 fragment-level score：

[
S_{\mathrm{CT}}
===============

## \sum_{i\in A}w_{\mathrm{UV},i}

\sum_{i\in D}w_{\mathrm{UV},i}.
]

比较其与 TDDFT/NTO 给出的 donor-to-acceptor charge transfer trend 是否一致。不要第一篇大规模硬对齐 transition density。

---

# 15. Baselines 与消融

主 baseline：

1. backbone + direct tensor readout；
2. sum pooling；
3. attention pooling；
4. multi-token pooling；
5. fixed-(K) latent token；
6. MTO without sign；
7. MTO without gated nonlinearity；
8. MTO with geometric center；
9. MTO without activity gate；
10. full MTO-Net。

消融不要只报 MAE，还要报：

[
\mathrm{MAE/RMSE},
\quad
S_{\mathrm{sub}},
\quad
\mathrm{TaskSim}*\mu,
\quad
\mathrm{TaskSim}*\alpha,
\quad
\mathrm{SlotIntervention\ Selectivity}.
]

关键结论应该是：

[
\boxed{
\textbf{
Full MTO-Net 在预测性能保持 competitive 的同时，在 task-stability、seed subspace-stability 和 chemical validation 上显著优于普通 pooling/token baseline。
}
}
]

这比单纯追逐 MAE 更符合文章的创新定位。

---

# 16. 数据集建议

第一篇最建议使用 QM9S-like 数据。

理由：

* 同一批分子有 (\mu,\alpha,IR,Raman,UV) 等多任务标签；
* 分子规模适中，适合 variable-(K)；
* 已有 DetaNet 等强 baseline；
* 不需要自己从零构造复杂谱图数据。

如果能额外构造小型 D-(\pi)-A / donor-acceptor 验证集，则解释性部分会更强。

---

# 17. 论文图表设计

## Figure 1：概念图

左边：普通等变 GNN

[
{h_i^{(l)}}
\rightarrow
\mathrm{pooling}
\rightarrow
y.
]

右边：MTO-Net

[
{h_i^{(l)}}
\rightarrow
{\mathcal O_k^{(l)},n_k}
\rightarrow
\mu,\alpha,IR,Raman,UV.
]

强调：MTO 不是真实 MO，而是 orbital-like latent response mode。

## Figure 2：模型结构图

展示：

* center-free routing；
* signed assembly；
* gated relaxation；
* variable-(K) MTO bank；
* activity gate。

## Figure 3：预测性能

展示 (\mu,\alpha,IR,Raman,UV) 的性能。

## Figure 4：Task-stability

同一分子在 (\mathcal T_1,\mathcal T_2,\mathcal T_3) 下的 response maps。

## Figure 5：Seed subspace-stability

不同 seed 的 (S_{\mathrm{sub}}) 分布，与 attention/multi-token baseline 比较。

## Figure 6：Chemical validation

donor-acceptor、共轭链、D-(\pi)-A 的 MTO attribution 与化学指标对照。

## Figure 7：Ablation

sign、gate、variable-(K)、activity gate、center-free 的消融。

---

# 18. 文章贡献点

最终贡献可以写成四点：

## Contribution 1：Molecule-level response modes

提出 MTO 作为等变隐空间中的分子级轨道式响应模式，而不是普通 pooling token。

## Contribution 2：Center-free signed nonlinear assembly

提出无几何中心、带符号、归一化、门控非线性的 MTO assembly，用于从原子张量场形成分子级响应模式。

## Contribution 3：Valence-adaptive active capacity

提出：

[
K(N_{\mathrm{val}})
===================

(1+\gamma)
\left\lceil
\frac{N_{\mathrm{val}}}{2}
\right\rceil
]

使 MTO 容量随价电子复杂度变化。

## Contribution 4：Gauge-invariant stability validation

提出 task-stability、seed subspace-stability、slot intervention 和 chemical validation，证明 MTO 是稳定响应模式，而不是普通 attention artifact。

---

# 19. 最终可执行 MVP

最稳的第一篇版本是：

[
\boxed{
\textbf{
MTO-Net v6-lite-response =
Equivariant backbone
+
center-free signed nonlinear MTO assembly
+
valence-adaptive MTO bank
+
charge-conserving activity gate
+
\mu,\alpha,IR,Raman,UV response learning
+
task/seed/chemical stability validation.
}
}
]

不做主线：

[
\boxed{
E,\mathbf F,Hessian,MD,TS,\psi,H_{\mathrm{DFT}},\rho(\mathbf r).
}
]

这些只在 Discussion 中写：

[
\boxed{
\textbf{
The proposed MTO bottleneck can in principle be extended to energies, forces, electron densities, and other quantum-chemical targets, but these extensions are beyond the scope of the present response-centered study.
}
}
]

---

# 20. 最终判断

这版方案已经足够成熟，而且主线非常清楚：

[
\boxed{
\textbf{
MTO-Net 不是神经量子化学计算器，而是一个在等变分子模型中形成稳定分子级响应模式的方法。
}
}
]

最重要的故事是：

[
\boxed{
\textbf{
原子级等变张量场并不只能被黑箱 pooling 掉；它们可以被重组为一组稳定、可复用、可解释的分子级张量响应模式。
}
}
]

这就是你这篇文章的灵魂。它保留了分子轨道理论的美感，但没有过度宣称；保留了深度学习的表达力，但没有变成黑箱；保留了 AI4S 的物理品味，但没有陷入完整量子化学重构的泥潭。

[1]: https://arxiv.org/abs/2207.09453?utm_source=chatgpt.com "e3nn: Euclidean Neural Networks"
[2]: https://pubmed.ncbi.nlm.nih.gov/38177591/?utm_source=chatgpt.com "A deep learning model for predicting selected organic ..."
