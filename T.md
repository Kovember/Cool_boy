# Transformer 基础

## 1 从 RNN 到 Transformer

### 1.1 MLP—固定窗口的映射

对于固定窗口大小为 $T$ 的输入序列 \(X = [x_1, x_2, ..., x_T] \in \mathbb{R}^{T \times d}\)，MLP 将每个位置独立处理：

输入窗口长度固定，需展平为 $\text{vec}(X) \in \mathbb{R}^{Td}$：

$$
y = \sigma(W \cdot \text{vec}(X) + b)
$$

**关键缺陷：**

- 只能处理**固定长度**的输入窗口，无法适应变长序列。
- 各位置**独立处理**，完全丢失词序信息。
- 窗口外的上下文被截断，**无法建模长距离依赖**。

### 1.2 RNN—状态递归

$h_t$ 表示当前时间步的隐状态，$x_t$ 表示当前输入：

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

$$
y_t = W_y h_t + b_y
$$

**优点：**

* 引入**递归状态**，天然支持**变长序列**。
* 隐状态沿时间步传递，保持**顺序敏感性**。
* 理论上可依赖**任意远的过去信息**.

**缺点：**

- **串行计算**：每个时间步依赖前一时刻的隐状态，无法并行训练，处理长序列慢。
- **长距离遗忘**：全局信息传播依赖于递归过程，远距离信息会衰减，难以捕获全局依赖。

### 1.3 Transformer—全局注意力，完全并行

Transformer的端到端模型：

* 嵌入层（Embedding）

  - **Token 嵌入**：将输入 token 映射为稠密向量  
    $$
    \mathbf{X}_{\text{token}} = \text{Lookup}(E, \text{tokens}), \quad E \in \mathbb{R}^{V \times d_{\text{model}}}
    $$

  - **位置编码**：注入序列顺序信息
    $$
    \mathbf{X} = \mathbf{X}_{\text{token}} + \mathbf{P}, \quad \mathbf{P} \in \mathbb{R}^{T \times d_{\text{model}}}
    $$

* 注意力层（Attention）

  - **缩放点积注意力**：  
    $$
    \text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

  - **多头注意力**（自注意力情形，$Q=K=V=\mathbf{X}$）：  
    $$
    \text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O
    $$
    其中 $\text{head}_i = \text{Attn}(\mathbf{X}W_i^Q,\ \mathbf{X}W_i^K,\ \mathbf{X}W_i^V)$。

* 前馈层（FFN）
  $$
  \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
  $$
  或写作 $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$

## 2 Transformer 的架构组成

我们把 Transformer 从架构上分为三层

### 2.1 输入表征层

**目标**：把离散的 token 变成模型能处理的连续向量，并注入位置信息。

#### ① Token Embedding

- **原理**：每个 token 对应一个可学习的向量，形状为 `[vocab_size, d_model]`。
- **输入**：`input_ids`，形状 `[batch_size, seq_len]`（比如 `[2, 512]`）。
- **输出**：`[2, 512, d_model]`（`d_model` 通常取 512、768、1024 等）。

#### ② Positional Encoding

**Why**：Self-Attention 本身是**置换等变的**——如果把输入序列打乱，Attention 输出也会对应打乱，但它**没有内置的顺序概念**。所以我们需要显式注入位置信息。

**三种主流方式**：

1. **Sinusoidal（正弦/余弦）编码**（原始 Transformer）
   $$
   PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad 
   PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
   $$

   - 优点：可以外推到比训练时更长的序列；无需额外参数。
   - **Why 用 sin/cos？** 使得对于任意偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换，便于模型学习相对位置。

2. **可学习位置编码**（BERT、GPT 等常用）

   - 直接初始化一个 `[max_seq_len, d_model]` 的参数矩阵，随网络一起训练。
   - 优点：更灵活，可以适应任务；缺点：最大长度固定，不能外推。

**输出**：两种编码都与 token embedding **相加**，形状不变 `[batch, seq, d_model]`。

3. **RoPE（旋转位置编码）**

   - 不是将位置向量加到词向量上，而是通过旋转矩阵对 **Query 和 Key 向量** 施加与位置相关的变换。对于第 $i$ 维子空间，旋转角度为 $\theta_i = \text{base}^{-2i/d}$，位置 $m$ 的变换为：

   $$
   f_q(q, m) = q \cdot R_{\theta_i}(m), \quad f_k(k, n) = k \cdot R_{\theta_i}(n)
   $$

   其中旋转矩阵为：
   $$
   R_{\theta_i}(m) = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
   $$

   实际计算中不显式构造矩阵，而是利用复数乘法或按维度公式：
   $$
   \begin{aligned}
   q'_0 &= q_0 \cos m\theta_i - q_1 \sin m\theta_i \\
   q'_1 &= q_1 \cos m\theta_i + q_0 \sin m\theta_i
   \end{aligned}
   $$

   **代码片段（复数实现）**：

   ```python
   def precompute_freqs_cis(dim, seq_len, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2] / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
   
   def apply_rotary_emb(x, freqs_cis):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs_cis[:, None, :]
        return torch.view_as_real(x_rotated).flatten(3).type_as(x)
   ```

   - **意义**
     - **相对位置建模**：内积结果 $f_q(q,m) \cdot f_k(k,n)$ 只依赖于 $m-n$。
     - **长序列外推友好**：可通过 Position Interpolation 等方法扩展上下文窗口。
     - **无额外参数**：旋转是确定的。

**面试问题与回答要点**

1. **RoPE 与绝对位置编码（如 Sinusoidal）本质区别？**  
   → 绝对位置编码是在输入层加位置向量，RoPE 直接修改 Q/K，使注意力分数隐含相对位置。
2. **如何用 RoPE 实现 4k → 32k 上下文外推？**  
   → 位置插值（PI）：将位置索引从 $m$ 缩小为 $m \times (L_{\text{train}} / L_{\text{test}})$；NTK-aware scaling：调整 base 值。
3. **手写 RoPE 旋转公式（对一对维度）** → 见上方公式。


### 2.2 注意力层

这是 Transformer 的“核心引擎”，也是面试必问的地方。

#### ① 缩放点积注意力（Scaled Dot-Product Attention）

**公式**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- **Q, K, V** 由同一个输入 $X$ 通过三个不同的线性变换得到。
- **Shape**：假设输入 $X$ 为 `[B, S, D]`，线性变换后依然 `[B, S, D]`。  
  为了多头，后面会切分，但这里先看单头。

**计算过程**：

1. $ QK^T $：`[B, S, D]` × `[B, D, S]` → `[B, S, S]`，表示每个位置对其他所有位置的“相似度”。
2. 除以 $ \sqrt{d_k} $（其中 $d_k = D / H$，H 为头数）。
   - **Why？** 假设 $q, k$ 的每个元素均值为 0，方差为 1，那么 $q \cdot k$ 的方差就是 $d_k$。当 $d_k$ 较大时，点积结果会非常大，导致 softmax 进入饱和区（梯度极小）。除以 $ \sqrt{d_k} $ 使方差回到 1，保持梯度稳定。
3. softmax 按行（最后一个维度）归一化，得到注意力权重。
4. 乘以 $ V $：`[B, S, S]` × `[B, S, D]` → `[B, S, D]`，加权聚合信息。

**代码（单头）**：

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [batch, seq_len, d_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, S, S]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # [B, S, d_k]
    return output, attn_weights
```

#### ② 多头注意力（Multi-Head Attention, MHA）

**Why 多头？**  
单头注意力只能学到一种“关系”。多头让模型在不同的子空间里分别计算注意力，从而捕捉多种类型的关系（比如语法、语义、共现等）。类似于 CNN 中用多个卷积核。

**实现步骤**：

1. 线性变换得到 $Q, K, V$，形状 `[B, S, D]`。
2. 将最后一维切分成 $H$ 个头：`[B, S, H, D_k]`（$D = H \times D_k$）。
3. 交换维度，变成 `[B, H, S, D_k]`，方便并行计算。
4. 对每个头独立做缩放点积注意力，得到 `[B, H, S, D_k]`。
5. 交换回 `[B, S, H, D_k]`，合并成 `[B, S, D]`。
6. 最后一个线性投影，输出 `[B, S, D]`。

**伪代码**：

```python
def multi_head_attention(x, num_heads, d_model):
    batch, seq, _ = x.shape
    d_k = d_model // num_heads

    # 线性变换
    q = nn.Linear(d_model, d_model)(x)  # [B, S, D]
    k = nn.Linear(d_model, d_model)(x)
    v = nn.Linear(d_model, d_model)(x)

    # 切头
    q = q.view(batch, seq, num_heads, d_k).transpose(1, 2)  # [B, H, S, D_k]
    k = k.view(batch, seq, num_heads, d_k).transpose(1, 2)
    v = v.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # 缩放点积注意力
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, S, S]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [B, H, S, D_k]

    # 合并头
    out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)  # [B, S, D]
    out = nn.Linear(d_model, d_model)(out)  # 最终投影
    return out
```

#### ③ 掩码（Mask）

**两种掩码**：

- **Padding Mask**：对输入中的填充位置（如 `[PAD]`）进行屏蔽，防止模型关注它们。  
  方法：在 softmax 之前，将对应位置设为 `-inf`（或一个极小的负数），使 softmax 后的权重接近 0。

- **Causal Mask（因果掩码）**：在 Decoder 中，保证位置 $i$ 只能看到位置 $j \leq i$ 的 token，防止“看到未来”。  
  方法：构造一个上三角矩阵（不含对角线），对每个 `(i, j)` 其中 $j > i$ 的位置设为 `-inf`。

**Shape**：通常 mask 是 `[B, 1, 1, S]` 或 `[1, 1, S, S]`，通过广播机制与 `[B, H, S, S]` 对齐。

### 2.3 结构层

#### ① 前馈网络（FFN）

**公式**：
$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

- $W_1$ 的形状：`[d_model, d_ff]`，通常 $d_{ff} = 4 \times d_{model}$。
- $W_2$ 的形状：`[d_ff, d_model]`。
- **Why 需要 FFN？**  
  Attention 负责在**不同 token 之间**交换信息（线性加权），FFN 负责在**每个 token 内部**做非线性变换，提升模型表达能力。两者交替，形成了 Transformer 的“通信-计算”结构。

**现代变体**（如 LLaMA 使用 SwiGLU）：
$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)
$$
效果更好，但参数略多。

#### ② 残差连接 + 层归一化（Residual + LayerNorm）

**结构**：
$$
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$
（原始 Transformer 为 Post-Norm，现代更常用 Pre-Norm）

- **Why 残差？**  
  解决深层网络梯度消失问题，保证梯度能直接从损失流回浅层。

- **Why LayerNorm，而不是 BatchNorm？**  
  - LN 对每个样本的**特征维度**做归一化，不依赖于 batch 大小，对变长序列友好。
  - BN 依赖 batch 统计量，且对变长序列（不同样本长度不一）处理复杂。
  - 在 Transformer 中，LN 能使训练更稳定。

- **Pre-Norm vs Post-Norm**：
  - **Post-Norm**（原始）：$ \text{LN}(x + \text{Sublayer}(x)) $。收敛慢，需要 warmup，但理论表达能力强。
  - **Pre-Norm**（主流）：$ x + \text{Sublayer}(\text{LN}(x)) $。梯度流更顺畅，无需 warmup，训练稳定，但可能略低于 Post-Norm 的理论上限。几乎所有大模型（GPT、LLaMA）都用 Pre-Norm。

#### ③ 最后的 Softmax

在解码器输出后，通过一个线性层（`[d_model, vocab_size]`）将隐状态映射为 logits，再 softmax 得到概率分布，用于预测下一个 token。

## 3 Transformer三种架构

| 架构                | 代表模型                 | 核心结构                      | 注意力掩码                                       | 训练任务                                | 典型应用                           |
| ------------------- | ------------------------ | ----------------------------- | ------------------------------------------------ | --------------------------------------- | ---------------------------------- |
| **Encoder-Only**    | BERT, RoBERTa            | 堆叠 Encoder 层               | 双向（无掩码）                                   | MLM（掩码语言模型）+ NSP（可选）        | 理解类任务：分类、实体识别、相似度 |
| **Decoder-Only**    | GPT 系列, LLaMA, Mistral | 堆叠 Decoder 层（带因果掩码） | 因果（只能看左边）                               | 自回归语言建模（Next Token Prediction） | 生成类任务：对话、写作、代码、推理 |
| **Encoder-Decoder** | T5, BART, M2M100         | Encoder + Decoder             | Encoder：双向<br>Decoder：因果 + Cross-Attention | 去噪自编码（Span Corruption）或翻译     | 序列到序列：翻译、摘要、结构化转换 |

### 3.1 深入对比与 Why

#### ① Encoder-Only（以 BERT 为例）

- **结构**：多层双向 Attention。
- **为什么双向？**  
  理解任务（如情感分类）需要同时看到上下文，才能准确判断语义。比如 “not good” 中的否定需要结合后面的词才能理解。
- **训练任务**：**MLM**（Masked Language Model）——随机 mask 掉 15% 的 token，让模型预测。迫使模型学会利用双向上下文。
- **局限**：无法做生成（因为生成需要从左到右自回归）。

#### ② Decoder-Only（以 GPT 为例）

- **结构**：多层因果 Attention（上三角 mask）。
- **为什么因果？**  
  生成文本时，只能根据已经生成的 token 预测下一个，不能“偷看”未来，符合自回归生成的逻辑。
- **训练任务**：自回归语言建模，即给定前文预测下一个 token。
- **为什么现在主流是大 Decoder-Only？**  
  1. **通用性**：生成任务天然涵盖理解任务（通过 prompt 可以要求模型做分类、问答等）。  
  2. **训练效率**：相比 Encoder-Decoder，参数利用率更高（没有额外 Encoder）。  
  3. **涌现能力**：大规模的 Decoder-Only 模型（如 GPT-3、LLaMA）在上下文学习（In-context Learning）、思维链（CoT）等方面表现出色。

#### ③ Encoder-Decoder（以 T5 为例）

- **结构**：Encoder 对输入做双向编码，Decoder 通过 Cross-Attention 对齐 Encoder 输出，自回归生成。
- **为什么需要这种结构？**  
  当输入和输出**长度差异大**或**结构不同**时（比如翻译：中文->英文，摘要：长文->短摘要），Encoder 专门负责理解源文本，Decoder 专门负责生成目标文本，中间用 Cross-Attention 进行对齐，比单一的 Decoder 更自然。
- **训练任务**：T5 采用 **Span Corruption** —— 将输入中若干连续 token 替换为一个哨兵 token，Decoder 需要恢复这些被 mask 的 span。这种训练方式结合了双向理解和自回归生成。

### 3.2 关于 Decoder-Only 成为主流的深入思考

面试官可能追问：“既然 Encoder-Decoder 对 Seq2Seq 更自然，为什么大模型几乎都是 Decoder-Only？”

- **回答思路**：
  1. **规模效应**：在相同参数规模下，Decoder-Only 的 FLOPs 更低（因为不需要 Cross-Attention 的双向计算？不对，其实 Decoder-Only 也有双向？）—— 更准确地说，Decoder-Only 的**参数利用效率高**，所有参数都用于生成任务，而 Encoder-Decoder 中 Encoder 参数只用于理解源文本，生成时全靠 Decoder，参数利用率低。
  2. **指令微调与通用性**：Decoder-Only 通过 prompt 可以完成各种任务（如 “请分类：...” ），一个模型通吃。而 Encoder-Decoder 往往需要为不同任务调整结构。
  3. **涌现能力**：大规模 Decoder-Only 模型在上下文学习上表现出更强的涌现能力，可能与因果掩码带来的“顺序推理”特性有关。

**补充**：目前也有一些混合架构（如 Encoder-Decoder 的大型模型，例如 T5-11B），但主流开源模型（LLaMA, Mistral, Qwen）和闭源模型（GPT-4）都选择 Decoder-Only。

## 4 MoE

### 4.1 MoE 的架构细节

#### Token Choice

这是最常见的方式，例如 Switch Transformer、Mixtral、DeepSeek-MoE 都采用 Token Choice。

**原理**：每个 token 独立地选择最合适的 Top-K 个专家。Router 对每个 token 输出一个对所有专家的概率分布，然后每个 token 挑选概率最高的 K 个专家，将自身的表示发送给这些专家，专家的输出按路由概率加权求和。

**公式**：
对于 token $x$，Router 输出 logits $h(x) = W_g x$（$W_g \in \mathbb{R}^{E \times d}$），然后 softmax 得到概率 $p = \text{softmax}(h(x))$。选择 Top-K 索引集合 $\mathcal{T}$，最终输出：
$$
y = \sum_{i \in \mathcal{T}} p_i \cdot \text{Expert}_i(x)
$$

**特点**：

- 每个 token 的计算量固定（K 个专家）。
- 不同 token 可能选择不同的专家，专家负载可能不均衡（有的专家被很多 token 选择，有的很少）。
- 需要辅助损失（负载均衡损失）来鼓励均匀分配。

#### Expert Choice

这是一种较少见但有趣的方式，由例如 "Mixture-of-Experts with Expert Choice" 论文提出。

**原理**：每个专家选择它要处理的 token，而不是 token 选择专家。具体来说，对所有 token 的路由分数，每个专家挑选分数最高的 Top-K 个 token（或者按容量选择）。专家输出后，再根据路由分数加权聚合回每个 token。

**公式**（简化）：
设 batch 中有 $ T $ 个 token，每个 token 有路由分数 $s_{t,i}$ 表示 token $ t $ 与专家 $i$ 的匹配度。专家 $i$ 选择分数最高的 $C_i$ 个 token（$C_i$ 可以是容量，如 $C_i = \text{capacity\_factor} \times T/E$）。被选中的 token 集合记为 $\mathcal{T}_i$，专家 $i$ 输出 $y_{t,i} = \text{Expert}_i(x_t)$。最终 token $ t $ 的输出为：
$$
y_t = \sum_{i: t \in \mathcal{T}_i} \frac{s_{t,i}}{\sum_{t' \in \mathcal{T}_i} s_{t',i}} \cdot y_{t,i}
$$
即用该 token 在专家 i 的选中集合中的归一化分数作为权重。

**特点**：

- 负载天然均衡（每个专家处理固定数量的 token），强制均匀分布。
- 但每个专家处理的 token 数量固定，可能造成信息损失（如果某专家对所有 token 分数都很低，仍需强制选择一些 token）。
- 实现复杂度高，推理时难以动态适配。

**面试考察点**：

- Token Choice 为什么需要辅助损失？Expert Choice 如何避免负载不均衡？
- 在实际大模型中，哪种更常用？为什么？（Token Choice 更灵活，实现简单，配合辅助损失效果好。）


### 4.2 路由选择

Router的本质是一个线性层 $W_g \in \mathbb{R}^{E \times d}$，输入 token 的隐向量 $x \in \mathbb{R}^d$，输出 logits $z = W_g x$（维度 $E$，专家数量）。然后经过 softmax 得到概率分布。

**关点**：

- **噪声注入**（训练时）：Switch Transformer 等模型在路由 logits 中添加可调节的高斯噪声，鼓励探索，防止 Router 过早收敛到次优分配。公式：
  $$
  z_i = \frac{x \cdot W_g^{(i)} + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}}^{(i)})}{\text{temperature}}
  $$
  其中 $\epsilon \sim \mathcal{N}(0,1)$，$W_{\text{noise}}$ 是可学习的噪声参数。训练初期噪声大，后期逐渐降低。

- **温度系数**：可以引入温度 $ T $ 来平滑或锐化分布。$T<1$ 使分布更尖锐（偏向最大专家），$T>1$ 更平滑。通常 $T=1$。

**面试考察点**：

- 为什么需要在路由 logits 中加噪声？（防止 Router 早期崩溃到单一专家，促进探索）
- 路由 logits 的梯度如何传播？（通过 softmax 和 Top-K 选择，但 Top-K 操作本身不可微，通常采用 straight-through estimator 或使用 soft Top-K）

在 Token Choice 中，每个 token 不是选择所有专家，而是只选概率最高的 K 个专家（K 通常为 1 或 2）。

**为什么 K=1 或 2？**

- **K=1**：Switch Transformer 使用。每个 token 只由一个专家处理，计算量最小，但可能损失表达能力（单一专家可能无法处理复杂模式）。
- **K=2**：Mixtral、DeepSeek-MoE 使用。平衡了计算量和表达能力，且可以缓解负载均衡（因为 token 可以同时选两个专家，更容易均匀分布）。

**Top-K 的软硬选择**：

- **硬 Top-K**：直接选择概率最高的 K 个，其他专家输出为 0。这种方式不可微，但通过梯度估计（如将选择的专家的梯度回传，未选的不回传）仍然可以训练。
- **软 Top-K**：使用连续的近似，如对概率分布做 top-k 平滑（将非 Top-K 的概率置 0，再归一化），仍然可微但计算稍复杂。

**容量因子（Capacity Factor）**：
为了控制每个专家处理的 token 数量，常引入容量因子。每个专家的容量 = $\text{capacity\_factor} \times \frac{\text{total\_tokens}}{E}$。如果某个专家被分配的 token 超过容量，超出的 token 会被丢弃（或通过残差连接绕过专家）。容量因子通常设为 1.0~1.5，避免 token 被丢弃过多。

**面试考察点**：

- 为什么 Top-2 比 Top-1 更好？（降低负载不均衡，提高模型容量）
- 容量因子过小或过大会有什么影响？（过小导致 token 被丢弃，信息损失；过大导致负载不均衡和计算浪费）
- 如何解决 Token 被丢弃的问题？（使用更大的容量因子，或使用 Expert Choice）

### 4.3 MoE 的路由坍塌

**路由坍塌（Routing Collapse）** 是指 Router 将所有 token 都分配给少数几个专家，导致其他专家几乎不被训练，模型退化为一个小型 Dense 模型，失去了 MoE 的优势。这是 MoE 训练中最常见的问题。

导致坍塌的原因：

- 早期训练时，Router 随机初始化，某个专家偶然获得稍高的分数，该专家得到更多 token → 该专家梯度更新更多 → 它变得更擅长处理更多 token → 正反馈循环，其他专家逐渐被“饿死”。
- 缺乏足够的探索，Router 过早陷入局部最优。

解决方案：

#### 辅助损失（Auxiliary Loss）

这是最常用的方法，在训练目标中加入一个辅助损失，惩罚负载不均衡。常见的两种形式：

**a) Importance-based Loss（Switch Transformer）**
$$
\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$
其中：

- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{\text{token } t \text{ 选择专家 } i\}$，即专家 $i$ 被选中的 token 比例。
- $P_i = \frac{1}{T} \sum_{t=1}^{T} p_{t,i}$，即所有 token 对专家 $i$ 的平均路由概率。
- $\alpha$ 是系数，通常取 0.01。

**解释**：当专家 $i$ 被选中的频率 $f_i$ 高，同时 Router 给它的平均概率 $P_i$ 也高时，损失大。这鼓励 Router 使 $f_i$ 和 $P_i$ 都接近 $1/E$，即均匀分布。

**b) Load-based Loss（GShard）**
直接基于每个专家实际处理的 token 数量 $l_i$ 计算方差或与均值的差异：
$$
\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} \left( \frac{l_i}{T} - \frac{1}{E} \right)^2
$$
更直接地强制每个专家处理的 token 数量相等。

**面试考察点**：

- 辅助损失如何与主损失（如语言建模损失）平衡？系数 $\alpha$ 如何选择？（通常很小，如 0.01，否则会干扰主任务）
- 辅助损失是否会影响模型性能？（适当使用可提升性能，因为负载均衡本身也有利于充分利用专家容量）

#### 熵正则化（Entropy Regularization）

**原理**：鼓励 Router 的输出概率分布更“均匀”（即高熵），避免分布过于集中在少数专家上。

**公式**：
$$
\mathcal{L}_{\text{entropy}} = -\alpha \cdot \frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{E} p_{t,i} \log p_{t,i}
$$
最大化熵（最小化负熵）使分布平坦，从而每个 token 不会过分依赖单一专家，间接促进专家利用的多样性。

**与辅助损失的区别**：

- 熵正则化作用于每个 token 的概率分布，鼓励 token 级别的均匀性。
- 辅助损失作用于全局统计，鼓励专家级别的负载均匀。
- 两者可以同时使用，相辅相成。

**面试考察点**：

- 熵正则化为什么能缓解路由坍塌？（防止 Router 输出尖锐分布，迫使每个 token 考虑多个专家）
- 熵正则化会不会导致每个 token 选择的专家过于分散，降低模型能力？（通过调节 $\alpha$ 可以平衡）

#### 硬约束（Hard Constraints）

不通过损失惩罚，而是直接对路由施加硬性限制，确保负载均衡。

**a) Expert Capacity 限制**
每个专家设置最大 token 容量（如 $ \text{capacity} = \lceil \frac{\text{total\_tokens}}{E} \times \text{capacity\_factor} \rceil $）。当某个专家被分配的 token 达到容量后，后续选择该专家的 token 会被强制重定向到其他专家（或直接丢弃/绕过）。

**实现**：在训练时，记录每个专家已处理的 token 数量，当超过容量时，将该 token 的该专家分数设为 $-\infty$，使其不再被选中。

**b) 强制均匀采样（Stochastic Routing）**
在训练初期，以一定概率随机分配专家（无视 Router 分数），强制每个专家都有机会训练。随着训练进行，逐渐退火到完全由 Router 决定。

**面试考察点**：

- 硬约束与软约束（辅助损失）的优缺点比较？硬约束保证绝对均衡，但可能丢弃 token 损失信息；软约束更平滑，但可能无法完全均衡。
- 容量因子如何设置？过小导致大量 token 被丢弃，过大则失去均衡作用。通常设为 1.0~1.5。=

#### 综合对比

| 方法         | 原理                   | 优点                       | 缺点                         |
| ------------ | ---------------------- | -------------------------- | ---------------------------- |
| **辅助损失** | 在损失中加惩罚项       | 平滑，不影响 token 分配    | 需要调整系数，可能干扰主任务 |
| **熵正则化** | 鼓励 token 级分布均匀  | 简单，防止 Router 过早尖锐 | 可能降低专家专业化程度       |
| **硬约束**   | 强制容量限制或随机分配 | 确保绝对均衡，直接有效     | 可能丢弃 token，实现复杂     |

在实际大模型（如 Mixtral、DeepSeek-MoE）中，通常**组合使用**多种方法：主要依赖辅助损失（负载均衡损失），配合熵正则化，同时设置合理的容量因子（硬约束），以保证训练稳定性和专家利用率。