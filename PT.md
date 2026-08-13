# 大模型基础、后训练与推理工程
## 一、Transformer 基础

### 1 从 RNN 到 Transformer

#### 1.1 MLP—固定窗口的映射

对于固定窗口大小为 $`T`$ 的输入序列 $`X = [x_1, x_2, ..., x_T] \in \mathbb{R}^{T \times d}`$，MLP 将每个位置独立处理：

输入窗口长度固定，需展平为 $`\text{vec}(X) \in \mathbb{R}^{Td}`$：

$$
y = \sigma(W \cdot \text{vec}(X) + b)
$$

**关键缺陷：**

- 只能处理**固定长度**的输入窗口，无法适应变长序列。
- 各位置**独立处理**，完全丢失词序信息。
- 窗口外的上下文被截断，**无法建模长距离依赖**。

#### 1.2 RNN—状态递归

$`h_t`$ 表示当前时间步的隐状态，$`x_t`$ 表示当前输入：

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

#### 1.3 Transformer—全局注意力，完全并行

Transformer的端到端模型：

- **Token 嵌入**：将输入 token 映射为稠密向量

$$
\mathbf{X}_{\text{token}} = \text{Lookup}(E, \text{tokens}), \quad E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

- **位置编码**：注入序列顺序信息

$$
\mathbf{X} = \mathbf{X}_{\text{token}} + \mathbf{P}, \quad \mathbf{P} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

- **缩放点积注意力**：

$$
\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- **多头注意力**：

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O,\quad \text{head}_i = \text{Attn}(\mathbf{X}W_i^Q,\ \mathbf{X}W_i^K,\ \mathbf{X}W_i^V)
$$

- **前馈层（FFN）**：$`\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2`$ 或写作 $`\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2`$

### 2 Transformer 的架构组成

我们把 Transformer 从架构上分为三层

#### 2.1 输入表征层

**目标**：把离散的 token 变成模型能处理的连续向量，并注入位置信息。

##### ① Token Embedding

- **原理**：每个 token 对应一个可学习的向量，形状为 `[vocab_size, d_model]`。
- **输入**：`input_ids`，形状 `[batch_size, seq_len]`（比如 `[2, 512]`）。
- **输出**：`[2, 512, d_model]`（`d_model` 通常取 512、768、1024 等）。

##### ② Positional Encoding

**Why**：Self-Attention 本身是**置换等变的**——如果把输入序列打乱，Attention 输出也会对应打乱，但它**没有内置的顺序概念**。所以我们需要显式注入位置信息。

**三种主流方式**：

1. **Sinusoidal（正弦/余弦）编码**（原始 Transformer）

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- 优点：可以外推到比训练时更长的序列；无需额外参数。
- **Why 用 sin/cos？** 使得对于任意偏移量 $`k`$，$`PE_{pos+k}`$ 可以表示为 $`PE_{pos}`$ 的线性变换，便于模型学习相对位置。

2. **可学习位置编码**（BERT、GPT 等常用）

- 直接初始化一个 `[max_seq_len, d_model]` 的参数矩阵，随网络一起训练。
- 优点：更灵活，可以适应任务；缺点：最大长度固定，不能外推。

**输出**：两种编码都与 token embedding **相加**，形状不变 `[batch, seq, d_model]`。

3. **RoPE（旋转位置编码）**

- 不是将位置向量加到词向量上，而是通过旋转矩阵对 **Query 和 Key 向量** 施加与位置相关的变换。对于第 $`i`$ 维子空间，旋转角度为 $`\theta_i = \text{base}^{-2i/d}`$，位置 $`m`$ 的变换为：

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
  - **相对位置建模**：内积结果 $`f_q(q,m) \cdot f_k(k,n)`$ 只依赖于 $`m-n`$。
  - **长序列外推友好**：可通过 Position Interpolation 等方法扩展上下文窗口。
  - **无额外参数**：旋转是确定的。

**常见问题**

1. **RoPE 与绝对位置编码（如 Sinusoidal）本质区别？**
   → 绝对位置编码是在输入层加位置向量，RoPE 直接修改 Q/K，使注意力分数隐含相对位置。
2. **如何用 RoPE 实现 4k → 32k 上下文外推？**
   → 位置插值（PI）：将位置索引从 $`m`$ 缩小为 $`m \times (L_{\text{train}} / L_{\text{test}})`$；NTK-aware scaling：调整 base 值。
3. **手写 RoPE 旋转公式（对一对维度）** → 见上方公式。


#### 2.2 注意力层

这是 Transformer 的“核心引擎”，也是理解 Transformer 的核心部分。

##### ① 缩放点积注意力（Scaled Dot-Product Attention）

**公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- **Q, K, V** 由同一个输入 $`X`$ 通过三个不同的线性变换得到。
- **Shape**：假设输入 $`X`$ 为 `[B, S, D]`，线性变换后依然 `[B, S, D]`。
  为了多头，后面会切分，但这里先看单头。

**计算过程**：

1. $`QK^T`$：`[B, S, D]` × `[B, D, S]` → `[B, S, S]`，表示每个位置对其他所有位置的“相似度”。
2. 除以 $`\sqrt{d_k}`$（其中 $`d_k = D / H`$，H 为头数）。
   - **Why？** 假设 $`q, k`$ 的每个元素均值为 0，方差为 1，那么 $`q \cdot k`$ 的方差就是 $`d_k`$。当 $`d_k`$ 较大时，点积结果会非常大，导致 softmax 进入饱和区（梯度极小）。除以 $`\sqrt{d_k}`$ 使方差回到 1，保持梯度稳定。
3. softmax 按行（最后一个维度）归一化，得到注意力权重。
4. 乘以 $`V`$：`[B, S, S]` × `[B, S, D]` → `[B, S, D]`，加权聚合信息。

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

##### ② 多头注意力（Multi-Head Attention, MHA）

**Why 多头？**
单头注意力只能学到一种“关系”。多头让模型在不同的子空间里分别计算注意力，从而捕捉多种类型的关系（比如语法、语义、共现等）。类似于 CNN 中用多个卷积核。

**实现步骤**：

1. 线性变换得到 $`Q, K, V`$，形状 `[B, S, D]`。
2. 将最后一维切分成 $`H`$ 个头：`[B, S, H, D_k]`（$`D = H \times D_k`$）。
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

##### ③ 掩码（Mask）

**两种掩码**：

- **Padding Mask**：对输入中的填充位置（如 `[PAD]`）进行屏蔽，防止模型关注它们。
  方法：在 softmax 之前，将对应位置设为 `-inf`（或一个极小的负数），使 softmax 后的权重接近 0。

- **Causal Mask（因果掩码）**：在 Decoder 中，保证位置 $`i`$ 只能看到位置 $`j \leq i`$ 的 token，防止“看到未来”。
  方法：构造一个上三角矩阵（不含对角线），对每个 `(i, j)` 其中 $`j > i`$ 的位置设为 `-inf`。

**Shape**：通常 mask 是 `[B, 1, 1, S]` 或 `[1, 1, S, S]`，通过广播机制与 `[B, H, S, S]` 对齐。

#### 2.3 结构层

##### ① 前馈网络（FFN）

**公式**：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

- $`W_1`$ 的形状：`[d_model, d_ff]`，通常 $`d_{ff} = 4 \times d_{model}`$。
- $`W_2`$ 的形状：`[d_ff, d_model]`。
- **Why 需要 FFN？**
  Attention 负责在**不同 token 之间**交换信息（线性加权），FFN 负责在**每个 token 内部**做非线性变换，提升模型表达能力。两者交替，形成了 Transformer 的“通信-计算”结构。

**现代变体**（如 LLaMA 使用 SwiGLU）：

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)
$$

效果更好，但参数略多。

##### ② 残差连接 + 层归一化（Residual + LayerNorm）

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
  - **Post-Norm**（原始）：$`\text{LN}(x + \text{Sublayer}(x))`$。收敛慢，需要 warmup，但理论表达能力强。
  - **Pre-Norm**（主流）：$`x + \text{Sublayer}(\text{LN}(x))`$。梯度流更顺畅，无需 warmup，训练稳定，但可能略低于 Post-Norm 的理论上限。几乎所有大模型（GPT、LLaMA）都用 Pre-Norm。

##### ③ 最后的 Softmax

在解码器输出后，通过一个线性层（`[d_model, vocab_size]`）将隐状态映射为 logits，再 softmax 得到概率分布，用于预测下一个 token。

### 3 Transformer三种架构

| 架构                | 代表模型                 | 核心结构                      | 注意力掩码                                       | 训练任务                                | 典型应用                           |
| ------------------- | ------------------------ | ----------------------------- | ------------------------------------------------ | --------------------------------------- | ---------------------------------- |
| **Encoder-Only**    | BERT, RoBERTa            | 堆叠 Encoder 层               | 双向（无掩码）                                   | MLM（掩码语言模型）+ NSP（可选）        | 理解类任务：分类、实体识别、相似度 |
| **Decoder-Only**    | GPT 系列, LLaMA, Mistral | 堆叠 Decoder 层（带因果掩码） | 因果（只能看左边）                               | 自回归语言建模（Next Token Prediction） | 生成类任务：对话、写作、代码、推理 |
| **Encoder-Decoder** | T5, BART, M2M100         | Encoder + Decoder             | Encoder：双向<br>Decoder：因果 + Cross-Attention | 去噪自编码（Span Corruption）或翻译     | 序列到序列：翻译、摘要、结构化转换 |

#### 3.1 深入对比与 Why

##### ① Encoder-Only（以 BERT 为例）

- **结构**：多层双向 Attention。
- **为什么双向？**
  理解任务（如情感分类）需要同时看到上下文，才能准确判断语义。比如 “not good” 中的否定需要结合后面的词才能理解。
- **训练任务**：**MLM**（Masked Language Model）——随机 mask 掉 15% 的 token，让模型预测。迫使模型学会利用双向上下文。
- **局限**：无法做生成（因为生成需要从左到右自回归）。

##### ② Decoder-Only（以 GPT 为例）

- **结构**：多层因果 Attention（上三角 mask）。
- **为什么因果？**
  生成文本时，只能根据已经生成的 token 预测下一个，不能“偷看”未来，符合自回归生成的逻辑。
- **训练任务**：自回归语言建模，即给定前文预测下一个 token。
- **为什么现在主流是大 Decoder-Only？**
  1. **通用性**：生成任务天然涵盖理解任务（通过 prompt 可以要求模型做分类、问答等）。
  2. **训练效率**：相比 Encoder-Decoder，参数利用率更高（没有额外 Encoder）。
  3. **涌现能力**：大规模的 Decoder-Only 模型（如 GPT-3、LLaMA）在上下文学习（In-context Learning）、思维链（CoT）等方面表现出色。

##### ③ Encoder-Decoder（以 T5 为例）

- **结构**：Encoder 对输入做双向编码，Decoder 通过 Cross-Attention 对齐 Encoder 输出，自回归生成。
- **为什么需要这种结构？**
  当输入和输出**长度差异大**或**结构不同**时（比如翻译：中文->英文，摘要：长文->短摘要），Encoder 专门负责理解源文本，Decoder 专门负责生成目标文本，中间用 Cross-Attention 进行对齐，比单一的 Decoder 更自然。
- **训练任务**：T5 采用 **Span Corruption** —— 将输入中若干连续 token 替换为一个哨兵 token，Decoder 需要恢复这些被 mask 的 span。这种训练方式结合了双向理解和自回归生成。

#### 3.2 关于 Decoder-Only 成为主流的深入思考

进一步的问题是：“既然 Encoder-Decoder 对 Seq2Seq 更自然，为什么大模型几乎都是 Decoder-Only？”

- **回答思路**：
  1. **规模效应**：在相同参数规模下，Decoder-Only 的 FLOPs 更低（因为不需要 Cross-Attention 的双向计算？不对，其实 Decoder-Only 也有双向？）—— 更准确地说，Decoder-Only 的**参数利用效率高**，所有参数都用于生成任务，而 Encoder-Decoder 中 Encoder 参数只用于理解源文本，生成时全靠 Decoder，参数利用率低。
  2. **指令微调与通用性**：Decoder-Only 通过 prompt 可以完成各种任务（如 “请分类：...” ），一个模型通吃。而 Encoder-Decoder 往往需要为不同任务调整结构。
  3. **涌现能力**：大规模 Decoder-Only 模型在上下文学习上表现出更强的涌现能力，可能与因果掩码带来的“顺序推理”特性有关。

**补充**：目前也有一些混合架构（如 Encoder-Decoder 的大型模型，例如 T5-11B），但主流开源模型（LLaMA, Mistral, Qwen）和闭源模型（GPT-4）都选择 Decoder-Only。

### 4 MoE

#### 4.1 MoE 的架构细节

##### Token Choice

这是最常见的方式，例如 Switch Transformer、Mixtral、DeepSeek-MoE 都采用 Token Choice。

**原理**：每个 token 独立地选择最合适的 Top-K 个专家。Router 对每个 token 输出一个对所有专家的概率分布，然后每个 token 挑选概率最高的 K 个专家，将自身的表示发送给这些专家，专家的输出按路由概率加权求和。

**公式**：
对于 token $`x`$，Router 输出 logits $`h(x) = W_g x`$（$`W_g \in \mathbb{R}^{E \times d}`$），然后 softmax 得到概率 $`p = \text{softmax}(h(x))`$。选择 Top-K 索引集合 $`\mathcal{T}`$，最终输出：

$$
y = \sum_{i \in \mathcal{T}} p_i \cdot \text{Expert}_i(x)
$$

**特点**：

- 每个 token 的计算量固定（K 个专家）。
- 不同 token 可能选择不同的专家，专家负载可能不均衡（有的专家被很多 token 选择，有的很少）。
- 需要辅助损失（负载均衡损失）来鼓励均匀分配。

##### Expert Choice

这是一种较少见但有趣的方式，由例如 "Mixture-of-Experts with Expert Choice" 论文提出。

**原理**：每个专家选择它要处理的 token，而不是 token 选择专家。具体来说，对所有 token 的路由分数，每个专家挑选分数最高的 Top-K 个 token（或者按容量选择）。专家输出后，再根据路由分数加权聚合回每个 token。

**公式**（简化）：
设 batch 中有 $`T`$ 个 token，每个 token 有路由分数 $`s_{t,i}`$ 表示 token $`t`$ 与专家 $`i`$ 的匹配度。专家 $`i`$ 选择分数最高的 $`C_i`$ 个 token（$`C_i`$ 可以是容量，如 $`C_i = \text{capacity\_factor} \times T/E`$）。被选中的 token 集合记为 $`\mathcal{T}_i`$，专家 $`i`$ 输出 $`y_{t,i} = \text{Expert}_i(x_t)`$。最终 token $`t`$ 的输出为：

$$
y_t = \sum_{i: t \in \mathcal{T}_i} \frac{s_{t,i}}{\sum_{t' \in \mathcal{T}_i} s_{t',i}} \cdot y_{t,i}
$$

即用该 token 在专家 i 的选中集合中的归一化分数作为权重。

**特点**：

- 负载天然均衡（每个专家处理固定数量的 token），强制均匀分布。
- 但每个专家处理的 token 数量固定，可能造成信息损失（如果某专家对所有 token 分数都很低，仍需强制选择一些 token）。
- 实现复杂度高，推理时难以动态适配。

**常见问题**：

- Token Choice 为什么需要辅助损失？Expert Choice 如何避免负载不均衡？
- 在实际大模型中，哪种更常用？为什么？（Token Choice 更灵活，实现简单，配合辅助损失效果好。）


#### 4.2 路由选择

Router的本质是一个线性层 $`W_g \in \mathbb{R}^{E \times d}`$，输入 token 的隐向量 $`x \in \mathbb{R}^d`$，输出 logits $`z = W_g x`$（维度 $`E`$，专家数量）。然后经过 softmax 得到概率分布。

**关点**：

- **噪声注入**（训练时）：Switch Transformer 等模型在路由 logits 中添加可调节的高斯噪声，鼓励探索，防止 Router 过早收敛到次优分配。公式：

$$
z_i = \frac{x \cdot W_g^{(i)} + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}}^{(i)})}{\text{temperature}}
$$

其中 $`\epsilon \sim \mathcal{N}(0,1)`$，$`W_{\text{noise}}`$ 是可学习的噪声参数。训练初期噪声大，后期逐渐降低。

- **温度系数**：可以引入温度 $`T`$ 来平滑或锐化分布。$`T<1`$ 使分布更尖锐（偏向最大专家），$`T>1`$ 更平滑。通常 $`T=1`$。

**常见问题**：

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
为了控制每个专家处理的 token 数量，常引入容量因子。每个专家的容量 = $`\text{capacity\_factor} \times \frac{\text{total\_tokens}}{E}`$。如果某个专家被分配的 token 超过容量，超出的 token 会被丢弃（或通过残差连接绕过专家）。容量因子通常设为 1.0~1.5，避免 token 被丢弃过多。

**常见问题**：

- 为什么 Top-2 比 Top-1 更好？（降低负载不均衡，提高模型容量）
- 容量因子过小或过大会有什么影响？（过小导致 token 被丢弃，信息损失；过大导致负载不均衡和计算浪费）
- 如何解决 Token 被丢弃的问题？（使用更大的容量因子，或使用 Expert Choice）

#### 4.3 MoE 的路由坍塌

**路由坍塌（Routing Collapse）** 是指 Router 将所有 token 都分配给少数几个专家，导致其他专家几乎不被训练，模型退化为一个小型 Dense 模型，失去了 MoE 的优势。这是 MoE 训练中最常见的问题。

导致坍塌的原因：

- 早期训练时，Router 随机初始化，某个专家偶然获得稍高的分数，该专家得到更多 token → 该专家梯度更新更多 → 它变得更擅长处理更多 token → 正反馈循环，其他专家逐渐被“饿死”。
- 缺乏足够的探索，Router 过早陷入局部最优。

解决方案：

##### 辅助损失（Auxiliary Loss）

这是最常用的方法，在训练目标中加入一个辅助损失，惩罚负载不均衡。常见的两种形式：

**a) Importance-based Loss（Switch Transformer）**

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

其中：

- $`f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{\text{token } t \text{ 选择专家 } i\}`$，即专家 $`i`$ 被选中的 token 比例。
- $`P_i = \frac{1}{T} \sum_{t=1}^{T} p_{t,i}`$，即所有 token 对专家 $`i`$ 的平均路由概率。
- $`\alpha`$ 是系数，通常取 0.01。

**解释**：当专家 $`i`$ 被选中的频率 $`f_i`$ 高，同时 Router 给它的平均概率 $`P_i`$ 也高时，损失大。这鼓励 Router 使 $`f_i`$ 和 $`P_i`$ 都接近 $`1/E`$，即均匀分布。

**b) Load-based Loss（GShard）**
直接基于每个专家实际处理的 token 数量 $`l_i`$ 计算方差或与均值的差异：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} \left( \frac{l_i}{T} - \frac{1}{E} \right)^2
$$

更直接地强制每个专家处理的 token 数量相等。

**常见问题**：

- 辅助损失如何与主损失（如语言建模损失）平衡？系数 $`\alpha`$ 如何选择？（通常很小，如 0.01，否则会干扰主任务）
- 辅助损失是否会影响模型性能？（适当使用可提升性能，因为负载均衡本身也有利于充分利用专家容量）

##### 熵正则化（Entropy Regularization）

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

**常见问题**：

- 熵正则化为什么能缓解路由坍塌？（防止 Router 输出尖锐分布，迫使每个 token 考虑多个专家）
- 熵正则化会不会导致每个 token 选择的专家过于分散，降低模型能力？（通过调节 $`\alpha`$ 可以平衡）

##### 硬约束（Hard Constraints）

不通过损失惩罚，而是直接对路由施加硬性限制，确保负载均衡。

**a) Expert Capacity 限制**
每个专家设置最大 token 容量（如 $`\text{capacity} = \lceil \frac{\text{total\_tokens}}{E} \times \text{capacity\_factor} \rceil`$）。当某个专家被分配的 token 达到容量后，后续选择该专家的 token 会被强制重定向到其他专家（或直接丢弃/绕过）。

**实现**：在训练时，记录每个专家已处理的 token 数量，当超过容量时，将该 token 的该专家分数设为 $-\infty$，使其不再被选中。

**b) 强制均匀采样（Stochastic Routing）**
在训练初期，以一定概率随机分配专家（无视 Router 分数），强制每个专家都有机会训练。随着训练进行，逐渐退火到完全由 Router 决定。

**常见问题**：

- 硬约束与软约束（辅助损失）的优缺点比较？硬约束保证绝对均衡，但可能丢弃 token 损失信息；软约束更平滑，但可能无法完全均衡。
- 容量因子如何设置？过小导致大量 token 被丢弃，过大则失去均衡作用。通常设为 1.0~1.5。=

##### 综合对比

| 方法         | 原理                   | 优点                       | 缺点                         |
| ------------ | ---------------------- | -------------------------- | ---------------------------- |
| **辅助损失** | 在损失中加惩罚项       | 平滑，不影响 token 分配    | 需要调整系数，可能干扰主任务 |
| **熵正则化** | 鼓励 token 级分布均匀  | 简单，防止 Router 过早尖锐 | 可能降低专家专业化程度       |
| **硬约束**   | 强制容量限制或随机分配 | 确保绝对均衡，直接有效     | 可能丢弃 token，实现复杂     |

在实际大模型（如 Mixtral、DeepSeek-MoE）中，通常**组合使用**多种方法：主要依赖辅助损失（负载均衡损失），配合熵正则化，同时设置合理的容量因子（硬约束），以保证训练稳定性和专家利用率。

### 5 多模态大模型：从图像到语言 Token

文本模型接收离散 Token ID，而图像本质上是像素矩阵。多模态模型首先要解决的不是“让 LLM 看图”，而是三个更具体的问题：

1. 如何把大小不一的图像转换成一串视觉 Token；
2. 如何把视觉特征对齐到 LLM 的表示空间；
3. 如何让文本 Token 在生成过程中读取并引用视觉信息。

先看一条最常见的视觉语言模型链路：

```text
Image → Patchify → Vision Encoder → Projector / Resampler
      → Visual Tokens + Text Tokens → LLM → Text / Tool Call / JSON
```

#### 5.1 第一步：把图像切成 Patch

ViT 把图像切成固定大小的 Patch，再把每个 Patch 展平并线性投影。若输入尺寸为 $H\times W$，Patch 尺寸为 $P\times P$，视觉 Token 数为：

$$
N_{vision}=\frac{H}{P}\times\frac{W}{P}
$$

例如 `448×448` 图像使用 `14×14` Patch，会产生 `32×32=1024` 个 Patch Token。长宽同时翻倍时，Token 数变为 4 倍；视觉细节增加的同时，LLM 的 Context、Prefill 和 KV Cache 压力也随之上升。

Patch Embedding 本质上可用一个步长等于 Patch 大小的卷积实现：

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, hidden_size=1024, patch_size=14):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, image):              # [B, 3, H, W]
        feature = self.proj(image)          # [B, D, H/P, W/P]
        return feature.flatten(2).transpose(1, 2)  # [B, N, D]
```

Patch 越小，局部细节越充分，但 Token 数和计算成本越高；Patch 越大则更便宜，但 OCR、小物体和细粒度定位更容易丢失信息。

#### 5.2 第二步：Vision Encoder 提取视觉语义

Patch Embedding 只完成像素分块，还不包含高级语义。Vision Encoder 通常是 ViT：为 Patch Token 加入二维位置信息，再经过多层 Self-Attention 和 FFN，使一个局部 Patch 能融合整张图的上下文。

二维位置信息尤其重要：文字“猫在桌子下面”不仅依赖物体是什么，还依赖它们的相对位置。常见方案包括绝对位置 Embedding、二维 RoPE、相对位置 Bias。输入分辨率变化后，固定位置 Embedding 还需要插值；处理不当会导致高分辨率或非训练宽高比下能力下降。

Vision Encoder 的输出通常为：

$$
F_v\in\mathbb{R}^{B\times N_{vision}\times D_v}
$$

它已经包含视觉语义，但维度 $D_v$、数值分布和 Token 数量通常都与 LLM 不匹配，因此不能简单地把它当作文本 Embedding 使用。

#### 5.3 第三步：Projector 对齐视觉与语言空间

Projector 将视觉特征从 $D_v$ 映射到 LLM hidden size $D_l$：

$$
F_l=Projector(F_v),\qquad F_l\in\mathbb{R}^{B\times N'\times D_l}
$$

常见方案的区别是“保留多少视觉 Token”：

| 方案 | 做法 | 特点 |
|---|---|---|
| Linear / MLP | 每个 Patch 独立映射到 LLM 维度 | 简单、细节保留多，但视觉 Token 较长 |
| Q-Former / Resampler | 用固定数量的 Query 从全部视觉特征中抽取信息 | Token 数稳定，但压缩可能损失 OCR 和局部细节 |
| Token Merger | 合并相邻或相似 Token | 在细节与计算之间动态折中 |

最简单的 Projector 只是一个 MLP：

```python
projector = nn.Sequential(
    nn.Linear(vision_dim, llm_dim),
    nn.GELU(),
    nn.Linear(llm_dim, llm_dim),
)
visual_tokens = projector(vision_features)
```

Projector 不是单纯“修改维度”。训练会让它学习：哪些视觉模式应落到 LLM 已有的语义空间，以及视觉 Token 应以怎样的分布进入语言模型。

#### 5.4 第四步：视觉 Token 如何进入 LLM

主流融合方式可以分成两类。

**Token 拼接式（Decoder-only）**：把 `<image>` 占位符替换为视觉 Embedding，再与文本 Embedding 一起送入 LLM：

```text
<BOS> User: [IMG_START] v1 v2 ... vN [IMG_END] 描述这张图
Assistant: ...
```

```python
text_embeds = llm.embed_tokens(input_ids)
inputs_embeds = replace_image_placeholders(text_embeds, visual_tokens)
logits = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
```

这类架构复用原有 Decoder-only LLM，工程简单；代价是视觉 Token 直接占用 Context，且所有视觉信息都参与 LLM Self-Attention。

**Cross-Attention 式**：文本仍走 LLM 主干，在部分层增加 Cross-Attention，让文本 Query 读取视觉 K/V。它不必把全部视觉 Token 塞进文本序列，但需要修改 LLM 结构和训练方式。

无论采用哪种方式，模型最终仍是自回归预测下一个文本 Token。视觉信息改变的是每一步预测所能读取的条件，而不是输出头的基本形式。

#### 5.5 分辨率、宽高比与多图输入

固定 Resize 会把所有图像拉成同一尺寸，Batch 简单，但可能造成形变或细节丢失。常见改进包括：

- **Dynamic Resolution**：保留宽高比，将图像调整到若干允许尺寸，并动态产生视觉 Token；
- **Tiling / Any-resolution**：保留一张低分辨率全局图，同时把原图切成多个高分辨率局部块；
- **Thumbnail + Crops**：全局缩略图负责整体关系，局部 Crop 负责 OCR 和细节。

Tiling 的难点是局部块数量可能暴涨，而且模型必须知道每块在原图中的空间位置。多图输入还要加入图像边界和序号，防止模型把图 1 的证据错误归到图 2。

#### 5.6 多模态模型如何训练出来

多模态能力通常不是一步 SFT 得到，而是逐阶段对齐：

1. **Feature Alignment**：冻结 Vision Encoder 与 LLM，只训练 Projector，用图文对让视觉特征进入语言空间；
2. **Multimodal Pretraining**：使用大规模 Caption、OCR、文档和交错图文数据，学习视觉概念与语言之间的对应；
3. **Multimodal SFT**：训练图像问答、图表理解、Grounding、多轮对话和结构化输出；
4. **Preference Alignment**：通过 DPO/RLHF 等方法改善视觉幻觉、拒答、安全和回答风格。

第一阶段只解决“看见的特征能否被 LLM 接收”；SFT 才进一步解决“是否按照用户指令使用这些信息”。如果 Vision Encoder 没有提取出小字或目标边界，只训练语言侧 LoRA 也无法凭空恢复这些信息。

#### 5.7 训练目标与 Label Mask

最常见的训练目标仍是文本的 Next Token Prediction。图像占位符、视觉 Token、System 和 User 内容用于提供条件，通常只对 Assistant 回答计算语言模型损失：

```text
图像 Token：mask    用户问题：mask    Assistant 答案：计算 loss
```

若任务包含 Bounding Box、Point、OCR 坐标或 Tool Call，可以把位置离散成特殊 Token 继续使用语言模型损失，也可以增加单独的检测/对齐损失。前者协议统一，后者通常对连续空间定位更直接。

#### 5.8 推理成本与常见瓶颈

一次多模态请求通常包含两段 Prefill：先运行 Vision Encoder，再让 LLM 消化视觉 Token 与文本 Token。主要瓶颈包括：

- 高分辨率或多图导致视觉 Token 数过多，TTFT 明显增长；
- 不同图像 Token 数差异大，Batch 中 padding 浪费严重；
- Vision Encoder 与 LLM 的计算形态不同，资源利用不均；
- 图片预处理、下载和解码可能成为 GPU 之外的延迟来源；
- 视觉特征缓存可以避免重复编码，但必须绑定图片内容、预处理配置和模型版本。

因此，多模态服务不能只按文本 Token 限流，通常要将图像数量、像素数或估算后的视觉 Token 一起计入 Admission Control。

**常见问题**

1. **Projector 与 Vision Encoder 分别解决什么问题？**
   Vision Encoder 从像素提取语义；Projector 将视觉语义映射或压缩到 LLM 能消费的表示空间。
2. **为什么分辨率翻倍后成本可能接近四倍？**
   长宽同时翻倍时 Patch 数变为四倍，并进一步增加 LLM Prefill 的序列长度。
3. **Token 拼接与 Cross-Attention 如何选择？**
   拼接式结构简单、便于复用现有 LLM；Cross-Attention 能更独立地读取视觉特征，但需要修改模型结构。
4. **视觉幻觉只是语言模型的问题吗？**
   不是。它可能来自视觉特征缺失、模态对齐不足、训练数据偏差，也可能来自解码阶段语言先验压过视觉证据。

## 二、后训练、对齐与推理工程

### 1 SFT 与参数高效微调

#### 1.1 SFT 数据标注格式

SFT 的核心是将预训练模型转化为能够遵循指令的对话助手。数据标注格式决定了模型如何理解用户输入和期望输出。

**基础格式**
最常用的格式是 **对话结构（conversations）**，每条数据包含一个多轮对话的列表，每个元素带有 `role` 和 `content` 字段。例如：
```json
{
  "message": [
    { "role": "user", "content": "什么是 Transformer 的注意力机制？" },
    { "role": "assistant", "content": "Transformer 的注意力机制包括缩放点积注意力和多头注意力……" }
  ]
}
```
其中 `role` 通常有 `system`（系统指令，可选）、`user`（用户）、`assistant`（模型回答）。`system` 用于设定全局行为，如“你是一个乐于助人的助手”。

**多轮对话的处理**
对于多轮对话，数据会按顺序展开为 `user` 和 `assistant` 的交替，例如：
```json
{
  "message": [
    { "role": "user", "content": "帮我写一个 Python 函数，计算斐波那契数列。" },
    { "role": "assistant", "content": "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)" },
    { "role": "user", "content": "能否用迭代方式优化？" },
    { "role": "assistant", "content": "当然，迭代版本如下：\ndef fib_iter(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a" }
  ]
}
```

**模板化**
在训练前，原始对话需要转换成模型特定的 **对话模板（chat template）**。例如 Llama 2 使用：
```
<s> [INST] {user_message} [/INST] {assistant_message} </s>
```
而 ChatML（如 OpenAI 的格式）则为：
```
<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>
```
错误的模板会导致模型行为混乱，因此 SFT 的第一步就是确认模型使用的模板，并在数据预处理时统一应用。

**数据质量与规模**
- **多样性**：需要覆盖不同任务（问答、编程、翻译、数学推理、创意写作等），避免数据分布偏斜导致模型只擅长某类任务。
- **准确性与一致性**：标注员需遵循统一的回答风格（如长度、格式、语气），同时确保答案正确（对于编程、数学类尤其严格）。
- **规模**：通常数千到数十万条高质量数据即可显著提升模型指令遵循能力。数据量并非越多越好，低质量数据反而会损害模型。

**常见问题**
- 如何构造高质量 SFT 数据？（数据来源、清洗、多样性控制）
- 模板错误会带来什么影响？（模型可能输出空内容、格式错乱或拒绝回答）
- 多轮对话在数据中如何组织？（需要保持上下文连贯，不能只保留最后一轮）


#### 1.2 SFT 端到端工程流程

SFT 不是「把 JSONL 丢给 Trainer」，而是一条从任务定义到部署回归的数据与模型工程链路：

```text
任务定义 → 数据规范 → 采集/合成 → 清洗去重 → 切分
        → Chat Template → Tokenize/Pack → 训练
        → 离线评测 → 误差分析 → 数据回流 → 合并/部署
```

##### 1.2.1 任务与输出契约

首先定义输入边界、输出 Schema、可接受错误、拒答规则和评估指标。分类任务要明确标签空间，生成任务要明确格式和事实依据，Tool-use 任务还要定义 Tool Call 与 Tool Result 的训练序列。如果训练目标不与线上解码契约对齐，后面的格式修补会成为无底洞。

##### 1.2.2 数据构建与治理

数据可来自人工标注、历史产品数据、教师模型合成、规则系统和 hard-case 回流。清洗通常包括：

- Schema 合法性、编码和字段完整性检查；
- 精确去重与 MinHash/Embedding 近重去重；
- PII、密钥、毒性内容和版权风险过滤；
- 过短、过长、自相矛盾、无依据和模板泄漏样本处理；
- 类别、长度、难度、来源与语言分布统计。

数据切分要在去重之后进行，并尽量按用户、文档、题目模板或时间簇分组，防止同源样本同时进入 train/test 造成泄漏。每条数据保留 `source`、`version`、`quality_score`、`created_at` 和处理链路，便于定位污染来源。

##### 1.2.3 Chat Template、Label Mask 与 Packing

Tokenizer 与 Chat Template 必须与基座模型一致，包括 BOS/EOS、role token、assistant generation prompt 和 Tool Call 特殊 token。一般只对 assistant 区间计算损失：

```python
input_ids = apply_chat_template(messages)
labels = input_ids.copy()
labels[system_user_tool_result_positions] = -100
loss = cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

是否对 Tool Call、reasoning 或特定结构计算损失取决于线上协议。必须用小样本可视化 `input_ids`、解码文本和 label mask，避免训练到用户输入或把 assistant 答案全部 mask 掉。

Packing 将多条短样本拼入一个定长序列提高 token utilization。不同样本之间必须有 EOS，并根据实现选择 block-diagonal attention mask 或允许因果 attention 跨样本但用 EOS 隔离。后者实现简单，但会引入轻微的跨样本关联。

##### 1.2.4 训练配置与稳定性

关键超参包括 learning rate、warmup ratio、effective batch size、epoch、max sequence length、weight decay 和 gradient clipping。

```text
effective_batch_tokens
  = micro_batch_size
  × gradient_accumulation_steps
  × data_parallel_world_size
  × average_non_padding_tokens
```

大模型通常使用 BF16；显存不足时结合 gradient checkpointing、Flash Attention、ZeRO/FSDP 和 sequence packing。训练时监控 train/eval loss、gradient norm、learning rate、tokens/s、padding ratio、OOM/NaN 及各数据分组损失。损失下降不等于任务变好，必须同时跑生成评测。

##### 1.2.5 评测、误差分析与数据飞轮

评测集要同时包含核心分布、长尾分布、对抗样本和回归样本。指标分为：

- **任务指标**：Accuracy/F1、Pass@k、事实性、人工偏好；
- **协议指标**：JSON 合法率、Schema 通过率、Tool Call 正确率；
- **能力回归**：通用问答、安全、多语言和长上下文能力是否退化；
- **服务指标**：TTFT、TPOT、显存和吞吐。

误差不应只按「对/错」分类，而应标记为数据缺失、标注冲突、指令理解、知识不足、推理错误、格式错误和过度拒答等类型。数据飞轮优先回流高价值 hard case，同时保留 replay buffer，避免新数据覆盖旧能力。

**常见问题**

1. **Loss 下降但生成异常，优先查什么？**
   优先检查 Chat Template、BOS/EOS、label shift 和 loss mask，确认模型训练与线上推理使用同一套 Token 协议。
2. **为什么要先去重再切分数据？**
   先随机切分会让同源或近重样本同时出现在 train/test，造成测试指标虚高。
3. **为什么不能只用 eval loss 选 checkpoint？**
   eval loss 不直接等于生成质量，还需要任务指标、协议合法率、人工偏好和通用能力回归。


#### 1.3 LoRA 与其他 PEFT 方法的区别

PEFT（Parameter-Efficient Fine-Tuning）旨在用极少的可训练参数达到接近全量微调的效果，尤其适合 MoE 这种参数巨大的模型。

**LoRA（Low-Rank Adaptation）**
- **原理**：假设微调时的权重变化 $`\Delta W`$ 是低秩的，即 $`\Delta W = BA`$，其中 $`B \in \mathbb{R}^{d_{\text{out}} \times r}`$，$`A \in \mathbb{R}^{r \times d_{\text{in}}}`$，$`r \ll \min(d_{\text{in}}, d_{\text{out}})`$。原始前向传播变为 $`h = W_0 x + BA x`$，训练时只更新 $`B`$ 和 $`A`$，$`W_0`$ 冻结。
- **优点**：推理时可将 $`BA`$ 合并到 $`W_0`$ 中，不增加额外延迟；参数量极少（通常 r=8~64），效果好，社区支持广泛。
- **缺点**：需要选择合适的秩 $`r`$；如果模型本身已经过拟合，低秩假设可能限制表达能力。

**Adapter**
- **原理**：在 Transformer 的每个子层（通常 FFN 后）插入一个小型 MLP，结构为“降维 → 激活 → 升维”，例如先将 768 维降为 64 维，再升回 768 维。只训练这些 Adapter 参数。
- **优点**：结构简单，参数数量可控（通常为模型参数的 0.5%~5%）。
- **缺点**：推理时会引入额外的计算层，增加延迟（无法像 LoRA 那样合并）。

**Prefix Tuning / Prompt Tuning**
- **原理**：在输入序列前添加一组可训练的连续向量（prefix 或 prompt）。这些虚拟 token 的表示会随着训练更新，从而影响模型输出。
- **区别**：Prefix Tuning 在每一层都加入 prefix，Prompt Tuning 只在输入层加入。
- **优点**：参数极少（例如 100 个虚拟 token × 维度），完全不修改模型权重。
- **缺点**：需要占用输入长度（减少可用上下文长度）；对模型规模敏感（小模型效果较差）。

**IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）**
- **原理**：对注意力机制的 K、V 以及 FFN 的输入分别乘以可学习的缩放向量（即对特征维度做逐元素缩放），每个向量长度等于特征维度。
- **优点**：参数量极少（三个向量，约 $`3 \times d_{\text{model}}`$），效果在不少任务上接近 LoRA。
- **缺点**：实现相对小众，社区支持不如 LoRA。

**常见问题**
- **LoRA 为什么有效？** 微调过程中，权重的变化往往位于一个低秩子空间，这已被实验和理论验证（如 Aghajanyan et al. 2021）。
- **LoRA 在 MoE 中如何应用？** 可以只对每个专家的 FFN 权重加 LoRA，也可以对 Attention 层加。通常为每个专家独立添加低秩矩阵，可训练参数仍然远少于全量微调。
- **PEFT 方法如何选择？** 如果追求推理速度且显存充足，LoRA 是首选；如果模型极大且希望参数极少，Prompt Tuning 可以一试；如果必须用旧版模型且不想改动原始权重，Adapter 也是常见选择。
- **全量微调 vs PEFT：** 当有大量高质量数据且需要最大化下游性能时，全量微调仍然是最佳选择，但显存和训练时间要求高。


#### 1.4 LoRA 端到端工程流程

##### 1.4.1 注入位置与参数量

对线性层 $W \in \mathbb{R}^{d_{out}\times d_{in}}$，LoRA 学习：

$$
y = W x + \frac{\alpha}{r} B A x
$$

其中 $A\in\mathbb{R}^{r\times d_{in}}$、$B\in\mathbb{R}^{d_{out}\times r}$。单层可训练参数为 $r(d_{in}+d_{out})$。常见 target modules 是 Attention 的 `q_proj/k_proj/v_proj/o_proj` 和 MLP 的 `gate_proj/up_proj/down_proj`。只选 Q/V 更省参数；同时选 Attention+MLP 通常容量更强。选择应通过消融实验，不要依赖固定配方。

常用初始化是 $A$ 随机、$B=0$，使训练开始时 LoRA 分支输出为 0，模型与基座模型完全一致。`lora_dropout` 仅应在数据小、过拟合明显时使用。

下面这个最小实现把公式中的两条路径直接对应到代码。`base` 冻结，反向传播只更新 `A/B`：

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)
```

这里最值得检查的不是代码能否运行，而是：目标 Linear 是否真的被替换、基座参数是否冻结、`B=0` 是否保证初始输出不变，以及 merge 前后 logits 是否足够接近。

##### 1.4.2 LoRA 训练步骤

1. 固定 base model revision、tokenizer、chat template 和数据版本；
2. 加载基座权重，冻结所有原始参数；
3. 按模块名注入 LoRA，打印 trainable/total parameter ratio 并检查遗漏；
4. 对一个 micro batch 做 forward/backward，确认只有 LoRA 和显式允许的 bias/norm 有梯度；
5. 使用独立的 LoRA learning rate 训练，按步保存 adapter、optimizer、scheduler 和 RNG state；
6. 跑离线指标与通用能力回归，选择 checkpoint；
7. 决定在线动态加载 adapter，还是先 merge 成独立权重再部署。

##### 1.4.3 QLoRA

QLoRA 将冻结的基座权重以 4-bit 形式加载，计算时反量化到 BF16/FP16，LoRA 参数和优化器状态仍使用较高精度。典型组合是 NF4、double quantization 和 paged optimizer。QLoRA 显著降低训练显存，但反量化会增加计算开销，且 4-bit base + adapter 不能等同于将 adapter 随意 merge 回 4-bit 权重。

##### 1.4.4 Merge、验证与部署

合并操作为 $W' = W + \frac{\alpha}{r}BA$。建议在 FP32/BF16 中合并，再根据目标服务格式量化，避免在低精度权重上反复 merge/unmerge 导致误差累积。合并前后用固定输入比较 logits 或生成结果，并检查 tokenizer、generation config 和特殊 token 是否一起发布。

动态 Adapter Serving 能让多个任务共享基座权重，但需要解决 adapter 缓存、版本路由、租户隔离、batch 内 adapter 切换和冷加载延迟。合并部署的运行时更简单，但每个任务都需要独立权重副本。

##### 1.4.5 常见失败模式

- 模块名未匹配，实际没有任何 LoRA 参数参与训练；
- Chat Template 错误或 label mask 偏移，损失正常但生成异常；
- rank/alpha 过大、学习率过高导致灾难性遗忘；
- 只看 loss 选 checkpoint，忽略格式和通用能力回归；
- 训练时与推理时 tokenizer、RoPE 配置或 special tokens 不一致；
- merge 后立即低比特量化，但未重新做端到端评测。

**常见问题**

1. **rank 和 alpha 分别控制什么？**
   rank 决定低秩更新的表达容量与参数量，`alpha / rank` 控制 LoRA 分支相对于基座权重的缩放。
2. **QLoRA 是否整个训练都在 4-bit 中进行？**
   不是。冻结基座权重以 4-bit 存储，前向时反量化；LoRA 权重、梯度和优化器状态仍使用更高精度。
3. **Merge 完成后为什么还要重新评测？**
   Merge 精度、后续量化、tokenizer 或 generation config 错配都会导致结果偏移，必须用固定样本比对 logits 或生成结果。



#### 1.5 知识蒸馏

知识蒸馏的目标是让小模型学会大模型的决策边界和输出行为，而不只是记住人工标签。根据能否访问教师内部信号，可分为两条路线：

- **White-box Distillation**：可获取教师 logits、hidden states 或 attention；
- **Black-box Distillation**：只能获取教师生成的文本、结构化结果或判别依据。

##### 1.5.1 Logit Distillation

分类任务中，教师的软分布包含类别之间的相对关系：

$$
\mathcal{L}=(1-\alpha)\mathcal{L}_{CE}(y_s,y)
+\alpha T^2 KL(p_t^T\parallel p_s^T)
$$

$$
p^T=softmax(z/T)
$$

$T>1$ 将分布变平，暴露非目标类的相对概率；$T^2$ 用于补偿温度带来的梯度缩放。KL 方向应明确表示「用学生逼近教师」，实现时需核对库函数的 input/target 语义。

PyTorch 的 `kl_div(input, target)` 要求 `input` 是 log-probability，因此实现时教师与学生的位置不能写反：

```python
def distill_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits.detach() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return (1 - alpha) * hard + alpha * soft
```

教师输出要 `detach`，否则会无意中为教师构建反向图；生成任务还需对 padding 和非目标 Token 做 mask，不能直接把所有位置平均。

生成模型的 logits 维度是 vocabulary，完整存储成本很高，可以只保留 top-k logits 与剩余概率质量，但要注意 teacher/student tokenizer 不同时无法直接对齐 Token 分布。

##### 1.5.2 Sequence-level Distillation

在只能调用教师 API 时，使用教师生成的完整答案训练学生：

```text
未标注输入
  → 教师多次生成
  → 规则/模型/人工校验
  → 去重、过滤与难度分层
  → 与人工数据混合
  → 学生模型 SFT
```

教师的「判别依据 + 结论」比单一标签包含更多过程信号，但 reasoning 不能默认为真实可靠。数据过滤应对结论合法性、证据一致性和任务正确性分别验证。

##### 1.5.3 蒸馏数据过滤

高质量蒸馏数据通常需要：

- **Schema/候选合法性**：结果必须属于允许空间；
- **Self-consistency**：对同一输入多次采样，保留结论稳定的数据；
- **Confidence/Margin**：保留教师高置信或第一、第二候选差距足够大的数据；
- **Agreement**：规则、多教师或多 Prompt 输出相互一致；
- **Deduplication**：防止高频模板数据占据训练主体；
- **Difficulty Balance**：保留部分边界样本和 Hard Case，不能只留教师最自信的简单题。

过度过滤会让数据只剩容易样本，学生离线指标看似很高，但决策边界反而没学会。应根据难度分桶采样，对不同置信区间设置不同权重。

##### 1.5.4 训练与评估

学生训练时需要混合人工真实数据与教师合成数据。只用合成数据会把教师偏差与错误一起复制给学生，也容易让输出风格过度单一。

评估至少包括：

- 学生相对于教师和未精调基座模型的任务差距；
- 按难度、类别、长度和数据来源分层的指标；
- 通用能力、拒答、格式与安全回归；
- 参数量、显存、TTFT、TPOT 与吞吐的效率收益。

**常见问题**

1. **没有教师 logits 还能做蒸馏吗？**
   可以，用教师生成的序列、结构化结果或判别依据做 sequence-level distillation。
2. **为什么不能只保留教师高置信样本？**
   高置信样本通常偏简单，只训练它们会丢失决策边界和长尾能力。
3. **学生模型比教师更好是否矛盾？**
   不矛盾。在特定窄域任务上，数据过滤、多教师集成和更匹配的训练分布可以让学生超过单次教师输出，但不代表学生的通用能力更强。

#### 1.6 多模态 SFT 全流程

前文已经说明视觉 Token 如何经过 Vision Encoder、Projector 进入 LLM。本节只关注在这一模型结构上如何组织监督数据、选择可训练模块并完成领域微调。

##### 1.6.1 数据格式

一条数据需要同时表达对话和媒体引用：

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "path": "images/001.jpg"},
      {"type": "text", "text": "请提取图中的表格。"}
    ]},
    {"role": "assistant", "content": "{\"rows\": [...]}"}
  ]
}
```

图像顺序必须与 `<image>` 占位 Token 一致，多图输入还要明确「图 1 / 图 2」的指代。训练与推理必须共用相同的 image processor、resize/crop 策略、归一化参数和 chat template。

##### 1.6.2 训练哪些模块

| 策略 | 显存/成本 | 适用场景 |
|---|---:|---|
| 只训 Projector | 最低 | 快速做模态对齐，但能力上限受限 |
| Projector + LLM LoRA | 中 | 领域指令跟随，常见性价比方案 |
| Vision LoRA + Projector + LLM LoRA | 较高 | 视觉域偏移明显，如医疗/卫星/工业图像 |
| 端到端全量微调 | 最高 | 数据与算力充足，追求能力上限 |

只对 LLM 做 LoRA 能改善指令遵循，但如果问题来自 Vision Encoder 没有提取出关键细节，语言侧无法凭空恢复丢失的视觉信息。

##### 1.6.3 Batch 与显存

图像分辨率不同会导致视觉 Token 数差异很大。如果只按样本数组 batch，少量高分辨率图像就可能 OOM。更稳定的方式是按「文本 Token + 视觉 Token」总预算做 bucketing 和 dynamic batching。

多模态训练常结合 BF16、FlashAttention、gradient checkpointing、sequence packing 和 LoRA/QLoRA。Packing 时要确保媒体特征的 offset 与文本占位 Token 一致，不能只拼接 `input_ids`。

##### 1.6.4 数据配比与能力退化

数据应同时覆盖：

- 普通图像问答与多轮对话；
- OCR、文档、表格和图表理解；
- 空间关系、局部细节、Grounding 与计数；
- 纯文本指令数据，用于保持语言能力；
- 图像无法支持结论时的拒答与不确定性数据。

如果训练集中每张图都必然有答案，模型容易学会在证据不足时也强行描述，形成视觉幻觉。

##### 1.6.5 评估

多模态评估不能只使用通用 VQA Accuracy，还应分别观察：

- OCR/文档中的 exact match、ANLS 或字符编辑距离；
- 结构化输出的 JSON Schema 通过率和字段准确率；
- Grounding 的 IoU/Pointing Accuracy；
- 视觉幻觉、错误引用和证据不足时的拒答率；
- 不同分辨率、宽高比、图像数量和文本长度下的分层结果；
- 纯文本能力是否因多模态 SFT 退化。

**常见问题**

1. **为什么需要 Projector？**
   Vision Encoder 输出与 LLM 的维度和特征分布不同，Projector 负责将视觉特征对齐到语言 Token 空间。
2. **为什么分辨率翻倍后成本不是简单翻倍？**
   长宽同时翻倍时 Patch Token 数约变为 4 倍，进而增加 Prefill 和 Context 成本。
3. **多模态 SFT 为什么要混入纯文本数据？**
   防止模型过度适应图文数据分布，导致原有语言理解、指令跟随和格式能力退化。

### 2 RLHF（基于人类反馈的强化学习）

#### 2.1 RLHF 数据标注格式

RLHF 通常分三步走：**SFT**（使用监督数据微调）、**奖励模型训练**、**强化学习**。这里聚焦后两步的数据标注。

**奖励模型数据**
奖励模型需要学习人类偏好，因此数据格式通常是 **比较（comparison）** 而非绝对分数。一个 prompt 对应多个回答（通常是 4 到 9 个），由标注员 **排序**，再转换成成对数据用于训练。

**数据组织方式**
常见的数据结构是将每个 prompt 下的最佳回答（chosen）和最差回答（rejected）作为一对，存储为：
```json
{
  "messages": [
    { "role": "user", "content": "写一首关于春天的诗" }
  ],
  "chosen": { "role": "assistant", "content": "春风拂面柳丝长，桃花映日笑颜芳。燕子归来寻旧垒，人间处处是春香。" },
  "rejected": { "role": "assistant", "content": "春天来了，花儿开了，真好。" }
}
```

对于多个回答，可以生成多对 `(chosen, rejected)`，例如从 4 个回答中选出最优和最差得到一对，或者用所有可能的两两组合（但数据会膨胀）。

**标注规范**
- 标注员需要根据 **有用性（helpfulness）**、**真实性（truthfulness）**、**无害性（harmlessness）** 等标准进行排序。
- 为了避免个人偏好偏差，通常采用多个标注员交叉验证，或采用“按多数排序”的方式。
- 数据量：奖励模型通常需要数万到数十万条比较数据，才能学习到稳定的偏好。

**常见问题**
- **为什么用排序而不是打分？** 不同人对同一回答的绝对分数差异很大（有人给 3 分，有人给 5 分），但排序的一致性更高。排序还能减少标注员疲劳，提升数据质量。
- **如何保证标注一致性？** 通过标注员培训、多人标注并取多数、设计清晰的标注指南。
- **数据规模要求**：奖励模型的性能与数据量正相关，但存在边际收益递减，通常 10 万条比较数据已能取得不错效果。


#### 2.2 奖励模型训练

**模型结构**
奖励模型通常基于 SFT 模型（也可以使用更小的模型），将最后的语言建模头替换为一个 **标量输出头**（如线性层加 sigmoid），用于输出一个奖励值 $`r(x,y)`$，表示在 prompt $`x`$ 下回答 $`y`$ 的好坏。

**损失函数**
使用 **pairwise ranking loss**（对比损失）：

$$
\mathcal{L} = -\log \sigma\left(r_\theta(x, y_{\text{chosen}}) - r_\theta(x, y_{\text{rejected}})\right)
$$

其中 $`\sigma`$ 是 sigmoid 函数。该损失鼓励模型给 chosen 的回答打更高的分，rejected 的回答打更低的分。

**训练技巧**
- **批处理**：由于每个 prompt 可能有多对 `(chosen, rejected)`，需要确保同一 prompt 下的所有对在同一个 batch 中，以便正负例来自相同上下文。
- **正则化**：使用权重衰减、dropout 等防止过拟合。
- **评估**：在验证集上计算 **一致性（accuracy）**，即模型对 $`r(x, y_{\text{chosen}}) > r(x, y_{\text{rejected}})`$ 的正确比例。此外，也可用人机对比评估。

**常见问题**
- **为什么不用均方误差（MSE）预测绝对分数？** 因为绝对分数难以标注且主观性强，pairwise 更稳定。
- **奖励模型是否需要与 SFT 模型同架构？** 不一定，但通常选择同架构以保证分布一致性，也可用更小的模型降低训练成本。
- **如何防止奖励模型过拟合？** 使用验证集监控，早停；增加正则化；使用更多的 prompt 多样性。


#### 2.3 RL 策略（PPO、DPO、GRPO）

##### PPO（Proximal Policy Optimization）

PPO 是在线 RL 算法，它将奖励模型作为环境，通过交互式采样来优化策略。

**核心流程**：
1. **采样**：从当前策略 $`\pi_\theta`$ 中采样一批 prompt 并生成回答。
2. **打分**：用奖励模型计算每个回答的奖励 $`r(x,y)`$。
3. **优势估计**：使用 GAE（Generalized Advantage Estimation）计算每个 token 的优势函数 $`A_t`$，通常需要一个 critic 模型（价值网络）来估计状态价值。
4. **策略更新**：先计算新旧策略对同一动作的概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

如果优势 $`A_t > 0`$，说明该动作值得提高概率；如果 $`A_t < 0`$，则应降低概率。PPO 不允许概率比 $`r_t(\theta)`$ 在一次更新中偏离 1 太远，因此使用裁剪后的策略目标：

$$
L_{clip}=\mathbb{E}_t\left[\min\left(r_tA_t,\ \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

`clip` 将有效更新限制在旧策略附近，避免少量高优势样本让策略一步走得过远。实际训练还要同时考虑三部分：

```text
总目标 = 策略目标 L_clip
       - 价值函数误差
       - β × 与参考策略的 KL
       + 熵奖励（可选）
```

其中 Critic 通过价值函数误差学习状态价值；KL 约束当前策略不要偏离参考策略（通常是 SFT 模型）太远；熵奖励用于避免策略过早变得确定。

5. **价值更新**：更新 Critic，改善后续 GAE 的优势估计。

**优点**：训练稳定，通过 KL 约束保留了 SFT 模型的生成能力；能充分利用奖励模型。
**缺点**：需要同时维护四个模型（actor, critic, reference, reward），显存占用大，实现复杂。


##### DPO（Direct Preference Optimization）

DPO 将 RLHF 转化为 **分类问题**，无需奖励模型和在线采样，直接使用偏好数据优化策略。

**核心思想**：从偏好数据中推导出一个隐式奖励函数，并通过最大似然直接优化策略。损失函数为：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

其中 $`\beta`$ 是控制 KL 惩罚强度的超参数，$`\pi_{\text{ref}}`$ 是固定的参考策略（通常为 SFT 模型）。

**优点**：
- 只需维护两个模型（策略和参考），无需 critic 和奖励模型，显存减半。
- 训练稳定，实现简单，效果在很多场景下与 PPO 相当甚至更好。
- 利用已有的偏好数据，无需在线采样，降低了工程复杂度。

**缺点**：
- 理论上 DPO 假设偏好数据符合 Bradley-Terry 模型，当实际偏好非此模型时可能有偏差。
- 由于不与环境交互，无法利用奖励模型进一步探索新回答，可能局限于训练数据中的偏好分布。


##### GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek 提出的 PPO 变体，旨在降低显存占用并简化训练流程。


**优势计算**（无 critic，用组内均值做基线）：

$$
A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

其中 $`G`$ 为每组采样回答数。

**策略损失**（带 clip）：

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|x)}{\pi_{\text{ref}}(o_i|x)} A_i,\ \text{clip}\left(\frac{\pi_\theta(o_i|x)}{\pi_{\text{ref}}(o_i|x)}, 1-\epsilon,1+\epsilon\right) A_i \right)
$$


**核心创新**：
- 对每个 prompt，采样一组回答（group），用组内的平均奖励作为基线（baseline），代替 critic 网络的价值估计。
- 优势函数定义为 $`A_i = r_i - \text{mean}(r_{\text{group}})`$，其中 $`r_i`$ 是第 $`i`$ 个回答的奖励。
- 策略更新仍使用 PPO 的 clip 目标，但无需单独的价值网络，从而节省显存。

**优点**：显存占用低于 PPO（少一个 critic 模型），且通过组内相对比较缓解了奖励尺度不一致的问题。
**缺点**：方法较新，社区验证较少；组大小选择需权衡（过小则基线不稳定，过大则增加采样成本）。

**常见问题**
- **PPO vs DPO 的核心区别**：PPO 是在线 RL，需要奖励模型和 critic，采样成本高但可探索；DPO 是离线优化，直接利用静态偏好数据，实现简单但依赖数据质量。
- **为什么 DPO 不需要奖励模型？** DPO 通过将偏好概率建模为策略与参考策略的比值，隐式构造了奖励函数，绕过了显式奖励模型训练。
- **GRPO 的创新在哪？** 用组内奖励作为基线，去掉了 critic，简化了训练流程，特别适合多机多卡的 MoE 模型（如 DeepSeek-V2 使用 GRPO 进行 RLHF）。

为什么 GRPO 能涌现推理能力

1. **组内相对比较**：模型必须在同 prompt 的多个回答中分辨好坏，迫使它学会区分推理步骤的优劣，而不是只追求绝对高分。
2. **无 critic 误差**：长链推理的价值很难准确估计，去掉 critic 避免了偏差，让策略更新更稳定。
3. **鼓励多样性**：组采样自然要求回答多样化（否则优势接近零），促使模型探索不同推理路径，发现更优解法。

结果：模型生成更长、更结构化的推理链，甚至出现自我校正行为。


### 3 推理优化

#### 3.1 KV Cache

**原理**
在自回归生成时，每一步需要计算当前 token 与之前所有 token 的注意力。如果每次都重新计算所有历史 token 的 K 和 V，会引入大量重复计算。KV Cache 将已生成的 token 的 K、V 缓存在内存中，新 token 只需计算自己的 K、V，然后与缓存中的 K、V 拼接后进行注意力计算。

**内存占用**
标准多头注意力下，整模型 KV Cache 的大小为：

```math
\text{内存} = 2 \times \text{num\_layers} \times \text{batch\_size} \times \text{num\_kv\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{sizeof(dtype)}
```

系数 2 对应 Key 和 Value。使用 GQA/MQA 时应代入 `num_kv_heads`，而不是 Query Head 数。

例如，Llama 2 7B 有 32 层、32 个 KV Head，`head_dim=128`。使用 FP16、`batch=1`、`seq_len=4096` 时，**单层** KV Cache 为：

$$
2 \times 1 \times 32 \times 4096 \times 128 \times 2\text{ bytes}
=64\text{ MiB}
$$

乘以 32 层后，整模型约为：

$$
32\times64\text{ MiB}=2048\text{ MiB}=2\text{ GiB}
$$

当上下文从 4k 增加到 32k 时，KV Cache 线性扩大 8 倍，单请求约占 16 GiB。再叠加 Batch 和并发序列后，KV Cache 很容易超过模型权重之外的剩余显存。

**优化技巧**
- **MQA（Multi-Query Attention）**：所有注意力头共享同一组 K、V，将 KV Cache 大小减少到原来的 $`1/\text{num\_heads}`$，但可能影响模型质量。
- **GQA（Grouped-Query Attention）**：将查询头分组，每组共享 K、V，在性能与内存之间取得平衡（如 Llama 2 7B 采用 GQA）。
- **KV Cache 量化**：将缓存的 K、V 量化为 INT8 或 FP8，可显著降低显存占用，但需要小心精度损失。

**常见问题**
- **KV Cache 为什么能加速？** 将每步的 $`O(S^2)`$ 复杂度降为 $`O(S)`$，但需要额外显存。
- **长上下文场景下 KV Cache 如何优化？** 可以通过 GQA、量化、分块处理（如 StreamingLLM 只保留部分缓存）来缓解。
- **如何实现 KV Cache？** 可以进一步用伪代码说明用伪代码描述生成过程中如何维护和更新缓存。


#### 3.2 Flash Attention

**原理**

Flash Attention 是一种 **IO-aware** 的精确注意力实现，核心思想是 **分块（tiling）** 和 **重计算**，以最小化 GPU 显存（HBM）与片上 SRAM 之间的数据移动。

标准注意力计算需要将 $`QK^T`$ 这个 $`S \times S`$ 矩阵写入 HBM，然后读取进行 softmax，再与 V 相乘。HBM 带宽远低于 SRAM，导致大量时间浪费在数据搬运上。

Flash Attention 将 Q、K、V 切分成小块（block），在 SRAM 中完成小块内的 softmax 和矩阵乘法，只将最终结果写回 HBM，避免了中间矩阵的读写。

**版本演进**
- **FlashAttention v1**：提出了分块 softmax 的数学技巧，使得在 SRAM 内可以安全合并不同块的 softmax 结果，实现了与标准 attention 完全等价的输出。
- **FlashAttention v2**：优化了并行策略（warp 级别的调度），减少了非矩阵乘的运算，速度比 v1 快 2-4 倍。
- **FlashAttention-3**（最新）：针对 H100 架构优化，利用异步指令和更高效的分块，进一步提升性能。

**适用场景**

Flash Attention 在长序列（如 8k、32k）时优势极为明显，短序列（如 512）提升有限。它同时支持训练和推理，且与 KV Cache 兼容（推理时仍可对缓存分块）。

**常见问题**
- **Flash Attention 如何减少内存访问？** 通过分块和在线 softmax，避免将 $`QK^T`$ 矩阵写入 HBM，大幅减少内存读写。
- **Flash Attention 与标准 attention 是否等价？** 是，输出数值上完全一致（忽略浮点误差）。
- **为什么 Flash Attention 对长序列特别有用？** 因为 $`QK^T`$ 的显存占用随 $`S^2`$ 增长，Flash Attention 避免了这一显存瓶颈，且能充分利用 SRAM 加速。



#### 3.3 INT8 量化

量化的目标是用更少的 bit 表示权重或激活，减少模型显存、显存带宽和在硬件支持下的矩阵乘开销。INT8 的优势首先来自数据量：相比 FP16，同样数量的权重理论存储减半。

##### 3.3.1 线性量化

对称 INT8 量化将实数 $x$ 映射到 $[-127,127]$：

$$
s=\frac{\max |x|}{127},\qquad
q=clip(round(x/s),-127,127),\qquad
\hat{x}=s\cdot q
$$

非对称量化额外使用 zero point：

$$
q=clip(round(x/s)+z,q_{min},q_{max}),\qquad
\hat{x}=s(q-z)
$$

对称量化计算更简单，常用于权重；非对称量化能更好覆盖不对称分布，但需处理 zero point。

以权重矩阵按输出通道量化为例，每一行独立计算 scale，避免一个异常通道拉低其他通道的有效精度：

```python
def quantize_per_channel(weight):          # [out_features, in_features]
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    qweight = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return qweight, scale

def dequantize(qweight, scale):
    return qweight.float() * scale
```

这段代码用于解释量化关系，不代表高性能 Kernel：真实推理需要在算子内部完成反量化或直接执行 INT8 GEMM，否则显式还原整块 FP 权重可能省显存却不省延迟。

##### 3.3.2 Per-tensor、Per-channel 与 Per-token

- **Per-tensor**：整个张量共享一个 scale，实现简单，但易被少数离群值拉大范围；
- **Per-channel**：权重每个输出通道使用独立 scale，通常能明显降低权重量化误差；
- **Per-token**：激活每个 Token 动态计算 scale，适应不同 Token 的幅值，但引入动态统计开销。

量化粒度越细，误差通常越小，但 scale 存储、Kernel 实现和调度更复杂。

##### 3.3.3 Weight-only 与 W8A8

**Weight-only INT8（W8A16）**：权重以 INT8 存储，计算前或 Kernel 内反量化到 FP16/BF16，激活保持高精度。它主要降低权重显存和读取带宽，精度风险较低，但若 Kernel 只是先反量化再做 FP16 GEMM，计算加速可能有限。

**W8A8**：权重和激活都使用 INT8，可直接利用 INT8 Tensor Core，但激活中的 outlier 会让量化更困难。因此 W8A8 的速度潜力更高，对校准数据和 Kernel 支持的要求也更高。

##### 3.3.4 Outlier 与 SmoothQuant

假设激活大部分位于 $[-1,1]$，却有少数值达到 50。如果整个 Tensor 用同一 scale，量化范围必须覆盖 50，$[-1,1]$ 中的大量数值就会挤在少数刻度中，丢失精度。

SmoothQuant 利用线性层 $Y=XW$ 的等价变换，将激活中难量化的幅值部分迁移到权重：

$$
Y=(X\,diag(s)^{-1})(diag(s)W)
$$

对激活做平滑后，激活更容易量化，权重通常比激活更能承受这部分尺度变化。平滑强度需在两侧误差之间取舍。

##### 3.3.5 PTQ、Calibration 与 QAT

**PTQ** 在训练后根据权重和少量校准数据确定 scale/clipping，成本低，是部署中的常用选择。Calibration 数据不需要很大，但必须覆盖真实任务的长度、语言、模态和激活分布；只用随机短文本会导致线上 outlier 范围估计失真。

**QAT** 在训练中插入 fake quantization，前向模拟 round/clip，反向通常使用 Straight-Through Estimator 近似梯度。它能让模型适应量化误差，但训练成本高、工程复杂。

##### 3.3.6 哪些部分需要保留高精度

不是所有算子都适合 INT8。常见做法是将 LayerNorm/RMSNorm、Softmax、采样与部分敏感层保留为 FP16/BF16，将主要 Linear/GEMM 量化。敏感层可通过逐层误差、消融实验或 Hessian/激活统计识别。

KV Cache 精度与权重量化互相独立。将权重改为 INT8 不代表 KV Cache 也自动变成 INT8，因此长上下文、高并发场景仍可能被 KV Cache 显存限制。

##### 3.3.7 量化后的端到端验证

量化效果不能只用权重文件大小衡量，应在同一硬件、并发、上下文与生成长度下比较：

- 任务 Accuracy/F1、困惑度或生成质量；
- JSON Schema/Tool Call 合法率与长序列稳定性；
- 峰值 GPU 显存，并区分权重、KV Cache 和 Runtime Workspace；
- TTFT、TPOT、P95/P99 与 output tokens/s；
- 不同长度、语言、类别和多模态数据分层指标。

显存减少也不必然等于延迟下降。如果硬件没有高效 INT8 Kernel，或频繁 Quantize/Dequantize，低精度可能只省显存而不加速。

**常见问题**

1. **W8A16 和 W8A8 的核心差别？**
   W8A16 主要压缩权重存储和带宽；W8A8 还量化激活，可使用 INT8 GEMM，但对 outlier 和校准更敏感。
2. **为什么 INT8 显存不一定正好是 FP16 的 50%？**
   只有被量化的权重接近减半，scale、高精度层、KV Cache、CUDA Workspace 和显存碎片都不按同一比例缩放。
3. **为什么量化后显存降了，速度却可能没提升？**
   运行时可能缺少原生 INT8 Kernel，或反量化、数据转换和其他非 GEMM 算子成为新瓶颈。

#### 3.4 分布式训练（TP、PP、DP）

##### 数据并行（DP）
- **原理**：每张 GPU 持有完整的模型副本，处理不同的数据分片。前向和反向独立计算，梯度通过 AllReduce 同步，确保所有副本参数一致。
- **通信**：每步需同步梯度（通常使用 AllReduce），通信量与模型参数量成正比。
- **优点**：实现简单，适合单机多卡。
- **缺点**：模型必须能完整放入单卡显存；当模型过大时，无法使用。

##### 张量并行（TP）
- **原理**：将单个层内的参数切分到多张 GPU 上，每张卡只存储部分权重。例如 MLP 层：将第一个线性层的权重按列切分，第二个线性层按行切分，使前向时只需一次 AllReduce 即可合并结果。Attention 层：将多头切分到不同 GPU（每个卡负责部分头）。
- **通信**：每层前向后向各进行一次 AllReduce，通信量相对密集。
- **优点**：可以训练单卡放不下的超大模型（如 100B+）。
- **缺点**：需要精细设计切分方案，通信开销较大。

##### 流水线并行（PP）
- **原理**：将模型按层切分成多个阶段（stage），每个阶段放在不同 GPU 上。输入数据切分成微批次（micro-batch），以流水线方式执行，减少空闲时间。
- **通信**：仅在相邻阶段间传递激活值和梯度，通信量远小于 TP。
- **优点**：适合非常深的模型，通信开销低。
- **缺点**：存在流水线气泡（pipeline bubble），即某些 GPU 空闲等待的时间；需要处理微批次调度。

**组合使用（3D 并行）**
在实际训练超大模型时，通常将三种并行策略结合：
- 在单机内使用 TP（机内 NVLink 带宽高）。
- 跨机使用 PP（机间通信相对较慢，但 PP 通信量小）。
- 同时使用 DP（数据并行）来扩大 batch size，充分利用数据并行度。

**ZeRO（Zero Redundancy Optimizer）**
ZeRO 是对数据并行的改进，它将优化器状态、梯度、参数分片到各 GPU，实现 **内存零冗余**。ZeRO 分为三个阶段：
- **ZeRO-1**：分片优化器状态。
- **ZeRO-2**：分片优化器状态和梯度。
- **ZeRO-3**：分片优化器状态、梯度和参数（需要时从其他 GPU 收集）。

ZeRO 允许在数据并行的框架下训练比单卡显存大得多的模型，且通信效率高，是当前大模型训练的主流方案。

**常见问题**
- **TP 与 PP 的区别**：TP 是层内切分，通信频繁但单步延迟低；PP 是层间切分，通信少但存在流水线气泡。
- **如何选择并行策略？** 根据模型大小、集群拓扑、显存限制综合决定。例如，超大模型通常用 TP+PP+DP 三维并行。
- **ZeRO 相比 DP 的优势**：ZeRO 消除了冗余，使显存占用随 GPU 数量线性扩展，可训练远超单卡容量的模型。
- **MoE 与专家并行**：MoE 天然适合专家并行（Expert Parallelism），将不同专家分布到不同 GPU，通过 All-to-All 通信实现路由，本质上是 TP 的一种特例。

#### 3.5 混合精度训练（BF16 vs FP16）

| 特性 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 动态范围 | ~5.96e-8 ～ 65504 | ~1.2e-38 ～ 3.4e38（同 FP32） |
| 溢出风险 | 高（需损失缩放） | 低（无需缩放） |
| 精度 | 较高 | 较低 |

**损失缩放（FP16 必需）**：
反向传播前将 loss 乘以 $`S`$（如 128），梯度更新后除以 $`S`$。

**代码（PyTorch）**：
```python
scaler = torch.cuda.amp.GradScaler()  # 仅 FP16
with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 或 torch.float16
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**意义**
- 训练速度提升 2~3 倍（Tensor Core 加速）。
- 显存占用减半。
- BF16 消除了梯度下溢/上溢问题，训练更稳定。

**常见问题**
1. **为什么大模型训练更倾向 BF16？**
   → 大模型梯度范围大，FP16 易溢出；BF16 动态范围与 FP32 相同，无需损失缩放。
2. **BF16 精度低会不会影响收敛？**
   → 通常不会，因为梯度更新本身具有噪声，且关键操作（如 softmax）仍可用 FP32。
3. **混合精度为什么能加速？**
   → 低精度矩阵乘法利用 Tensor Core，同时减少显存带宽占用。


#### 3.6 在线推理请求全链路

先用一个请求理解整条链路。用户发送「阅读这份 20k Token 的文档并总结」，并要求流式返回：

```text
1. 网关验证身份、计算 Token、检查配额
2. Router 选择一个健康且负载合适的 vLLM 副本
3. Scheduler 将请求排入队列，为它预留 KV Cache Block
4. Prefill 并行读完 20k Token，生成首个 Token
5. Decode 逐个生成后续 Token，网关通过 SSE 边生成边返回
6. 用户点击停止，cancellation 传到 vLLM，序列退出调度
7. KV Block 引用计数减少，可回收块返回 Block Pool
```

这条链路有两类调度：网关在多个推理副本之间做**粗粒度调度**，vLLM 在一个副本内对多条序列做**Token 粒度调度**。网关解决「请求送到哪里」，推理引擎解决「GPU 这一轮算哪些 Token」。

##### 3.6.1 先记住四个性能指标

| 指标 | 用户感受 | 主要受什么影响 |
|---|---|---|
| TTFT | 发出请求后，多久看到第一个字 | 排队 + Prefill + Prefix Cache |
| TPOT | 第一个字出现后，后续生成是否流畅 | Decode batch + 显存带宽 + 跨卡通信 |
| E2E Latency | 整个回答何时完成 | TTFT + 输出长度 × TPOT |
| Throughput | 集群每秒完成多少请求/Token | Batch、调度、GPU 利用率 |

例如一个请求 TTFT 为 1 s，之后生成 200 Token，TPOT 为 30 ms，那么用户大约在 1 s 后看到首字，完整等待时间约为 $1 + 200 \times 0.03 = 7$ s。优化 TTFT 与优化 TPOT 解决的是两种不同的体感问题。

##### 3.6.2 Prefill 与 Decode

可以把自回归生成想成「先读题，再逐字作答」：

- **Prefill**：一次处理全部输入 Token，计算量大且并行度高，通常更偏计算密集；
- **Decode**：每次生成一个新 Token，需要反复读取历史 KV Cache，通常更偏显存带宽密集。

Prefill 像一次性读完题目：所有 Prompt Token 可以并行计算，GPU 的矩阵乘法单元很忙。Decode 像逐字写答案：第 $t$ 步必须等第 $t-1$ 步产生的 Token，每次计算量不大，但都要从显存读取前面全部 Token 的 KV，因此容易受显存带宽限制。

假设 Prompt 长 10k Token，要生成 100 Token：

- 没有 KV Cache：第 1 步重算 10k Token，第 2 步重算 10001 Token，之后每步都重算整段历史；
- 有 KV Cache：Prefill 时将 10k Token 的 K/V 存下，Decode 每步只计算新 Token 的 Q/K/V，再与历史 K/V 做 Attention。

KV Cache 避免在每个 Decode 步骤重复计算历史 Token 的 Key 和 Value，但显存占用会随层数、KV Head 数、Head Dimension、序列长度、Batch 和数据类型线性增长。粗略估算为：

$$
M_{KV} \approx 2 \times L \times B \times S \times H_{kv} \times D_h \times bytes
$$

其中系数 2 分别对应 Key 和 Value。GQA/MQA 通过减少 KV Head 数降低缓存占用和 Decode 带宽压力。

**为什么 KV Cache 往往比想象中更贵？** 权重在所有请求之间共享，KV Cache 却基本是每条序列独有，并且随并发数和上下文长度线性增长。因此「模型权重能放进 GPU」不等于「能服务目标并发」。

##### 3.6.3 Block-based KV Cache

如果每个请求一来就按 `max_model_len` 预留一大段连续显存，就像酒店为每位客人都预留一整层楼：客人只住一晚，剩余房间也不能给别人用。Block-based KV Cache 改成按固定大小的「房间」分配，序列变长时再取新块，结束后归还。

物理块在显存中不需要连续，每条序列通过 Block Table 把自己的第 0、1、2 个逻辑块映射到任意可用的物理块。这与操作系统虚拟内存的分页思想类似，因此 PagedAttention 的核心价值不是改变 Attention 数学，而是让 KV Cache 能灵活分配、共享和回收。

核心对象包括：

```text
Block Pool      管理可用物理块
Block Table     记录序列的逻辑块映射
Reference Count 支持多序列共享缓存块
LRU Metadata    用于缓存复用与淘汰
```

块越小，尾块浪费越少，但块表、调度和 Kernel 寻址开销更高；块越大则相反，因此需要根据序列长度分布和并发负载选择。

**例子**：若每块存 16 Token，一条 35 Token 的序列占用 3 块，最后一块只用 3 个位置，浪费 13 个位置；但它不会因为之后可能生成到 4k Token，就提前占住 4k Token 的空间。

##### 3.6.4 Public Prefix Reuse

多个请求可能共享相同的 System Prompt、工具定义或公共文档前缀。对已计算的 Token Block 建立内容哈希，新请求按块匹配最长公共前缀，命中后增加引用计数并复用对应 KV Block，只对未命中后缀做 Prefill。

例如 100 个 Agent 请求都携带相同的 8k Token System Prompt 和 Tool Schema，只有末尾的 User Message 不同。没有 Prefix Cache 时，这 8k Token 要做 100 次 Prefill；命中同一副本的 Prefix Cache 后，后续请求可直接引用前缀 KV，只计算用户动态部分。

因此 Prefix Cache 是一种**避免重复 Prefill 计算**的机制，它不会让单条完全新的 Prompt 凭空变快，也不会减少后续 Decode 需要读取的 KV 量。

复用的正确性要求 Token IDs、模型与版本、位置编码相关配置及影响 KV 的推理参数一致。还要防止跨租户错误共享带来的数据泄露和时序侧信道。

**Redis 在 Prefix Cache 中存什么？** 通常不是把体积巨大的 KV Tensor 写入 Redis。KV Block 仍放在 vLLM 实例的 GPU/CPU Cache 中；Redis 更适合保存跨网关共享的**前缀索引与实例亲和性元数据**：

```text
prefix:{model_version}:{tenant}:{block_hash}
  → instance_id / block_count / last_seen / expire_at
```

网关先对稳定前缀的 Token IDs 分块计算 Hash，再查询哪些健康实例已经缓存该前缀，优先将请求路由过去：

```python
prefix_key = hash(model_version, tenant_id, stable_prefix_token_ids)
cached_instances = redis.smembers(f"prefix:{prefix_key}")
backend = router.pick_healthy(cached_instances) or router.pick_least_loaded()
```

Redis 解决的是“哪个副本可能命中”，真正是否命中仍由目标 vLLM 实例校验。元数据需要较短 TTL，并在实例摘流、模型重载或 Cache 淘汰后失效；否则只会产生一次错误亲和路由，不应影响推理正确性。

##### 3.6.5 LRU 淘汰与调度配合

当空闲块不足时，优先淘汰引用计数为 0 且最久未使用的 Prefix Cache Block。正在被活跃序列引用的块不可淘汰。实现上需要保证「查找、增加引用、释放、淘汰」的并发一致性，并在压力较高时让 Admission Control 与 Scheduler 共同决定是等待、抢占还是拒绝新请求。

评估时不仅要看吞吐，还要同时观察 TTFT、TPOT、P50/P95 延迟、KV Cache 利用率、Prefix Cache 命中率、预占/重算次数和 OOM 率。

#### 3.7 统一 LLM Serving 网关

网关是所有模型服务的统一入口。它不执行模型计算，而是将「调用哪个模型、请求能否进入、送到哪个实例、失败后如何处理」变成确定性工程规则。

```text
Client
  → 协议适配
  → 鉴权与配额
  → Token 级限流
  → 模型与实例路由
  → vLLM / 其他推理后端
  → SSE 流式返回
```

##### 3.7.1 模型适配

对上层提供统一的 Chat/Completions/Embeddings 协议，对下层适配不同 Provider 或推理引擎。适配不只是改 URL，还包括：

- messages、Tool Schema、多模态输入和 sampling parameters 的转换；
- Tool Call、finish reason、usage 和 error code 的归一化；
- SSE chunk、heartbeat、`[DONE]` 与取消语义的统一；
- model capability 描述：上下文长度、Tool Calling、JSON Schema、多模态和量化版本。

上层只依赖网关契约，替换底层模型或 vLLM 版本时不需要修改业务代码。

##### 3.7.2 鉴权、路由与限流

**鉴权**不只校验身份，还要将身份映射到租户、允许模型、RPM/TPM 配额、最大上下文和审计策略。

**路由**分两步：先根据模型名和 capability 选择模型池，再根据实例的活跃序列、排队 Token、KV Cache 水位、近期 TTFT 和 Prefix Cache 亲和性选择副本。只用 round-robin 会把 100 Token 和 100k Token 的请求当成同等负载。

**限流**不能只看 QPS，通常同时约束：

```text
RPM: 每分钟请求数
TPM: 每分钟输入/输出 Token
Concurrency: 租户活跃序列数
Queue Budget: 排队 Token 和最大等待时间
Request Guardrail: max_model_len、max_tokens、Tool Schema 大小
```

网关在入队前用对应 tokenizer 统计输入 Token，并用申请的最大输出长度估计资源上界。已经无法在 SLO 内处理时尽早返回 429/503，比让请求在队列中长时间等待更可控。

多网关副本下，限流计数不能只放在单机内存，否则同一租户可以把流量分散到不同网关绕过配额。Redis 常通过 Lua Script 原子地完成“读取水位、补充令牌、扣减令牌、设置 TTL”，实现共享的 Token Bucket：

```text
rate:{tenant}:{model}:rpm     请求令牌桶
rate:{tenant}:{model}:tpm     Token 令牌桶
active:{tenant}:{model}       当前并发数（带租约/过期时间）
```

一次请求应同时申请 RPM、预估 TPM 和并发额度；任意一项不足就整体拒绝或回滚。仅用 `INCR` 再 `EXPIRE` 的多个命令会有并发窗口，Lua 或事务的意义是让检查与扣减成为一个原子操作。

##### 3.7.3 超时、重试与幂等

一个流式请求需要分开三种超时：

- **Connection Timeout**：无法建立后端连接；
- **TTFT Timeout**：连接已成功，但首 Token 长时间未返回；
- **Inter-token Timeout**：流已开始，但相邻 Token 之间长时间无输出。

不能给整个 SSE 请求设一个简单的短超时，否则长回答会被误杀。重试只在「操作幂等 + 总预算未耗尽」时发生，采用指数退避和 jitter，并遵守 `Retry-After`。

首 Token 返回前，纯生成请求可重试到其他实例；首 Token 已返回后，新实例没有原请求的 KV Cache，也不保证重新生成与已输出内容一致，通常只能终止流并返回显式错误。

网关实现的关键是把“能否重试”绑定到流状态和总时间预算，而不是只根据异常类型判断：

```python
async def stream_with_failover(request, deadline):
    emitted = False
    for backend in router.healthy_candidates(request.model):
        try:
            async for chunk in backend.stream(request, deadline=deadline):
                emitted = True
                yield chunk
            return
        except RetryableError:
            if emitted or time.monotonic() >= deadline:
                raise StreamInterrupted()
            circuit_breaker.record_failure(backend)
    raise ServiceUnavailable()
```

实际系统还要传递 cancellation、限制最大尝试次数并使用 request ID 做审计；核心边界始终是：一旦已向客户端输出 Token，就不能静默换实例重新生成。

##### 3.7.4 熔断、降级、健康检查与异常切流

这四个机制是一条联动链路，不需要单独构造一套「高可用架构」：

```text
健康检查发现实例异常
  → Router 停止向该实例发送新请求
  → Circuit Breaker 打开，防止重试风暴
  → 未开始流式输出的请求切到健康实例
  → 容量不足时按能力契约降级到备用模型
  → 故障实例恢复后半开探测，逐步恢复流量
```

**健康检查**分为：

- liveness：进程和基础通信是否存活；
- readiness：权重是否加载、GPU 是否健康、队列/KV Cache 是否超高水位，实例能否接新流量；
- passive health：根据真实请求的超时、502、GPU OOM 和断流率判断。

为了避免一次网络抖动就误切流，需要连续失败阈值、滑动时间窗口、最小样本数和恢复滞回。「30 秒内切流」本质上由以下时间组成：

```text
T_failover
  = T_detect       故障被主动/被动检测发现
  + T_decide       达到阈值并打开熔断器
  + T_propagate    健康状态传播到各网关副本
  + T_reroute      新请求选择健康实例
```

要稳定控制在 30 秒内，不能只把 health-check interval 设为 30 秒，而是要对检测、判定、传播和重路由分别设定预算并演练。

**熔断**保护下游和网关自身；**降级**则在容量或模型不可用时保留有限服务。备用模型必须事先声明 capability，如上下文长度、Tool Calling、JSON Schema 和多模态能力；否则「有响应」不等于「服务可用」。

Redis 也可以保存网关副本间共享的熔断状态，例如：

```text
circuit:{model}:{instance}
  → state=open, failures=12, opened_at=..., probe_owner=...
```

每个网关先在本地快速判断，失败统计和 `open/half-open` 状态再同步到 Redis，避免某个网关已经熔断、其他网关仍持续把请求打向故障实例。半开阶段可用 `SET key value NX EX ...` 抢占少量探测权，防止所有网关同时探测造成流量尖峰。

Redis 本身也可能故障，所以它不应成为推理链路的单点：限流可按安全策略选择短时本地保守额度或 fail-closed；熔断应继续依赖本地状态和健康检查；Prefix 元数据不可用时则退化为普通负载路由，只损失命中率，不影响请求正确性。

##### 3.7.5 99.99% 可用性如何定义

99.99% 意味着一年理论不可用时间约为 52.56 分钟。但可用性数字只有在 SLI 统计口径明确后才有意义。

一个实用定义是：

$$
Availability = \frac{Valid\ Requests - Gateway\ Attributed\ Failures}{Valid\ Requests}
$$

`Gateway Attributed Failures` 可包括网关 5xx、超时、异常断流和路由失败；不应把客户端取消、非法参数和用户配额超限一律算成网关不可用。流式请求也不能在返回 HTTP 200 后就算成功，还要检查是否正常生成首 Token 并完成或有明确 finish reason。

高可用的核心不是「永不失败」，而是失败能被快速发现、不被重试放大、能切到健康实例，并且整个过程可观测、可演练。

**常见问题**

1. **为什么不能把所有超时都重试？**
   超时时后端可能仍在计算，盲目重试会放大 GPU 压力；流已开始时还会导致内容无法连续。
2. **熔断和限流的区别是什么？**
   限流是在请求进入前保护容量；熔断是在下游已经异常时停止继续尝试，防止故障扩散。
3. **30 秒切流如何验证？**
   主动注入进程崩溃、GPU OOM、网络不通和高延迟等故障，用 Trace 分别记录 detect、decide、propagate 和 reroute 时刻，对 P95/P99 而不只是平均值验收。

#### 3.8 vLLM 核心架构与推理机制

vLLM 的核心价值是把「单个模型的前向计算」变成「多请求共享 GPU 的高吞吐推理系统」。它不只是一个 HTTP Server，而是由请求管理、调度、KV Cache 管理和 Model Executor 组成的推理引擎。

```text
API Server
  → Input Processor / Tokenizer
  → Engine Core
       ├─ Scheduler
       ├─ KV Cache Manager / Block Pool
       └─ Model Executor
            ├─ Attention Backend
            ├─ Tensor/Pipeline Parallel Workers
            └─ Sampling / Structured Output
  → Streaming Output
```

不同 vLLM 版本的类名与进程组织会变化，但「Scheduler 决定本轮算什么、KV Manager 决定缓存放哪里、Executor 执行模型前向」的逻辑分层不变。

##### 3.8.1 Continuous Batching

静态 Batching 像「一车人必须一起上车、一起到终点」：只要有一条序列还没生成完，已经结束的位置也无法及时给新请求。

Continuous Batching 在每个调度轮次重新组织 batch：

```text
step 1: [A, B, C]
step 2: A 结束 → [B, C, D]
step 3: C 结束 → [B, D, E]
```

完成的序列立即离开，新请求可在下一轮加入，避免 GPU 等待最长序列。调度单位本质上是 Token 预算，Scheduler 在 Prefill 和 Decode 之间分配当轮算力。

它提高的是吞吐和 GPU 利用率，但 batch 过大时单请求 TPOT 与尾延迟会上升，因此仍是吞吐与延迟的取舍。

Continuous Batching 可以抽象为一个逐轮重排的调度循环：

```python
while waiting or running:
    budget = max_batched_tokens
    batch = select_decode_requests(running, budget)   # 先保护在线 TPOT
    budget -= token_cost(batch)
    batch += select_prefill_chunks(waiting, budget)  # 剩余预算接纳新请求

    outputs = model_executor.step(batch)
    running, finished = update_sequences(running, outputs)
    release_blocks(finished)
```

真实 vLLM 的实现远比这段伪代码复杂，但核心决策已经显现：每轮在 Decode、Prefill 和 KV Block 之间分配预算；完成序列及时退出，新请求不必等待旧 batch 全部结束。

##### 3.8.2 PagedAttention

PagedAttention 不是一种新的 Attention 数学公式，而是 KV Cache 的分页存储与访问方法。它将每条序列的逻辑 KV Block 映射到不连续的物理 Block，Attention Kernel 通过 Block Table 找到真实地址。

价值主要有三点：

- 不按最大序列长度连续预留，降低碎片和尾部浪费；
- 序列边生成边按需申请 Block，结束后立即回收；
- 通过引用计数共享 Prefix Block，支持 Prefix Cache 和分支序列。

##### 3.8.3 FlashAttention 与 PagedAttention 的区别

两者经常被混淆，但解决的问题不同：

| 机制 | 解决的问题 | 核心思想 |
|---|---|---|
| FlashAttention | Attention 计算中 $N\times N$ 中间矩阵读写 HBM 开销大 | Tiling + Online Softmax，在 SRAM 中分块计算 |
| PagedAttention | 多条变长序列的 KV Cache 难以高效分配 | 逻辑块—物理块映射 |

FlashAttention 主要优化「Attention 怎么算」，PagedAttention 主要优化「KV Cache 怎么存和怎么找」。Prefill 阶段通常更能从 FlashAttention 的 IO 优化中受益；Decode 阶段 query 通常只有一个或少量 Token，瓶颈更常是读取历史 KV 的显存带宽。

##### 3.8.4 Prefix Caching

vLLM 将已计算的完整 KV Block 按 Token 内容及相关配置建立哈希。新请求命中最长前缀后，只对未命中的后缀做 Prefill。

它只节省重复 Prefill，不会减少 Decode 需要读取的历史 KV。因此适合 System Prompt、Tool Schema、Few-shot Example 和公共文档前缀高度重复的场景。如果前缀中带有时间戳、随机 ID 或字段顺序不稳定，Token 序列不同就会直接 miss。

##### 3.8.5 Speculative Decoding

自回归 Decode 每次只产生一个 Token，存在强串行依赖。Speculative Decoding 用更快的 draft model 一次猜测多个 Token，target model 再一次并行验证这些候选：

```text
Draft:  猜 [t1, t2, t3, t4]
Target: 一次前向验证
        ├─ 接受 t1, t2, t3，t4 被拒绝
        └─ 从拒绝位置继续生成
```

在正确的接受—拒绝算法下，最终输出分布与直接使用 target model 一致。加速来自用一次 target forward 接受多个 Token，但只在 draft 足够快且接受率较高时有利。如果 draft 猜得差，验证开销会抵消收益。

决定收益的关键指标是 acceptance rate、平均每轮接受 Token 数、draft/target 速度比以及额外 KV Cache 开销。

##### 3.8.6 Chunked Prefill 与 Prefill/Decode 混部

一个超长 Prefill 如果在一轮中独占 GPU，正在 Decode 的用户会长时间收不到下一个 Token。Chunked Prefill 将长 Prompt 分成多个 chunk，Scheduler 可以在两个 chunk 之间插入 Decode：

```text
大 Prefill: [chunk 1] [chunk 2] [chunk 3]
Decode:                [d]       [d]       [d]
```

这能减少长 Prompt 对 inter-token latency 的干扰，但会增加调度和中间状态管理开销。核心取舍仍然是 Prefill TTFT、Decode TPOT 与整体吞吐。

##### 3.8.7 抢占、Swap 与 Recomputation

当 KV Block 不足时，Scheduler 可能需要抢占低优先级或后进入的序列。被抢占序列有两种恢复方式：

- **Swap**：将 KV Cache 移到 CPU，恢复时再搬回 GPU，消耗 PCIe 带宽；
- **Recomputation**：丢弃 KV，恢复时根据已有 Token 重做 Prefill，消耗 GPU 计算。

频繁 preemption 通常说明并发、最大上下文或 Token Budget 超出 KV Cache 能力。它不是免费的容量扩展，而是以延迟换取不立即 OOM。

##### 3.8.8 并行推理

- **Tensor Parallel**：将每层权重和 Attention Head 切到多张 GPU，每层需要 AllReduce/AllGather，适合高带宽机内互联；
- **Pipeline Parallel**：将不同层放到不同 GPU/节点，通信发生在 Stage 之间，但存在流水线气泡；
- **Data Parallel / Replica**：每个副本持有完整模型，网关将不同请求路由到不同副本，扩展集群总吞吐。

TP 并非越大越快。它减少每张卡的权重和计算，却引入每层集合通信。当模型已能放入单卡或卡间带宽不足时，增大 TP 可能反而变慢。

##### 3.8.9 常见性能问题

| 现象 | 首先检查 | 常见原因 |
|---|---|---|
| TTFT 高 | queue time、prefill time、cached tokens | 排队、长 Prompt、Prefix Miss |
| TPOT 高 | running sequences、memory bandwidth、NCCL | Decode batch 大、显存带宽或 TP 通信 |
| GPU 利用率低 | scheduler queue、tokenizer CPU、网关 | 请求喂不满或 CPU/网络瓶颈 |
| preemption 多 | KV usage、running/waiting seqs | 并发与长序列挤占 KV Cache |
| Prefix Cache 命中低 | Token 前缀、hash、副本路由 | 动态字段、Schema 顺序不稳定或路由分散 |

**常见问题**

1. **Continuous Batching 与普通 Dynamic Batching 有什么差别？**
   Dynamic Batching 通常在请求开始前等待组 batch；Continuous Batching 在生成的每个调度轮次都可以移除已完成序列、加入新序列。
2. **FlashAttention 和 PagedAttention 是否互相替代？**
   不是。前者减少 Attention 计算的 HBM IO，后者管理变长序列的 KV Cache，两者可以同时存在。
3. **Speculative Decoding 为什么能保持 target model 的输出分布？**
   draft 只负责提议 Token，target 通过接受—拒绝规则校正候选；正确算法下不是直接相信 draft 输出。
4. **为什么权重量化后仍可能 KV Cache OOM？**
   权重和 KV Cache 是两块独立显存。量化权重释放了模型显存，但 KV 精度、并发数和序列长度不变时，KV 占用仍然会线性增长。
