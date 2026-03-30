# 后训练

## 1 SFT（监督微调）

### 1.1 SFT 数据标注格式

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

**面试考察点**  
- 如何构造高质量 SFT 数据？（数据来源、清洗、多样性控制）
- 模板错误会带来什么影响？（模型可能输出空内容、格式错乱或拒绝回答）
- 多轮对话在数据中如何组织？（需要保持上下文连贯，不能只保留最后一轮）


### 1.2 LoRA 与其他 PEFT 方法的区别

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

**面试考察点**  
- **LoRA 为什么有效？** 微调过程中，权重的变化往往位于一个低秩子空间，这已被实验和理论验证（如 Aghajanyan et al. 2021）。
- **LoRA 在 MoE 中如何应用？** 可以只对每个专家的 FFN 权重加 LoRA，也可以对 Attention 层加。通常为每个专家独立添加低秩矩阵，可训练参数仍然远少于全量微调。
- **PEFT 方法如何选择？** 如果追求推理速度且显存充足，LoRA 是首选；如果模型极大且希望参数极少，Prompt Tuning 可以一试；如果必须用旧版模型且不想改动原始权重，Adapter 也是常见选择。
- **全量微调 vs PEFT：** 当有大量高质量数据且需要最大化下游性能时，全量微调仍然是最佳选择，但显存和训练时间要求高。


### 1.3 知识蒸馏

知识蒸馏是一种模型压缩技术，用一个大模型（教师） 的输出来指导小模型（学生） 的训练，使得学生模型在参数量更小的情况下尽量逼近教师模型的性能。

教师模型输出 logits $`z_t`$，学生模型输出 $`z_s`$。蒸馏损失为：

$$
\mathcal{L}_{\text{distill}} = (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(y_s, y_{\text{true}}) + \alpha \cdot \tau^2 \cdot \text{KL}(p_s^\tau \| p_t^\tau)
$$

其中：

$$
p_s^\tau = \text{softmax}(z_s / \tau), \quad p_t^\tau = \text{softmax}(z_t / \tau)
$$

- $`\tau`$：温度系数，$`\tau>1`$ 使概率分布更平滑。
- $`\tau^2`$ 用于平衡梯度量级（因为带温度的 softmax 梯度与 $`1/\tau^2`$ 相关）。

**代码示意**：
```python
def distillation_loss(student_logits, teacher_logits, labels, temp=4.0, alpha=0.7):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1)
    ) * (temp ** 2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

意义：
- 学生模型学习教师输出的“暗知识”（类间相似性）。
- 可在大幅减少参数量的同时保持较高性能。

面试问题与回答要点：
1. **温度 $`\tau`$ 的作用？**  
   → 软化概率分布，放大负类的信息，让学生学到更丰富的相对关系。
2. **为什么蒸馏比单纯用真实标签训练小模型更好？**  
   → 教师提供的软标签含有类间结构（例如猫和狗的相似度高于猫和汽车），学生可以借此泛化更好。
3. **蒸馏与量化的区别？**  
   → 蒸馏改变模型结构/参数量，量化改变数值精度，两者可叠加。


## 2 RLHF（基于人类反馈的强化学习）

### 2.1 RLHF 数据标注格式

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

**面试考察点**  
- **为什么用排序而不是打分？** 不同人对同一回答的绝对分数差异很大（有人给 3 分，有人给 5 分），但排序的一致性更高。排序还能减少标注员疲劳，提升数据质量。
- **如何保证标注一致性？** 通过标注员培训、多人标注并取多数、设计清晰的标注指南。
- **数据规模要求**：奖励模型的性能与数据量正相关，但存在边际收益递减，通常 10 万条比较数据已能取得不错效果。


### 2.2 奖励模型训练

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

**面试考察点**  
- **为什么不用均方误差（MSE）预测绝对分数？** 因为绝对分数难以标注且主观性强，pairwise 更稳定。
- **奖励模型是否需要与 SFT 模型同架构？** 不一定，但通常选择同架构以保证分布一致性，也可用更小的模型降低训练成本。
- **如何防止奖励模型过拟合？** 使用验证集监控，早停；增加正则化；使用更多的 prompt 多样性。


### 2.3 RL 策略（PPO、DPO、GRPO）

#### PPO（Proximal Policy Optimization）

PPO 是在线 RL 算法，它将奖励模型作为环境，通过交互式采样来优化策略。

**核心流程**：
1. **采样**：从当前策略 $`\pi_\theta`$ 中采样一批 prompt 并生成回答。
2. **打分**：用奖励模型计算每个回答的奖励 $`r(x,y)`$。
3. **优势估计**：使用 GAE（Generalized Advantage Estimation）计算每个 token 的优势函数 $`A_t`$，通常需要一个 critic 模型（价值网络）来估计状态价值。
4. **策略更新**：最大化目标函数  

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E} \left[ \min\left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A, \ \text{clip}\left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1-\epsilon, 1+\epsilon \right) A \right) \right]
$$
   
   同时加入 KL 散度惩罚项，防止策略偏离参考策略（通常是 SFT 模型）太远。
5. **价值更新**：更新 critic 网络，减小价值估计误差。

**优点**：训练稳定，通过 KL 约束保留了 SFT 模型的生成能力；能充分利用奖励模型。  
**缺点**：需要同时维护四个模型（actor, critic, reference, reward），显存占用大，实现复杂。


#### DPO（Direct Preference Optimization）

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


#### GRPO（Group Relative Policy Optimization）

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

**面试考察点**  
- **PPO vs DPO 的核心区别**：PPO 是在线 RL，需要奖励模型和 critic，采样成本高但可探索；DPO 是离线优化，直接利用静态偏好数据，实现简单但依赖数据质量。
- **为什么 DPO 不需要奖励模型？** DPO 通过将偏好概率建模为策略与参考策略的比值，隐式构造了奖励函数，绕过了显式奖励模型训练。
- **GRPO 的创新在哪？** 用组内奖励作为基线，去掉了 critic，简化了训练流程，特别适合多机多卡的 MoE 模型（如 DeepSeek-V2 使用 GRPO 进行 RLHF）。

为什么 GRPO 能涌现推理能力

1. **组内相对比较**：模型必须在同 prompt 的多个回答中分辨好坏，迫使它学会区分推理步骤的优劣，而不是只追求绝对高分。
2. **无 critic 误差**：长链推理的价值很难准确估计，去掉 critic 避免了偏差，让策略更新更稳定。
3. **鼓励多样性**：组采样自然要求回答多样化（否则优势接近零），促使模型探索不同推理路径，发现更优解法。

结果：模型生成更长、更结构化的推理链，甚至出现自我校正行为。


## 3 推理优化

### 3.1 KV Cache

**原理**  
在自回归生成时，每一步需要计算当前 token 与之前所有 token 的注意力。如果每次都重新计算所有历史 token 的 K 和 V，会引入大量重复计算。KV Cache 将已生成的 token 的 K、V 缓存在内存中，新 token 只需计算自己的 K、V，然后与缓存中的 K、V 拼接后进行注意力计算。

**内存占用**  
KV Cache 的大小为：

```math
\text{内存} = 2 \times \text{batch\_size} \times \text{num\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{sizeof(dtype)}
```

例如，Llama 2 7B（32 heads，head_dim=128）使用 FP16，batch=1，seq_len=4096 时，KV Cache 约

$$
2 \times 1 \times 32 \times 4096 \times 128 \times 2 \text{ bytes} = 2 \times 32 \times 4096 \times 256 \text{ bytes} = 2 \times 32 \times 1,048,576 \text{ bytes} = 67,108,864 \text{ bytes} \approx 64 \text{ MB}
$$

当 seq_len 达到 32k 时，占用约 512 MB，对于大 batch 或多头模型，KV Cache 会成为显存瓶颈。

**优化技巧**  
- **MQA（Multi-Query Attention）**：所有注意力头共享同一组 K、V，将 KV Cache 大小减少到原来的 $`1/\text{num\_heads}`$，但可能影响模型质量。
- **GQA（Grouped-Query Attention）**：将查询头分组，每组共享 K、V，在性能与内存之间取得平衡（如 Llama 2 7B 采用 GQA）。
- **KV Cache 量化**：将缓存的 K、V 量化为 INT8 或 FP8，可显著降低显存占用，但需要小心精度损失。

**面试考察点**  
- **KV Cache 为什么能加速？** 将每步的 $`O(S^2)`$ 复杂度降为 $`O(S)`$，但需要额外显存。
- **长上下文场景下 KV Cache 如何优化？** 可以通过 GQA、量化、分块处理（如 StreamingLLM 只保留部分缓存）来缓解。
- **如何实现 KV Cache？** 面试中可能会要求用伪代码描述生成过程中如何维护和更新缓存。


### 3.2 Flash Attention

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

**面试考察点**  
- **Flash Attention 如何减少内存访问？** 通过分块和在线 softmax，避免将 $`QK^T`$ 矩阵写入 HBM，大幅减少内存读写。
- **Flash Attention 与标准 attention 是否等价？** 是，输出数值上完全一致（忽略浮点误差）。
- **为什么 Flash Attention 对长序列特别有用？** 因为 $`QK^T`$ 的显存占用随 $`S^2`$ 增长，Flash Attention 避免了这一显存瓶颈，且能充分利用 SRAM 加速。


### 3.3 量化

**原理**  
量化将高精度浮点数（如 FP32、FP16）转换为低精度整数（如 INT8、INT4），从而减少显存占用和计算量。量化过程通常涉及 **缩放因子（scale）** 和 **零点（zero point）**。

**对称量化与非对称量化**  
- **对称量化**：假设数值范围关于 0 对称，公式为 $`x_q = \text{round}(x / s)`$，其中 $`s = \frac{\max(|x|)}{2^{b-1}-1}`$。适用于分布对称的权重。
- **非对称量化**：允许零点偏移，公式为 $`x_q = \text{round}((x - z)/s)`$，其中 $`z`$ 是零点，用于补偿分布不对称。适用于激活值。

**常见量化方法**  
- **训练后量化（PTQ, Post-Training Quantization）**：直接对训练好的模型进行量化，无需重新训练。简单快速，但可能带来精度损失。需要少量校准数据来确定量化参数。
- **量化感知训练（QAT, Quantization-Aware Training）**：在训练过程中模拟量化误差（如使用伪量化节点），使模型适应低精度表示。精度恢复较好，但需要重新训练。
- **GPTQ（Generalized Post-Training Quantization）**：一种针对权重的 PTQ 方法，通过逐层最小化量化误差，在 INT4 下仍能保持较高精度，常用于大模型（如 LLaMA）。
- **AWQ（Activation-aware Weight Quantization）**：根据激活值的分布来指导权重量化，保护重要通道的精度，常用在 4-bit 推理。
- **GGUF（GGML Universal Format）**：专为 CPU 推理设计的量化格式，支持多种量化精度（如 Q4_0、Q5_K_M），常用于本地运行大模型（如 llama.cpp）。

**精度损失与权衡**  
量化会引入近似误差，可能导致模型输出质量下降。通常权重对量化更鲁棒，激活值更敏感。通过混合精度（如保留部分关键层为 FP16）可以平衡速度与质量。

**面试考察点**  
- **量化为什么能加速？** 低精度矩阵乘法可以利用硬件加速（如 INT8 的 Tensor Core 比 FP16 快 2-4 倍），且显存占用减少，提高带宽利用率。
- **PTQ 和 QAT 的区别**：PTQ 无需训练，适合快速部署；QAT 需要训练，但精度恢复更好，适合对质量要求高的场景。
- **如何选择量化格式？** 根据部署平台（GPU 或 CPU）、模型大小、质量要求综合选择。例如，GPU 推理常用 GPTQ/AWQ，CPU 推理常用 GGUF。
- **量化对 MoE 的影响**：MoE 模型参数量大，量化能大幅减少显存占用，但需要小心处理负载均衡（量化可能使 Router 对专家的选择产生偏差）。


### 3.4 分布式训练（TP、PP、DP）

#### 数据并行（DP）
- **原理**：每张 GPU 持有完整的模型副本，处理不同的数据分片。前向和反向独立计算，梯度通过 AllReduce 同步，确保所有副本参数一致。
- **通信**：每步需同步梯度（通常使用 AllReduce），通信量与模型参数量成正比。
- **优点**：实现简单，适合单机多卡。
- **缺点**：模型必须能完整放入单卡显存；当模型过大时，无法使用。

#### 张量并行（TP）
- **原理**：将单个层内的参数切分到多张 GPU 上，每张卡只存储部分权重。例如 MLP 层：将第一个线性层的权重按列切分，第二个线性层按行切分，使前向时只需一次 AllReduce 即可合并结果。Attention 层：将多头切分到不同 GPU（每个卡负责部分头）。
- **通信**：每层前向后向各进行一次 AllReduce，通信量相对密集。
- **优点**：可以训练单卡放不下的超大模型（如 100B+）。
- **缺点**：需要精细设计切分方案，通信开销较大。

#### 流水线并行（PP）
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

**面试考察点**  
- **TP 与 PP 的区别**：TP 是层内切分，通信频繁但单步延迟低；PP 是层间切分，通信少但存在流水线气泡。
- **如何选择并行策略？** 根据模型大小、集群拓扑、显存限制综合决定。例如，超大模型通常用 TP+PP+DP 三维并行。
- **ZeRO 相比 DP 的优势**：ZeRO 消除了冗余，使显存占用随 GPU 数量线性扩展，可训练远超单卡容量的模型。
- **MoE 与专家并行**：MoE 天然适合专家并行（Expert Parallelism），将不同专家分布到不同 GPU，通过 All-to-All 通信实现路由，本质上是 TP 的一种特例。

### 3.5 混合精度训练（BF16 vs FP16）

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

**面试问题与回答要点**
1. **为什么大模型训练更倾向 BF16？**  
   → 大模型梯度范围大，FP16 易溢出；BF16 动态范围与 FP32 相同，无需损失缩放。
2. **BF16 精度低会不会影响收敛？**  
   → 通常不会，因为梯度更新本身具有噪声，且关键操作（如 softmax）仍可用 FP32。
3. **混合精度为什么能加速？**  
   → 低精度矩阵乘法利用 Tensor Core，同时减少显存带宽占用。

