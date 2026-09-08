---
title: "从 Prompt Engineering 到生产级 Agent Harness"
subtitle: "Runtime、Tool-use、Context、Memory、Sandbox 与多 Agent 的系统化理解"
date: "2026 年 7 月"
lang: zh-CN
---

大模型本身只负责根据输入生成下一段文本或结构化动作。真正让模型能够持续完成复杂任务的是围绕模型建立的一套确定性工程系统：它负责构建上下文、驱动 ReAct Loop、持久化状态、纠偏容错、限制权限、调度子Agent等。这套系统就是 **Agent Harness**。

本文的核心结论是：

> **Agent Harness 是围绕 Model‑Tool 为原子能力的 ReAct Loop 构建的，具备健壮可靠、纠偏兜底、成本友好的生产级 Agent 系统工程框架**。
---

# 一、Agent 架构的演进

## 1.1 Prompt Engineering：告诉模型“应该怎么回答”

Prompt Engineering 的核心是设计模型指令，包括角色、任务、约束、示例和输出格式。例如：

```text
你是一名高级 Python 工程师。
请阅读错误日志，定位测试失败原因，给出修复方案。
输出必须包含：原因、修改位置、验证方式。
```

这类 Prompt 可以显著改善一次性回答，但它仍然存在根本限制：模型没有仓库文件、不能执行测试，也无法知道修复是否真的有效。

在统一案例中，模型最多给出一份“看起来合理”的建议：

```text
可能是时间格式解析错误，请检查 parser.py 中的时区处理。
```

但“可能”不是证据。仅靠 Prompt，系统无法完成真实世界中的闭环。

> **Prompt Engineering 主要解决：模型应该如何理解和表达。**

## 1.2 Context Engineering：让模型“看到正确的信息”

下一步是把错误日志、相关代码、项目文档和历史对话放入上下文。RAG 是其中常见的一种机制，但 Context Engineering 不等同于 RAG。

完整的 Context 来源通常包括：

```text
System Instruction
User Message
Conversation History
Retrieved Documents
Tool Call / Tool Result
Active Skill
Workspace / Memory
Goal / Plan
Subagent Result
```

这些来源并不意味着要被全量塞入 Context。**Context 是模型这一次能看到的输入投影，不是系统全部事实的副本。**

完整的运行事实应留在外部：状态数据库保存 Tool / Task 生命周期，文件系统与 Artifact Store 保存原文件和大结果，Memory / RAG 按相关性召回，Subagent 在隔离 Context 内完成局部探索后只返回结论与证据引用。

模型缺什么，再通过 Tool 或 Reference 按需展开；这样既不把有限窗口耗在低价值原文上，也让压缩后仍可以追溯和恢复。

> **Context Engineering 主要解决：这一次模型调用应该看到什么。**

## 1.3 Harness Engineering：让模型“可靠地行动”

Harness Engineering 进一步处理模型调用之外的问题：

```text
如何选择和执行 Tool？
如何保存运行状态？
如何中止、恢复和重试？
如何把长期状态投影为本轮 Context？
如何拆分任务并调度 Subagent？
如何记录成本、延迟、错误和副作用？
```

因此，三者不是相互替代，而是逐层扩展：

![Prompt Engineering 到 Harness Engineering 的能力演进](figures/prompt-context-harness-evolution.png)

| 阶段 | 核心问题 | 主要机制 | 仍未解决的问题 |
|---|---|---|---|
| Prompt Engineering | 怎么告诉模型做事 | 指令、示例、格式约束 | 无法访问环境、无法验证结果 |
| Context Engineering | 给模型哪些信息 | RAG、历史选择、压缩、检索 | 无法安全执行、无法持久恢复 |
| Harness Engineering | 怎样让模型长期可靠行动 | Runtime、Tool、State、Sandbox、Orchestration | 需要持续工程治理与评估 |

可以用一句话概括发展过程：

> **Prompt 决定模型如何理解，模型输入上下文 决定模型这一轮真正看到什么，Harness 决定模型如何在外部世界持续、高效、健壮地行动。**

---

## 1.4 Agent Harness 总体架构

### 一句话定义

Agent Harness 是包围模型循环的工程控制系统。它不负责替代模型推理，而负责把概率性的模型输出转化为受约束、可观测、可恢复的执行过程。

一个生产级 Harness 通常包含：

```text
Agent Runtime
Context
Tool Pipeline
State Management
Workspace & Sandbox
Task Orchestration
```

![Agent Harness 总体架构](figures/agent-harness-architecture.png)

Runtime 就是循环的驱动器；Harness 则是除模型推理能力之外，支撑并约束整个循环的工程系统。不要让 Runtime 直接耦合每一种 Harness 机制：

```python
# 不推荐：让 Loop 知道每一种 Harness 机制
if tool_name == "update_plan":
    ...
if should_checkpoint:
    ...
if needs_subagent:
    ...
```

这种写法会让 Loop 逐渐变成不可测试、不可替换的“超级控制器”。更好的设计是：

- Loop 只识别统一的 Tool Call 和 Tool Result；
- Plan、Memory、Goal 通过普通 Tool 读写 DB/Store；
- Checkpoint 由 Runtime/Harness 在状态边界触发；
- Subagent 由 Scheduler 异步调度，但对模型可以表现为一个 Tool；
- 权限、审计、重试通过 Tool Executor 与 Hook 实现。

一次 Agent Turn 可以概括为三个过程：

```text
Environment（Prompt + State + Tool Results）+ Memory / RAG
    → Context Builder → 模型输入上下文
    读取、选择、压缩并冻结

Tool_calls
    → Parse → Registry Lookup → tool.execute(args, context)
    将模型意图映射为真实执行

Tool Result / Observation
    → Normalize → Tool Call State + Event Log + Artifact Ref
    → 追加为 Tool Message，供下一轮 Context Builder 读取
    记录事实、结果与外部引用
```

这里要区分**执行产物**与**返回结果**：`tool.execute` 执行时才可能写入外部系统、Thread State 或 User Memory Store；Tool Result 本身只是这次执行的 Observation。Harness 将其规范化、持久化并回填 Context。

这就是 Harness 的核心闭环：**持久状态与外部信息被投影给模型，模型输出结构化 Tool Call，Harness 再把该调用安全地映射到真实 Tool 实现。**

---

# 二、Agent Loop

Agent Loop 是 Agent Harness 的执行闭环：它围绕 ReAct 推进当前一轮的模型调用与工具执行。状态持久化、取消恢复、工作区和多 Agent 等跨轮能力统一由后文的 Runtime State 管理。

ReAct Loop 的原子能力只有两个：

```text
Model Call
Tool Use
```

模型根据当前 Context 生成文本或 Tool Call；Harness 执行工具并把结果返回模型，直到模型不再调用工具。

## 2.1 Model Call 与多 Provider 适配

### Chat Completions API

一次带工具的模型请求通常包含：

```text
messages
    当前模型能够看到的 Context

tools
    可用工具的名称、描述和参数 Schema

model parameters
    temperature、max_completion_tokens 等生成参数
```

示例 Payload：

```json
{
  "model": "example-model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful research assistant."
    },
    {
      "role": "user",
      "content": "搜索今天的重要新闻。"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0.2,
  "max_completion_tokens": 2048
}
```

Tool Description 和 Schema 会随 Messages 一起进入模型 Context。模型据此判断是否需要调用工具，以及应该生成哪些参数。

如果模型决定搜索新闻，Response 类似：

```json
{
  "id": "chatcmpl_123",
  "object": "chat.completion",
  "model": "example-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "{\"query\":\"今日重要新闻\",\"limit\":5}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 1250,
    "prompt_tokens_details": {
      "cached_tokens": 800
    },
    "completion_tokens": 86,
    "completion_tokens_details": {
      "reasoning_tokens": 42
    },
    "total_tokens": 1336
  }
}
```

其中：

```text
message.tool_calls
    模型请求执行的工具及参数

finish_reason
    本次生成结束的原因

usage
    输入、输出、缓存和推理 Token 消耗
```

模型不会直接执行 `web_search`，而只会返回结构化 Tool Call。Harness 解析工具名称和参数，再映射到真正的 Tool：

```python
call = parse_tool_call(response)
tool = tool_registry.get(call.name)
result = await tool.execute(call.arguments, context)
```

如果模型不需要工具，则会直接返回文本，`finish_reason` 通常为 `stop`。

### 多 Provider 适配

Chat Completions 只是模型 API 的一种形式。不同 API 和 Provider 对同一概念的表示并不一致：

```text
OpenAI Chat Completions
    messages / choices / tool_calls / finish_reason

OpenAI Responses
    input / output / function_call / status

Anthropic
    messages / content blocks / tool_use / stop_reason

Gemini
    contents / candidates / functionCall / finishReason
```

如果 ReAct Loop 直接依赖某个 Provider 的原始字段，切换模型时就必须修改 Runtime。

因此，Harness 需要通过 Model Adapter 将不同 Provider 的请求和响应转换为统一结构：

```python
@dataclass
class ModelRequest:
    messages: list
    tools: list
    model_config: dict


@dataclass
class ModelResponse:
    text: str | None
    tool_calls: list
    finish_reason: str
    usage: dict
```

完整调用链路为：

```text
Agent Runtime
→ 统一 ModelRequest
→ Model Adapter
→ Provider API
→ Model Adapter
→ 统一 ModelResponse
→ Agent Runtime
```

这样，Runtime 只需要处理统一的文本、Tool Call、结束状态和 Token Usage，不需要关心底层使用的是 OpenAI、Anthropic、Gemini 还是自建模型。

### 流式输出：SSE 也是 Model Adapter 的适配边界

模型适配不只是请求 Body 中 `messages`、`tools`、`max_tokens` 等字段的转换；流式生成时，还要统一 **SSE（Server-Sent Events）事件协议**。

SSE 本质上是一条保持打开的 HTTP 响应：客户端只发起一次请求，服务端随后持续向下推送 Token 增量。典型响应头为：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

每个 SSE 事件由若干字段组成，并以空行结束：

```text
event: message.delta
data: {"type":"text_delta","text":"你好"}

```

OpenAI Chat Completions 通常不显式使用 `event:` 字段，而是持续输出 `data:` 行，并以 `[DONE]` 作为结束标记：

```text
data: {"id":"chatcmpl_xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl_xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

不同 Provider 的差异不只在字段名：

| 差异 | OpenAI Chat Completions | Anthropic / Responses 等 |
|---|---|---|
| 事件形式 | 连续 `data:` Chunk | 常带 `event:` 类型 |
| 文本增量 | `choices[].delta.content` | `content_block_delta`、`response.output_text.delta` 等 |
| Tool Call | `delta.tool_calls`，参数分片到达 | `tool_use` / function-call 事件 |
| 结束信号 | `finish_reason` + `[DONE]` | `message_stop`、`completed`、`status` 等 |
| 用量 | 常在末尾 Chunk 返回 | 可能独立事件或仅最终响应返回 |

因此，Model Adapter 应同时完成两件事：

```text
请求适配：
统一 ModelRequest
→ Provider HTTP Header + JSON Body

响应适配：
Provider SSE Event Stream
→ 解析事件与增量
→ 映射为 OpenAI-compatible SSE Chunk
→ Agent Runtime / 上层客户端
```

对 Harness 而言，最关键的是把底层事件归一为统一语义：

```text
text_delta       文本增量
tool_call_delta  工具名 / arguments 增量
usage            Token 用量
finish           stop / tool_calls / length
error            上游异常、限流或断流
```

Tool Call 的 `arguments` 往往会被拆成多个 SSE Chunk，因此不能每收到一段就解析 JSON。Adapter 应按 `tool_call_id` 或 `index` 缓冲分片，在收到 `finish_reason = tool_calls` 后再拼接、解析并交给 Schema Validation：

```python
tool_buffers[call_id] += delta.arguments

if finish_reason == "tool_calls":
    arguments = json.loads(tool_buffers[call_id])
    validate_tool_call(name, arguments)
```

核心结论：**HTTP JSON Body 适配解决“模型怎样被调用”，SSE 事件适配解决“生成过程怎样被持续、正确地交付”。二者共同构成生产级 Model Adapter。**

### LLM 限流与 Fallback：不只限制 QPS，也要限制 TPM

普通 API 常只关心 QPS；LLM 一次请求的输入和输出长度差异很大，真正容易打满的往往是 Provider 按模型分配的 **TPM（Tokens Per Minute）**。因此通常同时有三道闸：

```text
全局 Provider / Model 配额
        ↓
租户 / 项目配额
        ↓
用户 / Thread / Run 配额

QPS / RPM：单位时间能发起多少请求
TPM：单位时间能消耗多少输入 + 输出 Token
Concurrency：同时保持多少条模型流 / 长连接
```

每个调用必须同时从对应层级的限流器获取许可；任一层不足即等待、排队或快速拒绝。实现上常用 Token Bucket / Leaky Bucket：QPS Bucket 每次扣 1 个请求令牌；TPM Bucket 按 Token 数扣额度。限流 Key 至少带上 `provider + model + tenant/project`，因为不同模型的实际配额、成本和上下文窗口常常不同。

TPM 的难点是输出 Token 在调用前未知。常见做法是**预扣再结算**：先用 tokenizer 估算输入 Token，并预留 `max_output_tokens`；流结束后根据 Provider 返回的 `usage` 用实际量结算，多扣的额度归还。对流式输出，也可随着 `text_delta` 近似递减预留，最终仍以 `usage` 为准。

```python
async def call_model(request: ModelRequest, route: Route):
    provider = route.primary
    input_tokens = tokenizer.count(request.messages, request.tools)
    reserved = input_tokens + request.model_config["max_output_tokens"]

    # 同时获取 QPS、TPM 与在途请求许可；不足则按请求类型排队或拒绝。
    permits = await limiter.acquire(
        key=(provider.name, provider.model, request.tenant_id),
        requests=1,
        tokens=reserved,
        concurrency=1,
    )
    try:
        response = await provider.stream(request)
        actual = response.usage.input_tokens + response.usage.output_tokens
        await limiter.settle(permits, actual_tokens=actual)  # 归还未使用的预留
        return response
    except TransientModelError as exc:  # 429、529、5xx、短暂断连等
        circuit.record_failure(provider, exc)
        backup = route.next_compatible(provider)
        if backup and circuit.allow(backup):
            return await call_model(request.with_model(backup), route.after(backup))
        raise
    finally:
        await limiter.release_concurrency(permits)
```

上面是表达职责的简化代码。真实实现还会给交互请求设置短等待队列、给后台任务设置延迟重试；Provider 若返回 `Retry-After`，应优先尊重它。QPS、TPM 和并发配额都应由策略配置和实时指标驱动，而不是写死在 Agent Prompt 中。

**Fallback 不是“任意换个模型再试”。** 路由层预先维护可替代关系：备用模型必须满足本轮需要的能力，如流式输出、Tool Calling、JSON / Structured Output、足够的 Context Window、安全策略和区域/成本约束。典型路径是：

```text
短暂 429 / 529 / 5xx / Provider Circuit Open
→ 同模型短退避重试
→ 兼容备用模型
→ 降低 max_tokens / 排队等待 / 返回系统繁忙
```

- **可以 Fallback**：临时限流、容量不足、网络断连、Provider 不可用；
- **不应 Fallback**：参数或 Schema 错误、鉴权失败、上下文超窗、内容策略拒绝。这些要由 Harness 裁剪 Context、修正请求或回给 Agent Replan。
- **有 Tool Call 的 Run**：模型故障后从最近稳定 Checkpoint 以持久化的 Tool / Task 状态恢复；已成功的 Tool Result 直接复用，不能因为换模型而重复执行有副作用的 Tool。

> **QPS 保护请求速率，TPM 保护模型容量与成本，并发上限保护长流连接；Fallback 由路由策略决定能力兼容性，不能由模型临场猜测。**

## 2.2 最小 ReAct Loop

ReAct Loop 只做四件事：

```text
调用模型
读取 Tool Call
执行 Tool
把 Tool Result 返回模型
```

全文以 OpenAI Chat Completions 的 Tool Calling 结构为准。模型返回的是 Assistant Message 中的 `tool_calls`：

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_test_1",
      "type": "function",
      "function": {
        "name": "run_tests",
        "arguments": "{\"target\":\"tests/test_parser.py\"}"
      }
    }
  ]
}
```

Harness 执行工具后，以相同的 `tool_call_id` 追加 `role: "tool"` 消息，再发送给模型。模型可能继续调用 `read_file`、`apply_patch` 和 `run_tests`，直到不再产生 Tool Call。

最小循环非常简单：

```python
async def react_loop(model, messages, tool_executor, tool_schemas):
    while True:
        reply = await model.complete(
            messages=messages,
            tools=tool_schemas,
        )
        messages.append(reply)

        if not reply.tool_calls:
            return reply

        for call in reply.tool_calls:
            result = await tool_executor.execute(call)
            messages.append(result)
```

Go 版的控制流相同：只要本轮仍有 Tool Call，就执行并把 Result 追加回消息；没有 Tool Call 才结束本轮 Run。

```go
func ReactLoop(
	ctx context.Context,
	model Model,
	messages []Message,
	tools []ToolSchema,
	executor ToolExecutor,
) (Message, error) {
	for {
		reply, err := model.Complete(ctx, messages, tools)
		if err != nil {
			return Message{}, err
		}
		messages = append(messages, reply)

		if len(reply.ToolCalls) == 0 {
			return reply, nil
		}
		for _, call := range reply.ToolCalls {
			result, err := executor.Execute(ctx, call)
			if err != nil {
				return Message{}, err
			}
			messages = append(messages, result)
		}
	}
}
```

它形成的闭环是：

```text
Model
→ Function Call
→ Tool Execution
→ Tool Result / Observation
→ Model
```

Memory、Plan、Checkpoint 和 Subagent 都不应写成 ReAct Loop 中的特殊分支，而应通过 Tool、State Store、Hook 或 Scheduler 接入。

**源码对照：** Pi 的 [`agent-loop.ts`](https://github.com/earendil-works/pi/blob/main/packages/agent/src/agent-loop.ts) 和 [`packages/agent/README.md`](https://github.com/earendil-works/pi/blob/main/packages/agent/README.md)。

## 2.3 异步 Tool 执行：事件推进下一轮

Agent Loop 不是一个持续占用线程的 `while` 循环。模型负责决定下一步调用什么；Harness 异步执行 Tool，并在结果事件到达时调度下一次模型调用。取消、恢复与跨轮状态治理由第 5 章 Runtime State 负责。

```text
Context → Model → Tool Call
                   ↓
              异步执行 Tool
                   ↓
              Tool Result Event
                   ↓
        Harness 收敛结果 → 下一轮 Model Call
```

### 工具完成如何驱动下一轮：异步通知

模型一次可以输出多个 Tool Call，但它不会观察执行过程。Harness 必须异步执行 Tool / Subagent：任务运行时挂起当前协程，结果到达后再通过事件唤醒；如果同步阻塞或持续轮询，会长期占用线程、连接与计算资源。

![Agent Harness 异步 Tool Call 通知机制](figures/agent-harness-async-tool-notification.png)

| 任务类型 | 实现方式 | 适用场景 |
|---|---|---|
| 短任务 | Async I/O：`await`、Event Loop、I/O 多路复用、Future/Promise 完成事件 | Web API、RPC、数据库查询、搜索等秒级任务 |
| 长任务 | MQ / Durable Task：任务持久化后由 Worker 消费，完成后投递 Result Event | Browser、Sandbox、Deep Research、大文件处理等长耗时任务 |

```text
短任务：Harness await Tool → 协程挂起 → I/O Ready / RPC 返回 → 恢复协程

长任务：Harness 记录 PENDING → 投递 MQ → Worker 执行
       → 写回终态 → 发送完成通知 → Harness 检查该 Step 是否收敛
```

无论 RPC 还是 MQ，完成通知都只需要携带关联键和结果引用：

```text
ToolCompleted {
  run_id, step_id, tool_call_id,
  status, result_ref
}
```

**状态表是事实来源，通知只是触发一次检查。** Harness 收到 `ToolCompleted` 后按 `run_id + step_id` 查询该批 Tool Call：仍有 `PENDING / RUNNING` 就继续等待；全部终态才按 `tool_call_id` 收集 `result_ref` 并进入下一轮 Model Call。

这不是要求“唤醒最初发起调用的那一个协程”。单机短任务中，Future 完成会让原协程重新进入 Event Loop；多 Pod / 长任务中，`ToolCompleted` 由任一健康 Harness Consumer 消费即可。它以 `run_id + step_id` 读取并条件更新持久化 State，再获取该 Run 的 lease / 调度权继续执行。因此 Pod 可以扩缩容或故障迁移，关联键始终是 Run 与 Step，而不是 Pod ID。

```text
Model 输出一批 Tool Call
→ 为每个 Call 落库 PENDING，Step = WAITING_TOOL_RESULTS
→ Dispatcher / MQ 分发
→ Executor 条件更新 PENDING → RUNNING → 终态 + result_ref
→ 发布 ToolCompleted(run_id, step_id, tool_call_id)
→ 任一 Harness Consumer 做 Step Join
   ├─ 仍有未终态 Call：结束本次消费
   └─ 全部终态：抢到 Run lease，按 tool_call_id 回填 Result，发起下一轮 Model Call
```

```python
TERMINAL = {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELLED"}

async def start_tool_step(runtime, calls):
    await runtime.create_calls(calls, status="PENDING")
    await runtime.set_step_status(calls[0].step_id, "WAITING_TOOL_RESULTS")
    for call in calls:
        await dispatcher.submit(call)  # 短任务可 create_task；长任务可 publish MQ

async def run_one_call(runtime, call):
    # 条件更新保证 MQ 至少一次投递时不重复真实执行。
    if not await runtime.claim(call.id, from_status="PENDING", to_status="RUNNING"):
        return
    result = await executor.execute(call, runtime.context(call))
    await runtime.finish(call.id, result)  # 写终态与 result_ref
    await event_bus.publish("tool-completed", call.run_id, call.step_id, call.id)

async def on_tool_completed(runtime, event):
    calls = await runtime.calls_of_step(event.run_id, event.step_id)
    if any(call.status not in TERMINAL for call in calls):
        return
    if not await runtime.acquire_run_lease(event.run_id):
        return  # 另一个 Consumer 正在推进，重复完成事件可安全忽略
    results = sorted(calls, key=lambda c: c.tool_call_id)
    await runtime.resume_model(event.run_id, event.step_id, results)
```

Go 的职责完全相同：条件更新负责幂等，完成事件负责触发 Join，Run Lease 防止多个 Consumer 重复推进模型。

```go
func StartToolStep(ctx context.Context, rt Runtime, calls []ToolCall) error {
	if err := rt.CreateCalls(ctx, calls, "PENDING"); err != nil { return err }
	if err := rt.SetStepStatus(ctx, calls[0].StepID, "WAITING_TOOL_RESULTS"); err != nil { return err }
	for _, call := range calls { dispatcher.Submit(ctx, call) } // goroutine 或 MQ
	return nil
}

func OnToolCompleted(ctx context.Context, rt Runtime, e ToolCompleted) error {
	calls := rt.CallsOfStep(ctx, e.RunID, e.StepID)
	if !AllTerminal(calls) { return nil }
	if !rt.AcquireRunLease(ctx, e.RunID) { return nil }
	return rt.ResumeModel(ctx, e.RunID, e.StepID, SortByToolCallID(calls))
}
```

#### 短任务：Async I/O 足够

Web API、RPC、数据库查询等任务通常在一个连接生命周期内结束。Harness 调用 `await tool.execute()` 后，协程会让出执行权挂起；底层 I/O Ready 或 Future 完成时，再将原协程放回 Ready Queue。因此一个 Runtime 可以同时承载大量等待中的 Tool Call。

```python
async def run_short_tool(call):
    state_store.mark_running(call.id)
    try:
        result = await asyncio.wait_for(
            tool_executor.execute(call), timeout=call.timeout
        )
        state_store.mark_succeeded(call.id, result)
    except asyncio.TimeoutError:
        state_store.mark_timeout(call.id)
    except Exception as error:
        state_store.mark_failed(call.id, error)
    finally:
        completion_event.notify(call.step_id, call.id)
```

#### 长任务：MQ 分发，RPC 或 MQ 回传完成通知

Browser、Sandbox、Deep Research 或大文件处理可能运行数分钟甚至更久。如果 Harness 一直维持 HTTP / TCP 连接，容易受到连接超时、实例重启和扩缩容迁移影响。
更稳妥的方式是先持久化 Tool Call，再投递 Command；Worker 独立消费和执行，完成后写回状态，并发出带 `run_id + step_id + tool_call_id` 的完成通知。Harness 可以释放当前请求资源，收到通知后再检查对应 Step。

```text
Harness：ToolCall=PENDING → ToolRequest Queue
Worker ：消费 ToolRequest → RUNNING → 执行 → 写 Result / 终态
Worker ：RPC Callback 或 ToolCompleted MQ → Harness 检查 Step
```

MQ 的核心作用是 Tool Request 的生产消费解耦、任务持久化、失败重试和削峰填谷。完成通知可直接 RPC 调用稳定的 Harness Callback，也可投递 `ToolCompleted` MQ；前者延迟低，后者更适合长任务与跨服务恢复。两条路径最终都回到同一个 Step 状态检查。

#### 多个结果如何收敛：Join

同一轮多个 Tool Call 共享 `step_id`，每个调用使用独立 `tool_call_id`。任一完成通知都会触发一次 Step Join；还有 `PENDING / RUNNING` 就不推进，全部进入终态后才开始下一轮 Model Call。Subagent 的 Join 同理，只是聚合键从 Tool Batch 换成 Task DAG / `task_id` 集合。

#### 心跳：只负责异常兜底

无论走哪条路径，Tool / Task 都应记录 `PENDING → RUNNING → SUCCEEDED / FAILED / CANCELLED / TIMEOUT`，长任务 Worker 还要定期更新 `last_heartbeat`。后台定时任务周期检查：

```text
now > deadline                   → TIMEOUT
now - last_heartbeat > threshold → Worker 失联，取消 / 重试 / 故障接管
状态已终结但通知未消费           → 补发完成事件
```

正常结果依靠事件通知立即返回；心跳扫描只处理超时、失联、通知丢失等异常，不让每个 Harness Thread 自己轮询任务状态。

# 三、Tool-use：Function Calling、MCP 与 Skill

Tool-use 是模型作用于外部世界的统一通道。无论底层是 Python 函数、CLI、浏览器、MCP Server，还是访问 DB 的 Goal / Plan 状态工具，Memory 召回和写入，对 ReAct Loop 都应表现为统一的 Tool Call / Tool Response 协议。

![Tool-use 生态：Function Calling、Skill 与 MCP](figures/tool-use-ecosystem.png)

这里要区分三个层次：**Function Calling 是模型表达动作的协议，Harness 是解释并执行动作的运行时，Skill 是由 Harness 按需读取并回填给模型的能力说明**。
Skill 不是绕过 Function Calling 独立注入模型；通常先由模型发起 `load_skill(skill_name)` 的 Function Call，获得逐步披露的指令与资源，再据此发起后续 Function Call。这样既避免一次加载全部 Skill 占满上下文，也让每一次读取和执行都经过同一套权限、审计与异常处理。

## 3.1 Tool Registry：全量注册，分层曝光，按需加载

参考 Codex 的实现，**注册**与**模型可见性**必须分开：所有本地 Tool、MCP Tool 与动态能力都先进入 Registry；每个条目再带有曝光策略。Registry 是服务端事实源，模型初始请求只看到其中一小部分。

```text
所有 Tool → Registry（ToolName → Schema / Policy / Executor / Version）
           ├─ DIRECT   → 初始模型可见，进入稳定 Tool Prefix
           ├─ DEFERRED → 不进初始上下文；可由 tool_search 找到并原生加载
           └─ HIDDEN   → 仅供 Runtime / 内部流程使用，模型不可见
```

Codex 的 `ToolRegistry` 用 `IndexMap<ToolName, RegisteredTool>` 保存全量 Runtime 与 `ToolExposure`；执行时再按 `ToolName` 直接 Resolve。其 `tool_search` 仅对 `DEFERRED` 条目构建 BM25 索引，命中后返回的不是一段工具说明文本，而是包含完整 Schema 的 `LoadableToolSpec[]`。这些定义被写成原生 `ToolSearchOutput` 对话项，模型随后直接调用真实 `tool_name`；不需要再经 `invoke_tool(tool_name, arguments)` 二次分发。

```text
第 N 轮：DIRECT Tool + tool_search
    ↓
模型调用 tool_search("创建 GitHub Issue")
    ↓
Registry 从 DEFERRED Tool 中检索并返回 LoadableToolSpec[]
    ↓  （Provider Native ToolSearchOutput）
完整 Schema 在对话的动态位置可用
    ↓
第 N+1 轮：模型直接调用 mcp__github.create_issue(...)
    ↓
Registry.resolve(tool_name) → Executor
```

这也是避免 Prefix Cache 抖动的关键：初始 Tool Prefix 不会因为长尾 Tool 的发现而重建；被召回的 Schema 作为动态对话项进入当前链路。Codex 相关实现可参见 [Tool Registry](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/registry.rs)、[ToolSearch Handler](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/tool_search.rs) 与 [ToolSearchOutput](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/context.rs)。

```python
from enum import Enum

class Exposure(str, Enum):
    DIRECT = "direct"
    DEFERRED = "deferred"
    HIDDEN = "hidden"

# 服务端全量目录；真实实现使用有序 Map，tool_name 是唯一键。
tool_registry: dict[str, Tool] = {}

def register(tool: Tool):
    assert tool.name not in tool_registry
    tool_registry[tool.name] = tool

def visible_specs():
    direct = [t.schema for t in tool_registry.values()
              if t.exposure is Exposure.DIRECT]
    return [*direct, tool_search_schema]       # 初始模型可见 Tool

def tool_search(query: str, ctx) -> ToolSearchOutput:
    candidates = [t for t in tool_registry.values()
                  if t.exposure is Exposure.DEFERRED and allowed(t, ctx)]
    matched = bm25_rank(query, candidates)[:8]
    # 不是普通文本；支持该协议的 Provider 会把定义加载到动态会话位置。
    return ToolSearchOutput(to_load=[t.loadable_schema for t in matched])

def resolve(tool_name: str) -> Tool:
    return tool_registry[tool_name]            # O(1) 执行分发
```

Go 版的核心是同一件事：Registry 管注册与 Resolve，Exposure 管模型可见性。

```go
type Exposure string
const (
	Direct Exposure = "direct"
	Deferred Exposure = "deferred"
	Hidden Exposure = "hidden"
)

type Tool struct { Name string; Exposure Exposure; Schema []byte; Run func(map[string]any) any }
var toolRegistry = map[string]Tool{}

func Register(t Tool) { toolRegistry[t.Name] = t }
func Resolve(toolName string) (Tool, bool) { t, ok := toolRegistry[toolName]; return t, ok }
func VisibleSpecs() (out [][]byte) {
	for _, t := range toolRegistry { if t.Exposure == Direct { out = append(out, t.Schema) } }
	return append(out, toolSearchSchema)
}
func ToolSearch(query string) []LoadableToolSpec {
	return BM25DeferredTools(query, toolRegistry, 8) // 原生 ToolSearchOutput 的 payload
}
```

区分规则由产品与治理策略配置，而不是模型决定：高频、通用、低风险且 Schema 小的能力标为 `DIRECT`；低频、领域专用、权限敏感或 Schema 很大的能力标为 `DEFERRED`；仅 Runtime 使用的控制能力标为 `HIDDEN`。无论曝光方式如何，真实执行都统一经过 `registry.resolve(...)`、Policy 与 Executor。

## 3.2 MCP：远程工具的注册与调用

MCP 和 Function Calling 位于不同边界：

```text
Function Calling
    Model ↔ Agent Harness

MCP
    Agent Harness ↔ MCP Server
```

更准确地说：

> Function Calling 让模型表达“我要调用哪个工具”；MCP 则让 Harness 通过 JSON-RPC 发现并调用远程工具。

MCP 的核心意义主要有两个：

```text
远程工具注册
远程工具调用
```

首先，Harness 作为 MCP Client，与 MCP Server 建立连接，并通过 JSON-RPC 完成初始化和工具发现：

```text
initialize
→ tools/list
→ 获取 Tool Name、Description 和 Input Schema
→ 注册到 Harness 的 Tool Registry
```

例如，GitHub MCP Server 返回一个 `search_issues` 工具后，Harness 可以将它注册为：

```text
github.search_issues
```

模型看到的仍然只是普通的 Tool Description 和 Schema，不需要关心它来自本地函数还是远程 MCP Server。

当模型需要调用该工具时，会返回 Function Call：

```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_123",
    "type": "function",
    "function": {
      "name": "github.search_issues",
      "arguments": "{\"query\":\"sandbox bug\"}"
    }
  }]
}
```

Harness 根据工具名称从 Registry 中找到对应的 MCP Tool Adapter，并执行：

```python
async def execute(self, args, context):
    return await mcp_client.call_tool(
        name=self.remote_name,
        arguments=args,
    )
```

底层的 `call_tool` 会转换为 MCP 的 JSON-RPC 请求：

```text
tools/call
```

完整过程可以概括为：

```text
MCP Server 提供远程工具
→ Harness 通过 tools/list 发现工具
→ 注册到 Tool Registry
→ 模型产生 Function Call
→ Harness 找到 MCP Tool Adapter
→ Adapter 发送 JSON-RPC tools/call
→ 远程结果转换为普通 Tool Result
→ 返回模型
```

因此，MCP 并不替代 Function Calling，而是 Function Calling 后面的远程工具接入层：

```text
Function Calling
    模型如何请求工具

Tool Registry
    工具名称映射到哪个执行对象

MCP
    远程工具如何被发现和调用
```

使用 `github.search_issues`、`database.run_query` 这样的 Namespace，可以避免不同 MCP Server 的工具重名。

**源码对照：** [MCP Tools Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) 与 Codex 的 [MCP Interface](https://github.com/openai/codex/blob/main/codex-rs/docs/codex_mcp_interface.md)。

------

## 3.3 Skill：通过 Tool Call 渐进式加载知识

Skill 是针对某类任务准备的程序性知识，例如代码审查、测试调试、数据分析或 PDF 生成。

一个 Skill 通常包括：

```text
名称和描述
完整执行说明
参考资料
可选脚本
```

系统可能安装数百个 Skill，但当前任务通常只需要一两个。如果把所有 Skill 内容一次性放进 Context，会造成 Token 浪费、注意力稀释和指令冲突。

因此，Skill 应采用渐进式披露。

启动时不需要把所有 Skill 名称、描述或完整内容预置到 Context。模型只看到固定的核心能力；需要某类知识时，先用 `tool_search(kind="skill", query="...")` 获取少量候选与 `skill_ref`，再按需加载。

当模型判断当前任务需要某个候选 Skill 时，会产生一次普通 Tool Call：

```json
{
  "role": "assistant",
  "tool_calls": [{
    "id": "call_skill_1",
    "type": "function",
    "function": {
      "name": "load_skill",
      "arguments": "{\"skill_ref\":\"skillref://python-test-debugging@v2\"}"
    }
  }]
}
```

Harness 执行 `load_skill` Tool，从 Skill Registry 中读取对应的 `SKILL.md`：

```python
async def execute(self, args, context):
    return skill_registry.load(args["skill_ref"])
```

Skill 内容作为 Tool Result 返回，并以追加消息的方式进入下一轮 Context。

完整过程是：

```text
模型调用 tool_search 发现少量 Skill 候选
→ 模型判断需要某个 Skill
→ 模型调用 load_skill
→ Harness 读取完整 SKILL.md
→ Skill 内容作为 Tool Result 返回
→ 下一轮 messages[] 保留该 Skill 内容
```

如果 Skill 还包含大量 Reference，也可以只先暴露 Reference 的名称，在模型真正需要时再通过 `load_skill_reference` 加载。

因此可以将 Skill 的渐进式披露理解为三层：

```text
第一层：tool_search 返回少量 Skill 摘要与 skill_ref
第二层：完整 SKILL.md
第三层：具体 Reference 或 Script
```

Skill 不等同于 Tool：

```text
Skill
    描述某类任务应该如何完成

load_skill Tool
    让模型按需获取 Skill 内容

业务 Tool
    执行搜索、读写文件、运行命令等实际动作
```

因此，更准确的表述是：

> Skill 是程序性知识，而 Skill Loading 本质上是一次标准 Tool Call。模型通过 `load_skill` 动态加载扩展技能和知识空间。

**源码对照：** Pi 的 [Skills 文档](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md)。

## 3.4 Tool Executor：从模型意图到真实执行

模型只负责生成 `tool_calls`，没有权限决定“能不能执行、重试几次、能等多久”。这些决策由 **Tool Registry 的静态策略**（下游 SLA、幂等能力和业务风险配置）与 **Executor 的运行时判断**共同完成：

```text
解析调用 → 约束与准入 → 执行与状态落库 → 重试 / 熔断处理 → 标准化 Result 回写
```

状态机负责“何时调度”（见 2.3），Registry 负责“是什么工具”，Executor 则负责“是否允许执行、如何执行及结果是什么”。

### 一次调用如何被安全执行

```text
tool_call
→ Resolve：Registry 按 tool_name 找到 Function Handler / MCP Adapter
→ 校验：JSON 解析、Schema、业务不变式
→ Admit：ACL / Policy / Approval / Sandbox / Quota / Concurrency / Budget
→ 状态 / Ledger：PENDING → RUNNING，记录 deadline、attempt、idempotency_key
→ Execute：受 Deadline 与 AbortSignal 控制
→ Finish：Normalize Result → SUCCEEDED / FAILED / TIMEOUT / CANCELLED
→ Tool Result：携带原始 tool_call_id 回给模型
```

其中 Runtime State 以 Executor 写回的真实状态和 Result 为准，模型不能声明“已经执行成功”。外部 MCP / Web 输出按不可信数据处理并限制大小；大结果转存 Artifact Store，只向 Context 回写摘要与 `artifact_id`。凭证由 Executor 最小权限、短时注入，绝不交给模型。

下面的代码只表达执行责任边界；PENDING、RUNNING 与完成事件的异步推进由 2.3 处理。

```python
async def execute(call, context):
    tool = resolve(call.tool_name)                 # Map 查询真实 Tool
    args = json.loads(call.arguments)
    validate(tool["schema"], args)

    decision = admit(tool, args, context)          # ACL / Approval / Sandbox / 配额 / 预算
    if not decision.allowed:
        return result(call.tool_call_id, "REJECTED", decision.reason)
    if breaker_open(tool["name"]):
        return result(call.tool_call_id, "FAILED", "circuit open")

    for attempt in range(tool["retry_policy"].max_attempts):
        try:
            data = await run_with_deadline(
                tool["handler"](args, context), context.remaining_deadline
            )
            return result(call.tool_call_id, "SUCCEEDED", normalize(data))
        except TemporaryError as err:
            if not can_retry(tool, call, attempt):
                return result(call.tool_call_id, "FAILED", classify(err))
            await backoff_with_jitter(attempt, context.remaining_deadline)
        except TimeoutError:
            return await resolve_unknown_or_timeout(call, tool)
        except Exception as err:
            return result(call.tool_call_id, "FAILED", classify(err))
```

Go 版控制流相同：

```go
func Execute(ctx context.Context, call ToolCall) ToolResult {
	tool := Resolve(call.ToolName)
	args, err := ParseAndValidate(call.Arguments, tool.Schema)
	if err != nil { return Result(call.ID, "FAILED", err.Error()) }
	if d := Admit(tool, args, ctx); !d.Allowed { return Result(call.ID, "REJECTED", d.Reason) }
	if BreakerOpen(tool.Name) { return Result(call.ID, "FAILED", "circuit open") }

	for attempt := 0; attempt < tool.Retry.MaxAttempts; attempt++ {
		data, err := RunWithDeadline(ctx, RemainingDeadline(ctx), tool.Run, args)
		if err == nil { return Result(call.ID, "SUCCEEDED", Normalize(data)) }
		if IsTimeout(err) { return ResolveUnknownOrTimeout(ctx, call, tool) }
		if !IsTemporary(err) || !CanRetry(tool, call, attempt) { return Result(call.ID, "FAILED", Classify(err)) }
		BackoffWithJitter(ctx, attempt)
	}
	return Result(call.ID, "FAILED", "retry exhausted")
}
```

### Retry：只重试可恢复的瞬态故障

最重要的判断是：

> **Transient Error → Retry；Semantic Error → Replan。**

只有“环境恢复后，同一次调用可能成功”的故障才可自动重试：网络超时 / 连接重置、临时 `429`、`5xx / 529 Overloaded`、MCP HTTP/SSE 断连或暂时不可用。策略采用**指数退避加随机抖动**，并设置最大重试次数和统一的重试预算。

`InvalidArguments`、Schema Error、File Not Found、Test / Compile Failure、明确的 Auth / Permission Error、业务语义错误都不应盲目重试；它们以标准 Tool Result 回灌给模型，Agent 再改参数、修复任务或换 Tool。重试只能由一个最了解幂等语义的层级负责，通常是 Executor / Adapter，不能让 SDK、MCP Client 和 Agent Loop 各自重试。

### 幂等与“状态未知”

Retry 还必须判断动作是否可以安全重放。`Read`、`Search`、`GET`、`QueryStatus` 等只读操作通常天然幂等；`SendEmail`、`CreateOrder`、`Deploy`、`CreateIssue`、`Payment` 等写操作则不能把 Timeout 当作“未执行”。可能发生的是服务端已成功执行，但响应丢失，客户端才超时。

因此写操作要携带 `idempotency_key` / `operation_id`，并在异常后查询 Ledger 或下游状态：

```text
Timeout
→ Query operation status
   ├─ SUCCESS     → 复用已有结果
   ├─ NOT_STARTED → 可安全 Retry
   └─ UNKNOWN     → UNKNOWN_SIDE_EFFECT，不自动重放
```

> **At-least-once Retry 必须配合 Idempotency，不能用重复副作用换可靠性。**

### Timeout：限制单次调用，不等整个 Agent 卡死

不同执行单元应有独立 Deadline：

```text
Model API Timeout
Tool / Bash Timeout
MCP Timeout
Hook Timeout
Background Subagent Stall Timeout
Task / Run Deadline
```

单步 Timeout 负责把 Hang 变成显式 `TIMEOUT`；Task / Run Deadline 控制整个逻辑调用的生命周期。Retry 只能消费**剩余 Deadline**，不能每次重试都重新获得完整超时时间。例如 Task 总 Deadline 为 30 秒，前两次调用与退避已消耗 22 秒时，第三次最多只有 8 秒。

```text
Hang → Timeout → 按错误类别与幂等性分类
     → Retry（可安全） / 标准 Tool Result（需 Replan）
```

### Circuit Breaker：局部依赖熔断 + Agent 全局熔断

**局部 Circuit Breaker** 针对具体故障域，如 `web_search:serper`、`mcp:github`、`provider:openai`。维护滑动窗口健康状态：

```text
CLOSED -- failure / timeout rate 超阈值 --> OPEN
OPEN -- cool_down --> HALF_OPEN -- probe success --> CLOSED
                              └-- probe fail ------> OPEN
```

例如最近 60 秒内至少 20 次请求、失败率超过 50% 即 OPEN。OPEN 后新的 Tool Call 直接 Fast Fail 为 `CircuitOpen`，不再真实请求下游；HALF_OPEN 仅放少量 Probe。窗口只统计网络失败、Timeout、可用性 `5xx` 与慢调用；参数错误、权限拒绝、用户取消、测试失败和模型调错 Tool 都不计入。一次逻辑 Tool Call 即使内部 Retry 多次，最终也只记录一次成功或失败，避免人为放大失败率。多实例时将窗口和状态置于 Redis / 共享存储并原子更新。

**Agent 全局熔断**解决的是 ReAct Loop 自己不收敛：

```text
Tool A → Tool B → Tool C → Tool A → ...
```

Run 层维护 `max_turns`、`max_tool_calls`、`max_wall_time`、`max_tokens / max_cost`、`max_subagents` 与用户取消信号。任一 Budget 达到即停止进入下一轮 Model Call，返回 `RunBudgetExceeded`。同时可检测 `tool_name + arguments_hash + result_hash` 的重复模式：重复 N 次先返回 `RepeatedToolCall` 让模型 Replan；仍无进展则由全局 Budget 强制停止。

> **Local Breaker 防止依赖拖死 Agent；Global Breaker 防止 Agent 自己拖死自己。**

### 约束解码与服务端校验各解决什么

```text
Tool JSON Schema
→ Grammar / FSM 约束解码
→ JSON 解析
→ Schema 校验
→ 业务不变式 / 资源存在性校验
→ Policy / Permission
→ Execute
```

约束解码在生成时屏蔽不可能组成合法 JSON 的 Token，可保证结构、枚举、字段类型等部分约束；它不能保证路径存在、SQL 安全、金额合理或用户有权操作。因此 Schema 必须尽量小而强（`required`、`enum`、`additionalProperties: false`、范围 / pattern），危险动作最好拆为“计划 / 预览”和“确认执行”两个 Tool，服务端校验永远不能省。

最终，Tool Result 必须带回原始 `tool_call_id`，并区分成功、参数错误、策略拒绝、可重试失败、超时取消和副作用未知等状态。Trace / Ledger 还应记录 policy、deadline、attempt、`idempotency_key` 与错误类别，才能解释“为什么没重试、为什么被拒绝、下次从哪里恢复”。

# 四、Context Engineering

Context Engineering 解决的核心问题是：

> 每次模型调用时，应该让模型看到哪些信息，以及如何在有限的 Token 预算内组织这些信息。

Context 不是持久状态本身，而是 Harness 从各类状态与数据源中构建出的**本轮模型输入**。

## 4.1 Context Builder：从状态和工具调用结果构造本轮输入

### Context 只保留“当前决策所需”，其余留在外部事实源

长任务中最危险的做法，是把“系统知道的一切”都拼进 Prompt。窗口有限，内容越多，Prefill 越慢、注意力越稀释；即使窗口足够大，后续压缩也可能改变历史 Token。因此 Context Builder 应遵循 **引用优先、按需展开、结果回写**：模型 Context 只放当前一步确实需要的信息，完整事实留在可查询的外部来源。

| 外部来源 | 保存什么 | 何时进入 Context |
|---|---|---|
| Thread State DB | Tool / Task 状态、已完成步骤、重试次数、产物引用 | 当前决策依赖执行进度时，投影结构化摘要而非完整日志 |
| Workspace / Artifact Store | 文件、命令输出、大 Tool Result、截图与原始证据 | 模型通过 `read_file` / `artifact_read` 按需读取片段；Context 保留路径、摘要与 artifact ID |
| Memory / RAG | 用户偏好、稳定决策、知识库证据 | 由 Harness 或模型 Tool 检索少量高相关内容 |
| Subagent Run | 最终结果、中间 Observation、失败与风险 | 主 Agent 只接收最终结论、证据/产物引用、风险与未解决问题 |

这也解释了为什么压缩不会等于“遗忘”：被压缩的是模型可见的消息表示，原始工具结果、持久化文件和 Artifact 仍是事实来源。需要核验细节时，Agent 应重新读取外部引用，而不是假设 Summary 能保存所有原文。

### Tool Result 归档后，下一轮仍要用怎么办

关键不是“归档后是否还能用”，而是区分 **Active Result** 与 **Archived Result**：

```text
刚完成、当前 Step / 后续 Task 明确依赖
→ Active / Pinned：保留完整结果或结构化关键字段在动态末尾

结果很长、后续可能需要但当前不必看原文
→ Archived：Context 保留 summary + artifact_id + retrieval_hint
             原文写入 Artifact Store

后续确实需要细节
→ artifact_read(artifact_id, query / range)
→ 仅回填当前决策需要的片段
```

Tool / Task 的依赖图和当前 Plan 决定哪些 Result 不能压缩：未被后续动作消费、尚处于 `WAITING` 的输入依赖，应标记为 `active/pinned`。已经消费、可通过引用重新读取的长结果才 Artifact 化。这样压缩丢掉的是重复原文，不是任务事实；模型需要证据细节时，由 Context Builder 或 `artifact_read` 精确 Rehydrate，而不是将整份 80KB Result 永久携带在 messages 中。

## 4.2 上下文三层结构

来源经过 Context Builder 选择后，最终模型输入应按**物理 Token 布局与变化方式**分成三段：

| 层次 | 典型内容 | 变化方式 | 处理策略 |
|---|---|---|---|
| 固定前缀 | System/Developer 指令、少数高频 Direct Tool JSON（如 `tool_search`、`load_skill`）、长期有效约束 | 全程不变 | 字节、排序和序列化均固定，形成可复用 Prefix |
| 中间历史 | Conversation History、已完成 Tool Call/Result、阶段决策与旧证据 | 正常只在尾部 append；超窗时低频压缩封闭片段 | 主要压缩对象；分段摘要、Artifact 化，并冻结压缩版本 |
| 动态末尾 | 本轮 User Query、本轮 Tool Result、当轮检索内容 | 每轮追加 | 保留原文，保证局部推理、指代和纠错信息完整 |

```text
┌──────────────── 固定前缀：System + Core Tool JSON，始终不变
├──────────────── 中间历史：Message 列表顺序追加，封闭后才低频压缩
└──────────────── 动态末尾：本轮 Query / Tool Result / Retrieval，每轮新增
```

其中中间历史的本体是按时间追加的 `messages[]`：每一项都是 `message { role, content }`。用户输入为 `role=user`；模型输出 Tool Call 时为 `role=assistant` 的结构化 `content`；执行结果以带原始 `tool_call_id` 的 `role=tool` 消息写回。因此 Tool 调用不是脱离对话历史的一条旁路，而是同一条消息序列中的一个闭环。

这三段只是模型输入 Context 的**堆栈布局**；这种布局同时服务两件事：长上下文中常见的“U 型注意力“ 即首尾更容易被模型利用现象，以及 Prefix Cache 命中。

Thread State、State DB、Workspace / File System、Artifact Store、Memory 与 Retrieval 是 Agent 可依赖的外部来源。Context 应采用“**外部引用 + 按需加载**”而不是全量载入上下文：完整文件、大 Tool Result、事件记录和检索原文持久化在外部，模型当前只保留路径 / `artifact_id`、摘要、证据引用或少量高相关片段；需要细节时再通过 `read_file`、`artifact_read`、RAG 或状态查询取回。



![Agent Context Engineering：分层输入、外部事实源与 Prefix Cache](figures/agent-context-engineering-architecture.png)

窗口预算不能用字符数估算，必须使用目标模型 Tokenizer，并预留输出、Tool Schema 变化和 safety margin：

```text
input_budget = context_window
             - reserved_output_tokens
             - tool_schema_budget
             - safety_margin
```

对过大 Tool Result，原文写入 Artifact Store，Context 中仅保留结构化摘要、关键证据和可再读取的 artifact ID，避免每轮重复携带。

## 4.3 Agent 上下文剪裁与压缩：Micro → Auto → Reactive

长任务不能无限追加全部历史，但“超过阈值就把所有消息总结一次”太粗糙。更接近 Claude Code 风格的设计，是把 Context 管理分为三类问题：

> **Observation 太多，Conversation 太长，或 Context 已经溢出。**
> 它们分别用低损耗剪裁、语义压缩和异常截断处理。

无论哪种机制，主要目标都是三层中的**中间历史消息层**：固定前缀保持字节与顺序不变，动态末尾保留最近原文；完整 Tool Result、文件与事件始终留在 Artifact / Event Store 中，而不是依赖 Summary 当事实源。

```text
每轮 Model Call 前
  → Micro Compact：去掉可重新获取的旧 Observation
  → 正常继续 append History
  → 达到 Auto Compact 水位
      → 优先 Session Memory 替换旧 History
      → 不足则 Legacy LLM Summary
  → 若突增导致 Prompt Too Long
      → Reactive Compact 紧急截断并重试
  → Post-Compact Cleanup：修正 Runtime 与 Context 标记
```

![Agent 上下文压缩的渐进式机制](figures/agent-context-compaction-lifecycle.png)

### Micro Compact：日常低成本剪裁，不调用 LLM

每轮 Model Call 前，Harness 都可以检查旧的 Tool Result 是否仍值得占用 Context。`Read`、`Grep`、`Glob`、`Bash`、Web Search 等 Observation 往往很大，但其原文已经在文件系统、Artifact Store 或外部服务中可重新获取。Micro Compact 只把这类**低价值原始 Observation**替换为简短占位、结构化摘要或 artifact ID；它不总结任务语义，也不改写 Goal、决策和最近推理。

```text
旧 read_file(file=app.py, 8000 tokens)
        ↓
Context 中保留："已读取 app.py 的鉴权逻辑；artifact://run/42/file/7"
原始内容保留：Artifact Store / Workspace
```

实践中可分两条路径：

- **Cache-aware Micro Compact**：若 Provider 支持 Prompt Cache 编辑或服务端缓存引用，优先在缓存层标记 / 替换旧 Observation，尽量不改客户端完整 History，从而保留已有 Prefix / KV Cache；
- **Time-based Micro Compact**：若长时间没有交互、Prompt Cache 已自然失效（例如超过约 60 分钟），再直接改写本地 History 中的旧 Observation。此时没有必要为了旧 Cache 保留大段无用原文。

所以 Micro Compact 的本质是：**删 Raw Observation，不做语义总结。**它成本低、可以高频执行，但只能延缓增长，不能替代真正的会话压缩。

### Auto Compact：接近窗口时主动重构会话

当输入 Token 达到 Auto Compact 水位（例如预留输出与 Tool Schema 后超过可用预算的 80%），才触发宏观压缩。压缩目标应带滞回：例如从 80% 压回 50%～60%，避免每轮都在阈值附近重复压缩。

优先采用两级策略：

1. **Session Memory Compact**：Agent 正常运行时，后台异步 Subagent 可以持续从Context / Event / State 中提取稳定记忆：用户目标与约束、关键决策、已验证事实、当前进度、失败路径和 Artifact 引用等。生成并持久化为版本化 Session Memory。触发压缩时，直接用 Session Memory 替换较早 Chat History，最近 Messages 保持原文；压缩当下不必临时读取全部历史再调用模型。它并非 Zero-LLM，而是把 LLM 成本从阻塞路径前移。
2. **Legacy Compact 降级**：若 Session Memory 缺失、提取失败或释放空间不足，才对封闭的远端 Conversation 调用模型生成结构化摘要，并重构为：

```text
[Session Memory / Historical Summary]
  + [Compact Boundary：覆盖的 Event Range、版本、Artifact 引用]
  + [Recent Messages 原文]
```

通用 Agent 的摘要字段应能覆盖目标、领域语义、工作区记录、问题进展及后续动作；其中 Workspace 不只指 Coding Agent 的文件，也包括 Artifact、Tool / Task 状态、DB 记录和其他外部资源。一份统一的九段式摘要可采用以下结构：

```yaml
primary_request_and_intent: 用户的核心需求、意图与验收条件
key_domain_concepts: 涉及的领域知识、术语与关键概念
workspace_records: 文件、代码片段、修改记录、Artifact、Tool / Task 状态、DB 或外部资源引用
errors_and_fixes: 遇到的错误、失败尝试与修复方案
problem_solving: 已解决与仍在排查的问题、关键决策及理由
all_user_messages: 所有非 Tool Result 的用户消息原文及其 Event / Message 引用
pending_tasks: 待完成任务
current_work: 压缩前正在进行的工作
optional_next_step: 可选的下一步计划
```

#### 独立模型压缩的 XML 输出契约

Legacy Compact 不应只要求“总结以上对话”。独立的压缩模型需要输出可审计、可恢复的结构化结果；可约定返回两个 XML 块：

```xml
<analysis>
  <!-- 按时间顺序审计消息：用户意图、领域概念、工作区变化、问题与修复、任务进展及外部引用。 -->
</analysis>
<summary>
  <!-- 可直接替换封闭 History 的正式摘要。 -->
</summary>
```

- **`<analysis>` 块**：压缩任务的工作草稿，按时间顺序核对每条 Message，提取用户意图、领域概念、工作区变化、问题与修复、任务进展及外部引用。生产环境不应将其当作模型私有推理链长期持久化，只保留对任务恢复有价值的结构化审计信息即可。
- **`<summary>` 块**：正式、稳定的会话摘要，供 Context Builder 在后续 Turn 中加载。字段严格遵循上方统一的九段式 YAML Schema，避免摘要随模型风格漂移。


关键原则是：**远端历史做 Summary，最近执行上下文保留原文；需要细节时按 artifact / state 引用重新读取。**压缩后需检查系统约束、Goal、未完成 Tool / Task 状态与关键证据引用是否仍存在。

### Reactive Compact：Prompt Too Long 的异常兜底

Auto Compact 也可能来不及。例如一次 `Read` 返回超大文件，或检索结果突然暴涨，直接触发 `413 Prompt Too Long`。此时目标不是生成高质量摘要，而是先让 Agent 从 Context Overflow 中恢复：优先裁掉最旧、可重新获取的 API rounds / Tool Results，保留 System、当前 Goal、最近 Messages 与未完成状态，然后重建请求。

```text
Model Request → 413 Prompt Too Long
      ↓
按优先级紧急剔除：旧 Raw Observation → 旧 API Rounds → 低相关 Retrieval
      ↓
保留：System / Active Tool / Goal / Running Task / Recent Messages
      ↓
重新发起 Model Call
```

Reactive Compact 通常不等待后台 Subagent，也不应再阻塞调用 LLM 做长摘要：它延迟低、信息损失最大，是最后一道故障恢复，而不是常规压缩策略。

### Post-Compact Cleanup：防止“模型忘了，Runtime 还以为记得”

压缩改变的是 Model Context，不会自动改变 Runtime State。比如 Runtime 记录“`foo.py` 已读取”，但对应 Result 已被从 Context 剪掉；若仍阻止模型再次读取，下一轮就会出现状态断裂。

因此压缩后应清理或降级与可见上下文绑定的标记，例如 `read_file_state`、过期的状态标记，并保留 State DB 中真实的 Tool / Task 事实。后续模型若需要详情，可以安全地重新调用 `read_file` / `artifact_read`；不要让 Runtime 假设模型仍记得被裁剪的 Observation。

### Context Compression 与 Prefix Cache 的协同

冲突的根源是：历史压缩会改写 Token，而 Prefix Cache 只能复用从开头连续一致的 Token。若每轮都重新总结整段会话，摘要后面的 Token 即使没有变化，也会因前缀断裂而全部失去命中。

解法不是放弃压缩，而是让三段承担不同职责：固定前缀永不因历史压缩而变化；中间历史只在越过水位后低频生成新版本；动态末尾保持原文并持续追加。

```text
[固定前缀：System + Core Tool JSON]                    ← 长期命中
[中间历史：summary_vN + 未压缩的封闭消息]              ← 低频变化
[动态末尾：本轮 Query + Tool Result + Retrieval]       ← 高频追加
```

具体策略：

1. **前缀不变性**：系统指令、Tool Schema 的字节内容和排序都要确定，不注入时间戳、随机 ID 等动态字段；
2. **分段压缩**：只压缩一个已封闭的旧历史段，不每轮重新摘要整段会话；
3. **摘要冻结**：生成 `summary_vN` 后保持不变，直到下一次跨越 hard waterline 才生成 `summary_vN+1`；
4. **块级 Cache**：新摘要会使变化点之后的 KV 失效，但 PagedAttention + vLLM APC 仍可保留和复用变化点之前连续不变的完整 Block；
5. **低频压缩**：使用水位和滞回区间，例如超过 80% 时压到 55%，避免在阈值附近每轮抖动；
6. **按实测优化**：同时记录 compression latency、压缩后输入 Token、prefix cached tokens 和任务质量，而不是只追求命中率。

PagedAttention 为 KV Block 的独立保留、引用和回收提供存储基础，vLLM APC 则用 Block Hash 查找可复用前缀。例如压缩前为 `Prefix + H1 + H2 + Tail`，压缩后变为 `Prefix + summary_v1 + Tail`：第一次压缩后，APC 仍可复用 `Prefix`；`summary_v1` 冻结后，后续轮次又能复用 `Prefix + summary_v1`。变化后的摘要与旧历史 Token 不同，其 KV 仍须重新计算。

最终目标不是让压缩永远不产生 Cache Miss，而是让大部分轮次只在最近层追加；跨越水位时只改写历史层并承担一次局部重算，随后冻结新摘要，重新建立可持续命中的前缀。

**常见问题**

1. **为什么 Context 不是越多越好？**
   过多内容会增加 Prefill 延迟，并带来注意力稀释、信息重复和指令冲突。
2. **压缩与 Prefix Cache 的根本冲突是什么？**
   压缩会重写历史 Token，而 Prefix Cache 只能复用从首 Token 开始连续一致的部分。因此稳定层必须固定，主要压缩封闭的历史消息层，最近轮次保留原文；摘要生成后再冻结多轮复用。
3. **怎么评估压缩方案？**
   同时看任务成功率、关键事实保留率、压缩延迟、输入 Token 和 cached tokens，不能只看压缩比。


## 4.4 Agentic RAG：检索、证据判断与补搜闭环

普通 RAG 是“检索一次，再让模型回答”；**Agentic RAG** 则把检索作为原有 ReAct Loop 中的一类 Tool Use：先定义需要检索的内容和验收标准，再根据证据缺口决定是否补搜、改写 Query 或停止。它没有另一套 Agent Runtime；解决的是“当前证据是否足以支撑交付”。

```text
Run 创建：Search Spec（关键 Claim、验收条件、检索 Budget）
    ↓
原 ReAct Loop：Model → Search / RAG / Browse Tool Call → Tool Result
    ↓                         └→ Evidence：抽取片段；全文归档 Artifact
模型依据 Evidence 决定补搜、改写 Query 或发起 verify_evidence
    ↓
Evidence Check
    ├─ 缺少证据 / 存在冲突 → 标准 Observation 回到同一个 ReAct Loop
    └─ PASS → 允许模型生成带引用的最终答案
```

### 4.4.1 检索底座：先保证“召得到”，再保证“答得对”

索引的最小单元应是带元数据的 Chunk：优先按章节、段落、表格或代码结构切分；Chunk 内保留标题、层级路径、版本、时间、权限与 `artifact_id`。需要更完整上下文时，可用 Parent–Child：小块负责召回，父块负责阅读。

- **BM25** 擅长专有名词、数字、缩写、代码符号等精确匹配；
- **Embedding** 擅长同义改写和语义近邻；
- **RRF** 只融合排名、不直接相加两路异构分数，适合作为稳定的第一阶段融合；
- Rerank / LLM 判别只处理有限候选，避免把全库内容直接送进模型。

```python
from collections import defaultdict

def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for docs in rank_lists:
        for rank, doc_id in enumerate(docs, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

async def hybrid_retrieve(query: str, scope: dict, top_k: int = 12):
    # 权限和元数据过滤在每一路检索前执行，不能先召回后过滤。
    sparse, dense = await gather(
        bm25.search(query, filters=scope, limit=40),
        vector.search(embed(query), filters=scope, limit=40),
    )
    candidate_ids = rrf([ids(sparse), ids(dense)])[:30]
    docs = await document_store.get_many(candidate_ids)
    return await reranker.rank(query, docs, top_k=top_k)
```

召回阶段优化 Recall@K，目标是“不漏掉”；精排或判别阶段才优化 Top-1 / NDCG，目标是“把正确证据放在前面”。候选集中没有答案，后续大模型也无法补救。

### 4.4.2 Agentic Search：原 ReAct Loop 中的 Search Spec 与 Evidence Gate

关键不是让 Agent 无限 Search，而是在 Run 开始时写入 `Search Spec`：关键子问题、每个关键 Claim 的证据要求、是否需要独立来源交叉验证、来源时效/可信等级，以及轮数、查询数、时间与成本上限。它属于当前 Run 的 State，Context Builder 每轮只投影当前缺口和相关 Evidence。

接下来仍是第 2 章的原 ReAct Loop：模型调用 `search_web`、`retrieve_kb`、`browse_page` 等 Tool；Tool Executor 按正常状态机执行，结果由 Evidence Store 提取片段并归档。模型读取本轮 Observation 后，自主决定继续补搜、改写 Query，或调用固定的 `verify_evidence` Tool。

```python
async def verify_evidence(call, context):
    # 这是 ReAct Loop 中一次普通的内部 Tool Call，不是嵌套 Agent Loop。
    spec = await state_store.get_search_spec(context.run_id)
    evidence = await evidence_store.for_claims(context.run_id, spec.claims)
    args = json.loads(call.arguments)
    verdict = await verifier.check(
        spec=spec,
        candidate_answer=args["candidate_answer"],
        evidence=evidence,
    )
    # pass / missing_claims / conflicts / insufficient_sources
    return tool_result(call.tool_call_id, verdict.to_observation())
```

`verify_evidence` 可以由主 Agent 自评估，也可以在高风险场景交给独立 Verifier。它输入的是 Search Spec、Candidate Answer、Claim–Evidence 映射和 Artifact 引用，输出的是结构化缺口，而不是一句“我觉得已经够了”。规则校验（检索 Budget、来源数量、时效）必须由 Harness 硬控制；Verifier 只负责评估证据是否支持结论、冲突是否已解释。

```text
verify_evidence = PASS
    → 模型继续原 ReAct Loop 的最终回答分支

verify_evidence = NEED_MORE_EVIDENCE / CONFLICT
    → Tool Result 写回 messages[]
    → 模型在下一轮 ReAct 中改写 Query、换 Search Tool 或交付 PARTIAL
```

为避免模型跳过校验直接宣称完成，研究类 Run 可配置 Finalization Guard：未取得有效 `PASS` 时，Harness 不直接发出最终响应，而是要求先完成 `verify_evidence`；若 Search Spec 的 Budget 已耗尽，则允许带缺口 / 冲突说明的 PARTIAL 结果。预算控制仍复用 Run 的 `max_turns`、`max_tool_calls`、时间与成本上限，而不是新建一个 Search while-loop。

### 4.4.3 Evidence 如何进入 Context

原始网页、长文档和完整 Tool Result 不应长期塞进 `messages[]`。`Evidence Store` 保存正文、快照和版本；Context Builder 只选择“与本轮 Claim 最相关的片段 + 来源引用 + 必要元数据”。若下一轮需要细节，再用 `artifact_read(artifact_id, range)` 精确回读。

因此：**Artifact 是原始证据的事实来源，Context 是当前推理所需的工作集。** 这也使压缩后仍可回溯原文，而不是只能依赖摘要。

### 4.4.4 怎么评估

| 层 | 关键指标 | 说明 |
| --- | --- | --- |
| 检索 | Recall@K、MRR | 正确证据有没有进入候选 |
| 精排 | NDCG、Top-1 Acc | 有限候选能否排对 / 判对 |
| Agentic Search | Claim 覆盖率、有效补搜率、Verifier 通过率 | 能否正确发现证据缺口并收敛 |
| 生成 | Faithfulness、Citation Accuracy | 输出是否真的受到证据支持 |
| 系统 | P95、空召回率、索引新鲜度 | 是否足够快、稳定、及时 |

## 4.5 Memory：从对话中沉淀可复用事实，而不是保存全部历史

Memory 不等于 Context，也不能替代 Runtime State。

```text
Memory        = Agent 未来还应记住的偏好、决策、反馈与经验
External State = 文件、Artifact、Tool / Task 状态等真实世界事实
Context        = Context Builder 本轮选择给模型看的工作集
```

例如“部署已成功”必须以 Deployment Tool 的状态或外部 `task_id` 为准；Memory 最多记录“这次采用了哪种部署策略、对应的证据在哪”，不能把模型的猜测当成事实。

### 4.5.1 分层与写入边界

| 层 | 存什么 | 什么时候读取 |
| --- | --- | --- |
| **Instruction Memory** | 长期规则、项目规范、权限边界 | 每个 Session 少量自动注入 |
| **Long-term Memory** | 用户偏好、反馈、项目决策、经验、参考入口 | 结合 Query / Task 按需检索 |
| **Session Memory** | 当前 Goal、进展、关键决策、失败路径、下一步 | 长任务续跑和 Context Compact |
| **External State** | Workspace、Artifact、Ledger、Task / Tool 状态 | 需要核验或读取原文时按引用访问 |

长期 Memory 只收录“跨 Session 有价值、重新获得成本高、来源可追溯”的信息，例如用户明确偏好、已确认的项目决策和高价值反馈。完整聊天记录、可重新查询的 Tool Result、临时日志、文件正文及模型猜测都不应直接写入。

### 4.5.2 Mem0 风格的写入范式：提取候选，再决定新增还是更新

Mem0 的典型思路不是把一轮消息原样塞进向量库，而是先让模型从对话中提取值得记住的事实，再以 Scope 和语义相似度定位已有 Memory。生产系统还需要在应用层补上冲突决策：新增、更新、标记旧记录失效，或直接忽略。

```text
Closed History / 用户显式“记住…”
    → Extract：抽取 preference / decision / feedback / reference 候选
    → Guard：Scope、ACL、隐私、证据与写入价值校验
    → Search Similar：向量 + 关键词 + metadata 找相近记录
    → Decide：ADD / UPDATE / SUPERSEDE / IGNORE
    → SQL 事实记录 + Vector 索引 + 可选 Entity/Graph 索引
```

```python
async def write_memories(messages: list[Message], scope: Scope):
    # Harness 的后台 Worker 和模型显式 memory_write 都走同一条管线。
    candidates = await memory_extractor.extract(
        messages,
        schema="preference | decision | feedback | reference",
    )

    for item in candidates:
        if not worth_remembering(item) or not policy.can_write(item, scope):
            continue

        similar = await memory_store.search(
            query=item.content,
            filters={"scope": scope.chain()},
            top_k=5,
        )
        action, old = await memory_judge.decide(item, similar)

        if action == "ADD":
            await memory_store.insert(item, scope=scope, source_refs=item.sources)
        elif action == "UPDATE":
            await memory_store.update(old.id, item, source_refs=item.sources)
        elif action == "SUPERSEDE":
            await memory_store.supersede(old.id, item, source_refs=item.sources)
        # IGNORE：重复、临时信息、来源不足或不应存储的内容。
```

`SUPERSEDE` 处理的是“用户以前偏好 pytest，现在明确要求此项目用 unittest”这类冲突：新记录在更具体的 `project` Scope 生效，旧记录保留版本历史但不再参与默认召回。Mem0 的 `add` 路径可以是追加式的；对通用 Agent 而言，**冲突检测、版本与失效策略不能省略**，否则很容易把互相矛盾的偏好同时送回模型。

每条 Memory 至少包含内容以外的治理信息：

```yaml
memory_id: mem_123
scope: user | organization | project | task-session
type: preference | decision | feedback | reference
content: 当前项目使用 unittest
source_refs: [message_123]
evidence_refs: [artifact_77]     # 可选；结论需要追溯时使用
importance: 0.9
confidence: confirmed
version: 3
status: active | superseded | expired
expires_at: null
```

### 4.5.3 读取范式：小索引常驻，详情按需回读

不要把所有 Memory 全量注入 Context。读取应遵循 `Index → Relevant Memory → Raw Evidence`：先选少量高权威规则和索引，再按当前任务检索相关记忆；仍需细节时沿 `source_refs / evidence_refs` 读取 Artifact、Event Log 或 Workspace 原文。

```python
async def recall_for_turn(query: str, scope: Scope, token_budget: int):
    always_on = await instruction_store.get_active(scope)
    candidates = await memory_store.search(
        query=query,
        filters={"scope": scope.chain(), "status": "active"},
        top_k=20,
    )
    ranked = rank_by_scope_relevance_recency_importance(candidates, scope)
    selected = pack_to_budget(ranked, budget=token_budget)

    return {
        "rules": always_on,              # 小、稳定、高权威
        "recalled_memories": selected,   # 当前任务真正相关的少量记录
        "evidence_refs": refs(selected), # 需要时再用 artifact_read / state_query 回读
    }
```

`scope` 至少区分 `global / user / organization / project / task-session`，并让更贴近当前任务的记录优先。例如全局“中文回答”不能覆盖项目级“提交信息使用英文”。检索可融合 metadata filter、关键词、Embedding、实体、时间与重要度；但 Scope / ACL 必须在召回前过滤，不能让模型自己判断是否有权看到。

### 4.5.4 自动写入、主动读写与生命周期

- **Harness 自动读写**：每轮读取少量规则；后台 Worker 消费已封闭历史，异步更新 Session Memory 或提取长期候选。这是长任务最常见的路径。
- **模型主动读写**：通过 `memory_search` 找历史背景；只有用户明确要求记住，或出现稳定且已验证的结论时，才允许 `memory_write`。它们仍复用同一套校验和审计链路。
- **生命周期治理**：`Capture → Consolidate → Retrieve → Forget`。用去重、版本、TTL、低价值归档与删除请求抑制 Memory Pollution；每次更新都保留来源、时间和变更历史。

一句话总结：**Memory 负责把有复用价值的“认知”沉淀下来；State / Artifact 负责保存可核验的“事实”；Context Builder 在每一轮只取二者中真正需要的片段。**

# 五、Agent Harness Runtime 架构

Agent Loop 只描述一轮 `Model → Tool Call → Tool Result → Model`；**Agent Harness Runtime** 则是让这个 Loop 能持续、并发且可恢复运行的系统。它接住模型的动作意图，调度模型、Tool 与 Subagent，并在用户停止、超时、故障或恢复请求到达时决定该如何继续。

```text
                         ┌──────────── 控制面 ────────────┐
                         │ Thread / Run / Goal / Plan     │
                         │ Policy / Budget / Deadline     │
                         │ Cancel / Checkpoint / Resume   │
                         └──────────────┬─────────────────┘
                                        ↓
Context Builder → Model Executor → Harness Runtime ← Tool / Task Result Event
                                  │        │
                                  │        ├── Tool Executor → Local Tool / MCP / Sandbox
                                  │        └── Task Scheduler → Child Runtime / Subagent
                                  ↓
                       持久化事实层：State DB + Event Log + Artifact Store
                                  ↓
                       外部真实状态：Workspace / 下游服务 / 数据库
```

Runtime State 是其中的持久化控制面：Context 是模型当前视图，Memory 是可检索经验，**Runtime State 是当前执行事实**，而文件和下游系统才是外部副作用的最终事实源。下面只沿“**记录哪些实体 → 如何推进 → 如何控制**”展开。

## 5.1 Runtime 的核心实体：Thread 容器，Run 执行，实体关联

最容易混淆的是 Thread、Run 与 Turn：**Thread 是长生命周期的任务容器；Run 是一次从开始、Resume 到终止的具体执行；Turn 是 Run 内的一轮模型调用。** 一个 Run 可以包含多个 Turn 和 Step；Thread 可以保留多个历史 Run，例如用户修改需求后重新执行，或故障后从 Checkpoint 恢复。

```text
Thread 1 ── N Run 1 ── N Step / Turn
                   ├── N Tool Call
                   ├── N Task ── 0..1 Child Run（Subagent）
                   ├── N Artifact / Evidence Ref
                   └── 1 Environment / Policy Snapshot
```

```yaml
thread:
  thread_id: th_123
  status: running | waiting_tool | waiting_user | cancelling | cancelled | completed | failed
  cwd: /workspace/project
  active_run_id: run_456
  context_cursor: event_789            # 当前模型上下文对应的事件位置
  policy_snapshot_ref: policy_v7

run:
  run_id: run_456
  thread_id: th_123
  status: running | waiting_tool | waiting_task | cancelling | cancelled | completed | failed
  turn_id: turn_12                     # 当前 Turn；需要审计时可拆出 turns 表
  step: 5
  model: provider/model@version
  deadline: 2026-08-30T12:00:00Z
  token_usage: {input: 0, output: 0, cached: 0}
  cost_usage: {model: 0, tool: 0}
  goal: {objective: ..., acceptance_criteria: [...], status: active}
  plan: {version: 3, items: [...]}     # Goal / Plan 归属本次 Run

tool_calls:
  tool_call_id: call_001
  run_id: run_456
  step_id: step_5
  tool_name: web_search
  status: pending | running | succeeded | failed | timeout | cancelled
  attempt: 1
  deadline: 2026-08-30T11:20:00Z
  idempotency_key: idem_xxx
  arguments_ref: artifact://args/call_001
  result_ref: artifact://result/call_001

tasks:
  task_id: task_001
  dag_id: dag_001
  run_id: run_456
  parent_task_id: null
  status: blocked | ready | queued | running | succeeded | failed | cancelled
  remaining_deps: 2
  assignee: lead | subagent
  child_run_id: run_child_001
  input_ref: artifact://task-input/task_001
  output_ref: artifact://task-output/task_001

task_edges:
  from_task_id: task_001
  to_task_id: task_003
  type: required

artifacts:
  artifact_id: art_001
  type: tool_result | file | evidence | report
  uri: workspace://report.md
  version: 4
  digest: sha256:...

environment:
  workspace_ref: workspace://project@git-sha
  permissions: permission_profile_v3
  sandbox_profile: sandbox_v2
  tool_registry_version: tools_v18
  skill_catalog_version: skills_v6
```

Tool / Task 只保存状态、重试信息和 Artifact 引用，大参数、长 Result 与原始文件外置；Environment 以版本快照固定本次运行的 Workspace、权限与能力边界。实体记录带 `version`、`updated_at` 与终态标记，状态迁移采用条件更新，重复完成事件按 `tool_call_id` 幂等处理。

## 5.2 事件驱动执行：Event Log 是过程，State DB 是当前视图

```text
事件：ToolCallStarted / ToolCallSucceeded / TaskCreated / RunCancelled ...
                        ↓ append-only
                    Event Log
                        ↓ reducer / projector
State DB：threads、runs、tool_calls、tasks、artifact_refs、environment_snapshots
                        ↓
Checkpoint：run 状态快照 + context cursor / summary ref
```

Event Log 保存变化顺序、用于审计和 Replay；State DB 供 Runtime 快速查询当前在途 Tool、Task 与预算；Artifact Store、Workspace 和下游服务保存原始内容与真实副作用。状态表是事实来源：状态已终态但通知丢失时，后台异常扫描会补发通知或恢复对应 Run。

以一次模型输出多个 Tool Call 为例：

```text
Model 输出 Tool Calls
→ 事务内创建 ToolCall = PENDING + 写 ToolCallRequested Event
→ 投递 Tool Command
→ Executor 消费：ToolCall = RUNNING
→ 结果写入 Artifact Store：ToolCall = SUCCEEDED / FAILED / TIMEOUT
→ RPC Callback 或投递 ToolCompleted，触发对应 Step Join
→ 聚合该 step：未收敛则继续 WAITING_TOOL；全部终态才进入下一 Turn
```

状态机的职责是约束事实流转，而非替模型做决策：

```text
Run:       CREATED → RUNNING → WAITING_TOOL / WAITING_TASK → RUNNING → COMPLETED / FAILED
任一非终态 Run ───────────────────────────────────────→ CANCELLING → CANCELLED
Tool Call: PENDING → RUNNING → SUCCEEDED / FAILED / TIMEOUT / CANCELLED
Task:      QUEUED → RUNNING → SUCCEEDED / FAILED / CANCELLED
```

模型只产生动作意图；Executor / Task Runner 的终态事件才推进 Tool、Task 与 Run。**事件负责唤醒与回放，状态表负责当前判断。**

## 5.3 Runtime 控制面：Goal、Plan、Evidence 与预算

Runtime 不是让模型“自由循环”，而是给一次 Run 加上可执行的目标、约束和退出条件：

```text
Goal       本次 Run 的目标与验收条件
Plan       当前步骤和依赖关系
Evidence   已验证的结果及其 Artifact / 外部引用
Budget     deadline、max_turns、token / cost、并发上限
Policy     权限、审批、Sandbox 与允许使用的能力版本
```

Goal / Plan 属于 Run，而不是 Thread。每轮 Model Call 前，Context Builder 从 State DB 读取当前 Run 的最新 Goal、Plan、预算与关键约束，作为系统状态块注入模型输入；模型更新计划或声明完成时，仍须经 Harness 校验并原子写入 Event Log / State DB。

模型说“已经完成”不等于 Run 已完成。完成或阻塞必须关联 Evidence，例如测试退出码、报告、文件 Diff 或下游资源 ID。Runtime 据此判断是否满足验收，而不是把自然语言当成事实。

### 完成门禁：Task Spec → 自评估 → Verifier 交叉验证

防止 Agent 幻觉的重点不是要求模型“永不出错”，而是禁止模型的自然语言声明直接改变任务状态。

1. **Task 创建前定义验收标准**：Goal / Prompt 不只描述要做什么，还要给出可验证的 `completion_contract`。例如报告必须覆盖指定子问题、关键结论具有引用；代码必须存在目标文件、测试通过；外部操作必须有下游资源 ID 或可查询终态。
2. **主 Agent 自评估**：执行中，模型对照验收项检查自身产物的覆盖、证据与缺口，未满足时重新规划、补搜、重试或明确标记不确定性。自评估是软判断，不能单独把 Run 置为 `COMPLETED`。
3. **Verifier 独立交叉验证**：确定性部分优先由规则 / Tool 校验，例如 Schema、测试退出码、Artifact 是否存在、状态表和下游资源状态；语义部分再由 Verifier Agent / Judge Model 检查 Claim–Evidence 对齐、结论完整性和未处理冲突。Verifier 返回 `PASS`、`NEED_MORE_EVIDENCE`、`PARTIAL` 或 `CONFLICT`。

```text
模型说“完成”
→ 收集 Tool Result / Artifact / Evidence
→ 规则校验 + Verifier
   ├─ PASS：Runtime 才原子写入 COMPLETED
   ├─ NEED_MORE_EVIDENCE：继续 Agent Loop
   └─ PARTIAL / CONFLICT：不伪装完成，显式交付缺口
```

因此，**模型负责提出行动和解释；Runtime、Tool、Artifact 与 Validator 负责记录真实发生的事实；只有验收门禁通过，系统才宣称任务完成。**

### Checkpoint、取消与可观测性

Checkpoint 是稳定边界的 **Runtime State Snapshot + Context Refs**：保存 Tool / Task / Goal / Plan / Budget、Artifact / Environment 引用、`context_cursor` 和当前摘要版本。它不复制完整对话；历史已压缩时，Runtime 仍可由 Cursor、摘要和 Artifact 引用重建下一轮工作集。

```text
User Stop → Thread / Run: CANCELLING
          → 取消模型流、Tool、子进程与 Subagent
          → 执行单元写入 CANCELLED，禁止下一轮 Model Call

Resume / Failover → 加载 Checkpoint
                  → 复用终态结果；核验在途 Tool / Task 的真实状态
                  → 从最后一个确定边界继续 Run
```

对于副作用调用，`RUNNING` 或状态未知不能直接重放：先用 `tool_call_id`、幂等键和下游 `task_id` 核验。每个事件统一关联 `thread_id → run_id → turn_id → tool_call_id / task_id`，由 Trace 记录延迟、Token、结果引用和错误分类，支撑调试与故障接管。

## 5.4 Task Scheduler：将 Subagent 作为 Child Run 调度

Multi-Agent 不是另一套运行时。先分清 **Plan Tool** 与 **Task Tool**：Plan 保存模型对目标、步骤和策略的语义规划，可以反复修改，并不直接触发执行；Task Tool 才创建可执行的 Task、依赖边和 Task DAG，并交给 Scheduler 调度。因此 Plan 与 Task 不要求一一对应。

Task 可以作为一个 Tool 被 Lead 调用；Scheduler 再为它派生受限的 Child Run。Tool 与 Child Run 共用状态、事件、超时、取消和预算机制。

每个 `Task` 至少记录：

```text
task_id / dag_id / parent_task_id / status / remaining_deps
instruction / expected_output / input_ref / output_ref / child_run_id
```

### Task DAG：拓扑校验一次，执行时按事件释放后继

Task Tool 创建 DAG 时先做一次 Kahn 拓扑排序：校验不存在环、计算初始入度，并将每个节点的当前入度持久化为 `remaining_deps`。运行时不重复全图排序；Task 完成就等价于从图中删除一个节点，其后继 Task 的依赖计数减一。

```text
Task A ─┐
        ├→ Task C (remaining_deps = 2)
Task B ─┘

Task A 完成 → C.remaining_deps = 1，C 仍 BLOCKED
Task B 完成 → C.remaining_deps = 0，C 进入 READY 并投递 MQ
```

### 如何入队、消费与推进 DAG

初始 `remaining_deps = 0` 的节点直接进入 `READY` 并投递 `TaskReady`：

```text
Task Tool 创建 Task / Edge
→ remaining_deps = 0：BLOCKED → READY
→ MQ: TaskReady { task_id, dag_id, input_ref, task_type }
→ Subagent Worker 消费 TaskReady
→ 条件更新 READY → RUNNING
→ 创建 Child Run 并执行
```

下游 Worker 不需要自己查询前置任务；**收到 `TaskReady` 就代表依赖已经全部满足**。重复消息通过 `READY → RUNNING` 的条件更新幂等处理。

Subagent 完成后，Worker 持久化 `output_ref` 与终态，并发送：

```text
TaskCompleted { task_id, dag_id, status, output_ref }
```

Scheduler 消费该事件，沿出边更新后继节点：

```text
TaskCompleted(A)
→ 查询 A 的后继 Task
→ 每个后继 remaining_deps - 1
→ 计数归零：BLOCKED → READY
→ 投递 TaskReady(B)
→ B 的 Subagent Worker 开始消费
```

因此 Scheduler 就是 Kahn 拓扑排序的事件驱动版本：**完成事件等价于删除节点，依赖归零等价于节点入队。** `TaskCompleted` 可按 `dag_id` 有序消费，`TaskReady` 则按 `task_id` 分发给多个 Worker，以同时保证 DAG 状态推进有序和下游执行并发。

Python 版可将“消费执行”和“完成后释放后继”拆成两个消费者。这里的 `complete_and_release_once` 必须在一个事务内完成：去重 `event_id`、写入终态、扣减后继 `remaining_deps`、将归零节点转为 `READY`。

```python
async def task_worker(store, mq, run_child):
    async for event in mq.consume("task-ready"):
        # 条件更新：重复 TaskReady 不会重复创建 Child Run
        if not await store.claim_running(event.task_id):
            continue

        try:
            output_ref = await run_child(event.task_id)
            completed = TaskCompleted(event.task_id, "SUCCEEDED", output_ref)
        except Exception as exc:
            completed = TaskCompleted(event.task_id, "FAILED", error=str(exc))

        await store.finish_task(completed)
        await mq.publish("task-completed", completed)


async def dag_scheduler(store, mq):
    async for event in mq.consume("task-completed"):
        if event.status != "SUCCEEDED":
            await store.apply_failure_policy(event)
            continue

        # 原子去重 + 扣减依赖 + BLOCKED → READY
        ready_tasks = await store.complete_and_release_once(event)
        for task in ready_tasks:
            await mq.publish("task-ready", TaskReady(task.id, task.dag_id))
```

Go 版的职责相同；MQ 可替换为 Kafka、队列或 Stream，关键仍是 `ClaimRunning` 与 `CompleteAndReleaseOnce` 的条件更新和事件去重。

```go
func TaskWorker(ctx context.Context, store TaskStore, mq MQ, runChild RunChild) error {
	for event := range mq.ConsumeTaskReady(ctx) {
		if !store.ClaimRunning(event.TaskID) { // READY → RUNNING；重复消息直接跳过
			continue
		}

		outputRef, err := runChild(ctx, event.TaskID)
		completed := TaskCompleted{TaskID: event.TaskID, Status: "SUCCEEDED", OutputRef: outputRef}
		if err != nil {
			completed = TaskCompleted{TaskID: event.TaskID, Status: "FAILED", Err: err.Error()}
		}
		if err := store.FinishTask(completed); err != nil {
			return err
		}
		if err := mq.PublishTaskCompleted(ctx, completed); err != nil {
			return err
		}
	}
	return nil
}

func AdvanceDAG(ctx context.Context, store TaskStore, mq MQ) error {
	for event := range mq.ConsumeTaskCompleted(ctx) {
		if event.Status != "SUCCEEDED" {
			if err := store.ApplyFailurePolicy(event); err != nil { return err }
			continue
		}

		// 事务内：event 去重、remaining_deps--、BLOCKED → READY
		readyTasks, err := store.CompleteAndReleaseOnce(event)
		if err != nil { return err }
		for _, task := range readyTasks {
			if err := mq.PublishTaskReady(ctx, TaskReady{TaskID: task.ID, DAGID: task.DAGID}); err != nil {
				return err
			}
		}
	}
	return nil
}
```

### Lead 如何等待整个 DAG

Lead 创建 DAG 时同时记录 `TaskJoin { join_id, dag_id, lead_run_id, lead_step_id, wait_policy }`，然后自身进入 `WAITING_TASK`。Scheduler 每次处理 `TaskCompleted` 都检查 Join 条件，例如“所有 required Task 均终态”。满足时发送：

```text
TaskJoinCompleted { lead_run_id, lead_step_id, join_id }
→ RPC Callback 或 MQ 通知 Harness
→ Lead 查询 task_id → status + output_ref + evidence_ref
→ 继续下一轮推理、补充任务或交付
```

调度时，Runtime 为 Child Run 注入**聚焦任务、必要背景、受限 Tool 和最小 Workspace 权限**，而不是复制 Lead 的完整 Context；局部检索、冗长 Tool Result 与失败尝试留在子任务内，实现 Context 隔离。

关键 Task 失败时可按 `failure_policy` 让后继节点 `SKIPPED`、局部重试，或允许其他支路继续并以部分结果恢复 Lead。正常路径由 `TaskCompleted / TaskJoinCompleted` 事件推进，不轮询；后台心跳只处理超时、失联或通知丢失。主 Run 始终保留 Goal / Plan、最终写入权与交付决策。

## 5.5 Environment：Workspace、权限与 Sandbox

Environment 是 Runtime 可执行边界的一部分，必须和 Run 一起版本化：

```text
workspace_ref       当前目录、文件、Git / Worktree 与 Artifact 位置
permissions         可读写路径、允许的网络域名、凭证范围
sandbox_profile     进程、CPU / 内存、执行时长、输出大小等硬限制
tool / skill version 本次 Run 可调用的能力集合
```

Tool 通过受控 Workspace 接口读取、执行和写入，而不是直接操作宿主机。Policy / Approval 决定“某次操作是否允许”，Sandbox 则强制执行文件、网络、进程、资源和凭证边界；主 Run 与 Child Run 可使用不同权限或独立 Workspace。生产环境可采用容器、MicroVM 或远程隔离执行，但选择何种部署不是 Runtime 的核心，核心是**状态中记录了实际能力边界，执行层无法越界**。

## 5.6 Runtime 的关键控制条件

| 运行风险 | Runtime 如何控制 |
|---|---|
| 外部副作用完成但响应丢失 | Tool Ledger + 幂等键；恢复时先核验下游真实状态，不能盲目重放 |
| 用户 Stop 后后台仍执行 | `CANCELLING → CANCELLED`，取消信号传播到模型、Tool、子进程与 Subagent |
| Tool / Provider 暂时故障 | 按错误类型重试或熔断；终态错误作为 Tool Result 回给模型重新规划，必要时从 Checkpoint 恢复 |
| Agent 轮数、成本或子任务失控 | `max_turns`、deadline、Token / cost、并发和递归深度预算阻止继续调度 |
| Context 变长后遗忘约束 | Goal / Policy 固定注入；长 Result Artifact 化，Context 可压缩但 Runtime State 不丢失 |
| 外部内容诱导危险操作 | 外部内容视为不可信数据；Policy + Approval + Sandbox 三层约束，敏感目录和凭证不暴露给模型 |

> **一条主线：模型产生意图，Runtime 用状态和策略决定是否调度；Executor 用真实结果更新事实；事件唤醒下一步；Checkpoint 让任意健康实例从确定边界继续。**

# 总结

从 Prompt Engineering 到 Harness Engineering，Agent 系统的关注点逐渐从“如何写出更好的指令”，扩展到“如何构建一个可靠的执行系统”。

完整心智模型可以压缩为：

```text
Prompt
    决定模型如何理解任务

Runtime State
    持久保存 Thread / Run / Tool / Task 的运行事实与外部引用

模型输入上下文
    决定模型这一轮真正看到哪些状态与外部信息

Runtime
    决定 Model–Tool Loop 如何持续运行和响应控制信号

Tool-use
    决定模型如何提交动作意图；Executor 再更新运行事实并作用于外部世界

Workspace / Sandbox
    决定模型在哪里行动以及不能越过什么边界

Subagent
    决定复杂任务如何隔离、拆解与调度

```

最终，模型只是 Harness 中的一个推理组件。生产级 Agent 的可靠性主要来自模型之外的确定性工程：统一协议、状态机、持久化、幂等、安全隔离、任务调度和失败处理。

> **越是复杂和高风险的任务，系统能力越不能依赖模型“自觉”，而必须由 Harness 显式表达并强制执行。**

---

# 参考实现与进一步阅读

以下链接沿用原始文档中的参考入口，用于对照具体工程实现：

1. [learn-claude-code：Todo / Plan 示例](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s03-todo-write.md)
2. [Pi Agent Harness](https://github.com/earendil-works/pi)
3. [Pi agent-loop.ts](https://github.com/earendil-works/pi/blob/main/packages/agent/src/agent-loop.ts)
4. [Pi Extensions](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md)
5. [Pi Skills](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/skills.md)
6. [Pi Permission Gate 示例](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/permission-gate.ts)
7. [Pi Protected Paths 示例](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/protected-paths.ts)
8. [Pi Sandbox Extension](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/examples/extensions/sandbox/index.ts)
9. [Pi Gondolin Micro-VM](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/examples/extensions/gondolin)
10. [OpenAI Codex](https://github.com/openai/codex)
11. [Codex App Server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
12. [Codex MCP Interface](https://github.com/openai/codex/blob/main/codex-rs/docs/codex_mcp_interface.md)
13. [Codex Memory Pipeline](https://github.com/openai/codex/blob/main/codex-rs/core/src/memories/README.md)
14. [Model Context Protocol：Tools Specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
