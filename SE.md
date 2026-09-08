# 服务端工程化与架构面试笔记

> 这份笔记不追求背定义，而是回答生产系统设计中的五个问题：**事实存在哪里、并发如何控制、事件如何可靠送达、重复与失败如何恢复、何时需要降级与止损。**

## 0. 一张图理解生产系统

绝大多数服务端系统都可以拆为五个角色：

```text
Client / API
    ↓
MySQL：业务事实、状态机、审计记录（Source of Truth）
    ↓                         ↘
Redis：缓存、短期协调、限流       MQ：异步事件总线
                                      ↓
                               Worker：实际执行
                                      ↓
                         MySQL：推进终态、保存 result_ref
                                      ↓
                         Scanner：超时、漏事件、重试兜底
```

- **MySQL** 保存“世界实际上发生了什么”：订单是否创建、任务是否完成、余额是否扣减。
- **Redis** 保存“短期、可丢失或可重建的信息”：热点缓存、令牌、短时去重、协调锁；它不是多数业务事实的最终裁决者。
- **MQ** 传递“某件事应该被异步处理”的事件；它通常只能做到至少一次投递，重复处理由业务幂等负责。
- **Worker** 消费事件后，必须回写 MySQL 状态，而不是只把成功放在内存中。
- **Scanner** 不是正常主流程的轮询器，而是检查漏投递、超时、卡死和到期重试的恢复器。

先区分三个常被混淆的概念：

```text
消息至少一次送达  !=  业务只生效一次  !=  端到端恰好一次

MQ at-least-once
    + 业务 idempotency_key / 唯一约束 / 状态机 CAS
    → 业务上的 effectively-once（效果只生效一次）
```

面试设计题可按这个顺序回答：

1. **事实源**：状态和结果落在哪张表，状态机有哪些终态。
2. **并发控制**：唯一约束、条件更新、版本号还是行锁。
3. **事件链路**：什么时候发消息、谁消费、ACK 在什么时候提交。
4. **异常恢复**：超时、重复、乱序、漏消息、Worker 崩溃怎么处理。
5. **容量与止损**：限流、队列堆积、重试上限、死信、熔断。

---

## 1. MySQL：正确存事实，正确处理并发

### 1.1 索引：让查询沿有序路径定位，而不是扫描全表

InnoDB 的主键索引是**聚簇索引**：叶子节点保存整行数据；二级索引叶子节点保存“二级索引列 + 主键”。因此通过二级索引查到主键、再去主键索引取其他列，叫**回表**。

这是 InnoDB 的关键特征；MyISAM 的索引叶子通常保存数据文件地址，没有 InnoDB 这种按主键组织整行数据的聚簇索引布局。实际项目重点仍是 InnoDB，不必把 MyISAM 当作常规选型。

#### 为什么是 B+ 树，而不是二叉树或 Hash

```text
根节点 → 非叶子节点 → 叶子节点（按 key 有序，并由双向链表串联）
```

- **矮而宽，减少磁盘 I/O**：一个页可容纳很多 key 和 child pointer，千万级数据通常只需 3～4 层；一次根到叶子的查询只需少量页访问。二叉树高度高，磁盘 I/O 次数更多。
- **范围查询高效**：定位到范围起点后，可以沿叶子节点链表顺序扫描；这正是 `BETWEEN`、`ORDER BY`、分页范围扫描可利用 B+ 树的原因。
- **非叶子节点更紧凑**：B+ 树把完整行数据留在叶子节点，非叶子节点只存 key 和指针，同一页扇出更大，树更矮。
- **Hash 更适合等值、不适合范围**：Hash 对 `WHERE id = ?` 很快，却不保序，难以支持范围、排序和最左前缀；InnoDB 通用索引选择 B+ 树是为了服务更多查询模式。

```sql
CREATE TABLE task (
  task_id     BIGINT PRIMARY KEY,
  thread_id   BIGINT NOT NULL,
  status      VARCHAR(32) NOT NULL,
  updated_at  DATETIME NOT NULL,
  version     BIGINT NOT NULL DEFAULT 0,
  KEY idx_thread_status_updated (thread_id, status, updated_at)
);
```

联合索引 `(thread_id, status, updated_at)` 的核心是**最左前缀**：

```sql
-- 能利用完整联合索引
SELECT task_id, status
FROM task
WHERE thread_id = ? AND status = ?
ORDER BY updated_at DESC;

-- 只能利用 thread_id 前缀，status 之后的排序通常不能直接利用索引
SELECT * FROM task
WHERE thread_id = ?
ORDER BY updated_at DESC;

-- 没有 thread_id，通常不能有效利用该索引定位
SELECT * FROM task WHERE status = ?;
```

常见优化判断：

- 等值条件通常放在前，范围条件后的列往往难以继续用于定位和排序。
- 查询只取索引中的字段时可形成**覆盖索引**，避免回表；不要为覆盖索引盲目堆叠宽索引。
- `LIKE '%foo'`、在索引列上套函数、隐式类型转换会削弱索引利用。
- 索引不是越多越好：每个二级索引都增加写放大、页分裂和维护成本。

#### 覆盖索引：二级索引本身就能回答查询

```sql
-- idx_thread_status_updated(thread_id, status, updated_at) 已包含查询所需列
SELECT status, updated_at
FROM task
WHERE thread_id = ? AND status = ?;
```

此时引擎从二级索引叶子就能返回数据，`EXPLAIN` 的 Extra 常见 `Using index`，无需按主键再回聚簇索引取整行。注意两点：

1. 不是所有“查得快”都是覆盖索引；覆盖索引的严格含义是过滤列、排序列与返回列都能从同一索引得到。
2. 不要为了覆盖所有 `SELECT *` 建超宽索引。索引越宽，缓存命中率越低、写入越慢；覆盖高频关键查询即可。

面试一句话：

> 索引服务于确定的访问路径。先看高频查询的过滤、排序和返回列，再设计联合索引；不能只因为“字段常用”就建单列索引。

### 1.2 事务：ACID 解决什么

- **Atomicity（原子性）**：一组修改要么全部成功，要么全部回滚。
- **Consistency（一致性）**：事务前后满足业务不变式，例如库存不能为负。
- **Isolation（隔离性）**：并发事务的中间状态不能任意互相看见。
- **Durability（持久性）**：提交成功后，宕机恢复不能丢失。

注意：数据库能保证“行级读写一致”，不自动保证你的业务规则。比如“先查库存大于 0，再扣库存”若拆成两条无约束 SQL，仍可能超卖；业务不变式应写进条件更新：

```sql
UPDATE inventory
SET stock = stock - 1
WHERE sku_id = :sku_id
  AND stock > 0;
```

受影响行数为 0 时表示库存不足或发生并发竞争，不是单纯的数据库异常。

### 1.3 隔离级别与 MVCC：普通读为何不总加锁

| 隔离级别 | 典型问题 | InnoDB 中的理解 |
| --- | --- | --- |
| Read Uncommitted | 脏读 | 几乎不使用 |
| Read Committed | 不可重复读 | 每次一致性读可能建立新的 Read View |
| Repeatable Read | 幻读需额外讨论 | InnoDB 默认；普通快照读通常复用同一 Read View |
| Serializable | 并发很低 | 以更强锁/串行语义换正确性 |

三个现象要能一句话分清：**脏读**是读到其他事务尚未提交的数据；**不可重复读**是同一事务两次按主键读到不同已提交版本；**幻读**是范围条件下，其他事务插入/删除了满足范围的新行。MVCC 主要处理快照读可见性；当前读的范围保护仍需要锁。

**MVCC（Multi-Version Concurrency Control）**解决的是“读不必阻塞写、写不必阻塞普通读”。一条 InnoDB 记录除了业务字段外，逻辑上还关联事务版本信息和 Undo Log 版本链。

```text
当前行版本
  ↓ 找不到对当前 Read View 可见的版本
Undo Log 版本链
  ↓
找到可见的历史版本 → 返回给快照读
```

#### MVCC 到底如何判断“这个版本能不能看”

可以把一条记录理解为带有两个隐藏信息：

```text
DB_TRX_ID      最后修改该行的事务 ID
DB_ROLL_PTR    指向 Undo Log 中的上一个版本
```

事务第一次做一致性读时创建 **Read View**，其中至少记录“此刻仍活跃的事务 ID 集合”和边界。简化的可见性判断是：

```text
版本由当前事务写入                 → 可见
版本事务已在 Read View 创建前提交    → 可见
版本事务仍活跃 / 在快照后才开始      → 不可见，沿 roll_pointer 找 Undo 旧版本
```

因此同一行可以同时存在：A 事务正在写入的新版本、B 事务快照读到的旧版本。A 不必等 B 读完，B 也通常不必等 A 提交，这就是高并发读写共存的来源。

- **快照读**：普通 `SELECT` 读取 Read View 中可见的版本，通常不加锁。RC 通常在每次一致性读创建新的 Read View；RR 则在**第一次一致性读**创建后，后续快照读复用它，而非在事务一开始就必然生成。
- **当前读**：`SELECT ... FOR UPDATE`、`UPDATE`、`DELETE` 读取最新版本，并需要加锁控制并发修改。

不要把 MVCC 理解为“永远没有锁”。它只是让大量普通读通过版本可见性完成；修改同一事实、抢占任务、扣库存仍必须依赖锁或条件更新。

面试追问可以这样回答：

> MVCC 解决的是普通快照读的可见性与读写并发；它不解决“两个事务能否同时修改同一行”。后者仍要由当前读、行锁、唯一约束或 CAS 状态机裁决。

### 1.4 锁：先缩小锁范围，再处理竞争失败

常见锁类型：

- **S / X Lock**：共享锁允许并发读，排他锁用于修改；X 与其他 S/X 冲突。
- **IS / IX Lock**：表级意向共享/意向排他标记，告诉系统“表中某些行将被加 S/X 锁”，使表锁冲突判断无需逐行扫描。
- **Record Lock**：锁定某条索引记录。
- **Gap Lock**：锁定索引记录之间的间隙，防止插入。
- **Next-Key Lock**：记录锁 + 间隙锁；RR 下用于避免当前读中的幻读。
- **Intention Lock**：表级意向标记，协调行锁和表锁，不等于锁住整张表。

#### 锁是在什么情况下触发的

| 操作/条件 | RR 下典型锁行为 | 为什么 |
| --- | --- | --- |
| 普通 `SELECT` | 快照读，通常不加锁 | 由 MVCC 返回可见版本 |
| `SELECT ... FOR UPDATE` / `LOCK IN SHARE MODE` | 当前读，加 X/S 相关锁 | 后续要修改或要求读取期间不被改写 |
| `UPDATE/DELETE` 命中唯一索引的等值已存在记录 | Record Lock | 精确锁住该记录即可 |
| 非唯一索引范围查询、`BETWEEN`、范围 `UPDATE` | Next-Key Lock | 防止其他事务在范围中插入“幻影行” |
| 唯一索引等值查询但目标不存在 | 可能锁住对应 Gap | 防止并发插入同一个缺失 key |
| 条件无合适索引 | 扫描并锁住大量聚簇索引记录 | 锁范围失控，近似全表受阻 |

`READ COMMITTED` 下 InnoDB 会减少 Gap/Next-Key Lock 的使用，但外键检查、唯一键冲突检查等场景仍可能涉及间隙保护。面试里不要死背“RC 完全没有间隙锁”，应强调：**隔离级别、查询类型和索引命中方式共同决定最终锁范围。**

```sql
-- 领取一个待执行任务：必须在合适索引上做条件定位
SELECT task_id
FROM task
WHERE status = 'PENDING'
ORDER BY updated_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

生产原则：

1. 让 `WHERE` 命中合适索引，避免锁范围膨胀。
2. 多资源加锁保持相同顺序，降低死锁概率。
3. 死锁是并发系统中的正常现象：捕获数据库 deadlock 错误后，针对幂等事务做有限重试。
4. 长事务会拖住锁、Undo Log 和历史版本清理；事务内不要执行网络 RPC 或慢计算。

常见误区是“查询不走索引就自动退化为表锁”。更准确地说：InnoDB 仍以索引记录为锁对象；全表扫描会锁住大量聚簇索引记录，效果上接近锁全表，既拖慢并发也更容易制造死锁。真正的表级锁语义要另行显式使用。

### 1.5 三大 Log：各自解决不同问题

| 日志 | 主要用途 | 典型关键词 |
| --- | --- | --- |
| Undo Log | 回滚、MVCC 历史版本 | 事务撤销、版本链 |
| Redo Log | 崩溃恢复、WAL | 先写日志后落数据页 |
| Binlog | 复制、审计、增量订阅 | MySQL Server 层逻辑变更 |

**Redo Log** 是 WAL：事务提交时先保证必要的 Redo 已持久化，数据页可以之后再刷盘；宕机后据此重做已提交修改。**Undo Log** 记录反向信息，支持回滚和快照读找旧版本。**Binlog** 记录逻辑变更，供主从复制、CDC 等消费。

#### 三大日志的工作细节与边界

**Redo Log：为什么提交不必等待数据页刷盘**

```text
修改 Buffer Pool 中的数据页
    ↓
生成 Redo，写入 redo log buffer
    ↓
提交时按 durability 策略刷入 redo log file
    ↓
数据页在后续 checkpoint / 刷脏页时再落盘
```

Redo 记录的是对数据页的变更，用于崩溃后的 redo；它是循环使用的，checkpoint 推进后才能复用旧空间。WAL 的收益是把随机数据页写转化为更顺序、更小的日志写；代价是恢复时可能要重放尚未刷盘的数据页变更。

**Undo Log：为什么既能回滚，又能给 MVCC 提供旧版本**

- 更新/删除时，Undo 保存修改前的信息；事务失败时按 Undo 反向恢复。
- 已提交的旧版本不能立刻删除，因为仍可能被更早 Read View 的快照读访问。
- Purge 线程会在确认没有活跃 Read View 需要这些历史版本后清理 Undo；长期事务会阻碍清理，造成 History List Length 增长和存储压力。

**Binlog：为什么它不是崩溃恢复日志**

- Binlog 位于 MySQL Server 层，记录逻辑 DDL/DML，通常用于主从复制、审计、CDC 与基于时间点恢复。
- 常见格式是 `STATEMENT`、`ROW`、`MIXED`；生产中 `ROW` 更利于准确复制和 CDC，但会增加日志体积。
- Binlog 追加归档、按文件滚动，不像 Redo 那样作为有限环形空间循环覆盖。

InnoDB 的两阶段提交要解决的是 Redo 与 Binlog 不能一边成功、一边丢失：

```text
Redo prepare
  → 写入并刷盘 Binlog
  → Redo commit
```

恢复时可以根据两者状态判断提交与否，避免“主库已提交但 Binlog 没有”或“Binlog 有但存储引擎未提交”的复制不一致。

一句话收束：

> Redo 负责“宕机后数据页如何恢复”；Undo 负责“事务如何撤销、快照如何读旧版本”；Binlog 负责“其他系统如何知道发生过哪些逻辑变更”。

### 1.6 乐观锁：用状态和版本做 CAS

对“冲突较少、可重试”的状态推进，乐观锁往往比先加行锁更简单：

```sql
UPDATE task
SET status = 'RUNNING',
    version = version + 1,
    updated_at = NOW()
WHERE task_id = :task_id
  AND status = 'PENDING'
  AND version = :version;
```

```text
affected_rows = 1 → 当前消费者成功领取
affected_rows = 0 → 已被其他消费者领取、状态已终止或版本过期
```

后者需要重新读取状态：如果已成功则幂等返回；如果仍可执行则决定是否有限重试；不能不加判断地无限重试。

---

## 2. Redis：缓存、协调与流量治理

### 2.1 常用数据结构不是知识点，是原子操作选择

| 结构 | 常见用途 | 关键能力 |
| --- | --- | --- |
| String | 缓存、计数、分布式锁 | `GET/SET/INCR` 原子操作 |
| Hash | 用户/任务的稀疏字段 | 局部字段更新 |
| List | 简单队列、最近记录 | 双端 push/pop |
| Set | 去重、集合关系 | 成员唯一、集合运算 |
| ZSet | 排行榜、延时任务索引 | score 排序与范围取数 |
| Stream | 消费组事件流 | 消费确认、Pending Entries |

选型先问：是否需要排序、去重、字段更新、消费确认、原子扣减？不要把 Redis 当成无约束 JSON 数据库。

Redis 的底层编码会随元素数量、字段长度等条件在紧凑表示与通用结构之间转换；面试里不必死记阈值，关键是理解：数据结构决定可做的原子操作，也决定内存、复制和大 Key 风险。

Redis Stream 能提供消费组、Pending Entries 和 `XACK`，适合轻量内部事件；但重试、延迟、死信、堆积治理和跨系统可观测性通常需要自行补齐，因此不要在核心高可靠任务场景中因为“Redis 已经有 Stream”就跳过 MQ 设计。

### 2.2 缓存三问：穿透、击穿、雪崩

| 问题 | 根因 | 常用手段 |
| --- | --- | --- |
| 缓存穿透 | 请求不存在的数据，每次都打 DB | 空值缓存、Bloom Filter、参数校验 |
| 缓存击穿 | 单个热点 Key 失效，大量请求同时回源 | 互斥重建、逻辑过期、热点预热 |
| 缓存雪崩 | 大量 Key 同时失效或 Redis 故障 | TTL 打散、多级缓存、限流降级、预热 |

“逻辑过期”指缓存值本身带业务过期时间：过期后仍短暂返回旧值，同时只让后台单个任务刷新。它偏向读可用和抗击穿；如果读到旧数据不可接受，则改用互斥重建或直接读库，不能混称为强一致方案。

#### 先分清：三者的故障规模不同

```text
缓存穿透：不存在的 Key 反复查询                 → 每次都穿到 DB
缓存击穿：一个热点 Key 刚好失效                 → 同一时刻大量请求回源
缓存雪崩：大批 Key 同时失效，或 Redis 整体失效  → 大范围流量涌入 DB
```

它们不是同义词。击穿关注**一个热点**的瞬时并发；雪崩关注**一批 Key 或缓存集群**的系统性失效；穿透关注**本来就不存在的数据**被反复请求。

#### 2.2.1 缓存击穿：热点 Key 过期，不要让所有人一起回源

场景是爆款商品、热门用户主页等：缓存刚过期时，大量并发请求同时 miss，都去查 DB。

**方案一：逻辑永不过期 + 后台异步刷新（热点读多、允许短暂旧值时首选）**

```json
{
  "data": { "...": "..." },
  "logical_expire_at": "2026-09-09T12:00:00Z"
}
```

读请求发现逻辑过期后，仍可先返回旧值；同时用单飞/锁保证只有一个后台任务刷新。优点是请求路径几乎不回源等待，从根源减少 TTL 到点引发的击穿；缺点是数据可能短暂陈旧，刷新任务失败时需要监控与重试。

**方案二：互斥重建（必须尽量读新值时使用）**

```text
cache miss
  → 尝试获取 Redis 分布式锁
     ├─ 成功：查 DB → 回填缓存 → 释放锁
     └─ 失败：短暂等待后重读缓存，或直接返回降级结果
```

优点是同一时刻只让一个请求击中 DB；缺点是锁竞争增加等待和尾延迟，锁超时/续租处理复杂。它适合单个热点回源成本很高但不能长时间返回旧值的场景；不能用 JVM 本地锁替代多实例环境的分布式协调。

**方案三：提前主动刷新**

根据访问热度或临近过期时间，在后台预热更新。它适合规律性热点，但预测错误会浪费资源，因此通常与前两种之一组合。

> 热点数据常用组合：**逻辑过期 + 后台单飞刷新**；只有读旧值不可接受时，才以互斥重建换取更强的新鲜度。

#### 2.2.2 缓存雪崩：分别处理“TTL 扎堆”和“Redis 不可用”

**成因 A：大量 Key TTL 集中到期**

- 设置 `base_ttl + random_jitter`，例如 30 分钟加 0～5 分钟随机偏移，打散失效波峰；这是成本最低的标配。
- 热点核心数据用逻辑过期或主动预热，不参与批量硬过期。
- 发布、预热或迁移时避免给海量 Key 同时写入相同 TTL。

**成因 B：Redis 整体不可用（更危险）**

- Redis 主从 + Sentinel 或 Redis Cluster，降低单节点故障演变为整体不可用的概率。
- 本地缓存 / CDN / 网关缓存作为第二层，Redis 故障时仍能吸收部分读流量。
- 最后一层是限流、熔断与降级：返回静态数据、过期但可接受的旧值，或明确“稍后重试”，保护 DB 不被击穿。

> TTL 随机偏移只能解决“同时过期”的雪崩，**不能解决 Redis 宕机**。宕机需要高可用、多级缓存和流量保护共同兜底。

#### 2.2.3 缓存穿透：先证明请求有可能命中

**方案一：参数与权限校验**。非法 ID、格式错误、越权请求在入口直接拒绝，是最便宜的一层。

**方案二：缓存空值**。DB 查不到时写入短 TTL 的空标记：

```text
GET user:404
  → miss → DB not found
  → SET user:404 __NULL__ EX 60
```

短时间内同一个无效查询被缓存挡住；代价是攻击者构造海量随机 Key 时仍可能占用 Redis 内存，因此 TTL 要短，并配合限流。

**方案三：Bloom Filter**。提前把合法 ID 放入 Bloom Filter：

```text
Bloom says definitely-not-exist → 直接拒绝
Bloom says may-exist            → 再查 Cache / DB
```

它的特点是允许**误判存在**（False Positive），但不会把真实存在的元素判成不存在；这也是为什么它适合作为前置过滤器。缺点是经典 Bloom Filter 不便删除、数据频繁变更时维护复杂，不能把“可能存在”当作数据真的存在。

#### 生产背诵版

```text
穿透：参数校验 + 空值缓存；高并发恶意请求再加 Bloom Filter。
击穿：热点 Key 逻辑过期 + 后台异步刷新；读新值要求高时再用分布式锁。
雪崩：TTL 随机打散；Redis 高可用 + 本地多级缓存；最后限流、熔断、降级保护 DB。
```

**Cache Aside** 是最常见写策略：读时 Cache miss 再查 DB 并回填；写时先更新 DB，再删除缓存。

```text
更新 DB 成功
    ↓
删除缓存
    ↓
下一次读取回源并回填
```

为什么通常不是“先删缓存再写 DB”？因为删除和写 DB 之间可能有并发读把旧 DB 值重新回填。先写 DB 再删也不是强一致：删缓存失败、读写交错仍可能短暂读旧值。因此需要：

- 删除失败重试、订阅 Binlog/CDC 做补偿，或对关键数据直接读库。
- 给缓存设置 TTL，避免极端失败造成永久脏缓存。
- 认清缓存通常提供的是**最终一致性**，不能承担转账、库存扣减等强一致事实。

“延迟双删”是有人会使用的缓解手段：更新 DB 后删一次缓存，延迟一小段时间再删一次，试图覆盖并发读回填旧值的窗口。它既不是 Cache Aside 的必需步骤，也不能保证强一致；若采用，第二次删除仍要有失败重试或 CDC 补偿，不能只靠一次定时 sleep。

### 2.3 分布式锁：Redis 锁只是协调工具

最小正确获取锁：

```text
SET lock:order:123 <random-owner-token> NX PX 30000
```

- `NX`：仅不存在时创建，保证互斥抢占。
- `PX`：必须设置过期，避免持锁实例永久故障。
- value 必须是唯一 owner token，不能只存 `1`。

释放锁不能先 `GET` 再 `DEL`，应由 Lua 原子比较并删除：

```lua
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
```

锁的边界：

- 执行时间可能超过 TTL；续租需要持有者校验、失败处理和上限，不能无限续。
- GC pause、网络抖动后，旧 Worker 可能在锁过期后恢复执行。
- 涉及钱、订单、库存等最终裁决，仍应靠 DB 唯一约束、条件更新或事务；Redis 锁用于减少竞争，而不是替代持久化一致性。

“看门狗续期”本质是持锁 Worker 周期性延长自己仍持有的锁，适合可控时长的任务；它不是正确性的魔法。网络隔离、GC pause、Worker 被杀后仍可能出现旧执行者恢复，因此关键写操作仍要有 Fencing Token 或 DB 状态 CAS 作为最后防线。

### 2.4 大 Key 与热点 Key

**大 Key** 会放大网络、序列化、过期删除和主线程阻塞风险。处理方式：拆分 Hash/List、分页读写、对大集合采用渐进删除或异步删除、监控 value 大小和元素数量；不要在主路径反复读取超大 JSON。

**热点 Key** 的问题是单分片/单主节点成为瓶颈。可组合：

- 本地进程缓存短时间吸收读流量；
- 副本读、读写分离或按用户/维度拆 key；
- 热点预热和逻辑过期，避免瞬时回源；
- 对单 Key 单独限流，必要时返回降级结果。

### 2.5 令牌桶：允许短时突发，限制长期平均速率

令牌桶维护剩余令牌和上次补充时间：

```text
elapsed = now - last_refill
tokens  = min(capacity, tokens + elapsed * refill_rate)

if tokens >= cost:
    tokens -= cost
    allow
else:
    reject / queue
```

其中：

- `capacity` 决定最多能积攒多少令牌，也决定允许多大的短时突发。
- `refill_rate` 决定长期平均速率，例如每秒补 100 个令牌。
- `cost` 是一次请求消耗量；普通 API 通常为 1，LLM 请求可以按预估 Token 或模型权重计费。

例如桶容量为 200、每秒补 100 个令牌：正常时最多瞬时放行 200 个请求；持续流量最终会稳定在每秒 100 个。它和固定窗口计数器的区别在于：令牌桶允许积攒空闲额度，因此能优雅承受合理突发；固定窗口在窗口边界容易出现两倍突发。

Redis 实现必须用 Lua 把“补充 + 判断 + 扣减 + 更新”放在一个原子脚本中。对于 LLM 或昂贵 API，通常至少有两类桶：

```lua
-- Redis Hash: { tokens, last_refill_ms }
elapsed = now_ms - last_refill_ms
tokens = min(capacity, tokens + elapsed * refill_rate / 1000)
if tokens < cost then return {0, tokens} end
tokens = tokens - cost
HSET key tokens tokens last_refill_ms now_ms
PEXPIRE key ttl_ms
return {1, tokens}
```

实际脚本还要处理首次初始化、时钟异常、浮点精度和过期时间；关键不是逐行背代码，而是确保**补充、判断、扣减**不可被并发请求拆开。

```text
RPM / QPS：限制请求数
TPM：限制预估或实际输入输出 Token
```

一次请求可先按预估 Token 预扣，结束后按实际 usage 结算；否则只限制 QPS，超长请求仍可能压垮下游预算。

工程上通常分两层：本地令牌桶负责保护**单个实例**，Redis + Lua 负责租户、用户或 API Key 的**全局配额**。LLM 网关还常加 `inflight` 并发席位，防止请求数尚未超限但大量长推理同时占满连接、KV Cache 或下游队列。

#### LLM / 网关中的三维限制

```text
QPS / RPM      防止请求风暴
TPM            防止少量超长请求耗尽模型配额和成本
inflight       防止同时运行的长请求占满连接、GPU/KV Cache 或 Worker
```

一个请求可以先按 `prompt_tokens + max_output_tokens` 预扣 TPM，返回后再根据实际 usage 结算或退款；否则用户只要持续发很长的请求，即使 QPS 很低也会超出下游 Token 配额。`inflight` 通常用租约/计数器领取，请求完成、取消或超时后释放；必须有超时回收，避免进程崩溃后席位永久泄漏。

限流后的动作也要区分：

- 能排队且用户可等待：返回排队位置或 `Retry-After`，进入有界队列。
- 请求无价值或队列已满：直接拒绝，避免排队把系统拖死。
- 下游已故障：配合熔断快速失败或切换低成本模型/降级能力，而不是继续把请求塞进桶里等待。

---

## 3. MQ：可靠传递与异步解耦

### 3.1 MQ 保证什么，不保证什么

可靠消息链路至少有四段：

```text
Producer
  → Broker 持久化 / 副本确认
  → Consumer 收到消息
  → 业务处理并提交 DB 状态
  → ACK / offset commit
```

每一段都可能重试：生产者不确定 Broker 是否收到、Consumer 处理成功但 ACK 丢失、Consumer 崩溃后重复拉取。因此最稳妥的默认语义是：

> MQ 提供 **at-least-once**；Consumer 必须通过幂等键、唯一约束或状态机把重复消息折叠为一次业务效果。

不应直接宣称“Kafka/RocketMQ 可以保证业务恰好一次”。即便 Kafka 有幂等生产者与事务，跨越外部 DB、HTTP 写操作时，业务端仍要解决重复效果和状态未知。

### 3.2 生产与消费的可靠性设计

生产侧：

- 确认 Broker 已接收并按需要持久化/复制；Kafka 关注 `acks`、副本同步和幂等生产者，RocketMQ 关注可靠投递、刷盘/副本策略及发送结果。
- 消息只带稳定标识和必要元数据，例如 `task_id`、`event_id`、`attempt`，不要把可变大对象完整复制到消息体。
- 业务事实与消息投递存在双写窗口：DB 提交成功但 MQ publish 失败，会漏事件；先 publish 又可能事务回滚。

消费侧：

- 先处理并提交业务状态，再 ACK/提交 offset；否则先 ACK 再宕机会丢业务。
- 消费逻辑可重复执行，状态更新必须有唯一约束或条件更新。
- 按业务 Key 分区/路由以获得局部顺序；跨分区通常不承诺全局顺序。

### 3.3 重试、延迟与死信队列

```text
可恢复瞬态错误（网络超时、临时 429、5xx）
    → 有上限的延迟重试
    → 超上限进入 DLQ

不可恢复错误（参数错误、Schema 不合法、权限拒绝）
    → 直接失败 / DLQ / 人工修复
```

每次重试要携带 `attempt`、错误类别和 `next_retry_at`；退避时间应逐步增长并加入随机抖动，避免故障恢复时的重试风暴。

**DLQ（Dead Letter Queue）不是垃圾桶**：它应该被监控、告警、归因和可控重放。重放前通常要修复代码、数据或下游依赖；直接无限 requeue 只会毒化主队列。

### 3.4 Kafka 与 RocketMQ：先看消息生命周期，再看性能

> 这里的 RMQ 指 **RocketMQ**，不是 RabbitMQ。

| 维度 | Kafka | RocketMQ |
| --- | --- | --- |
| 本质 | 分区追加日志 | 面向业务消息的分布式消息中间件 |
| 消费 | Consumer Group 按 offset 拉取、可回放 | Consumer Group 消费队列，围绕业务消息确认与重试 |
| 强项 | 高吞吐事件流、长保留、回放、流式计算生态 | 延迟消息、事务消息、顺序消息、业务重试与死信治理更直接 |
| 重试/DLQ | 常以 retry topic / DLT 和消费者策略实现 | Broker 提供更完整的重试、死信相关能力与约定 |
| 消费过滤 | 按 Topic / Partition 订阅；消息 Header/Body 条件通常由 Consumer 收到后自行过滤 | Broker 侧支持 Tag 精确过滤，以及按消息属性的 SQL92 过滤 |
| 常见场景 | 埋点、CDC、日志、实时数据流、Flink 管道 | 订单、支付、通知、异步业务任务 |

选择依据：

- 需要保留事件流、按 offset 回放、供多组消费者独立消费，或与 Flink 等流计算体系配合，优先 Kafka。
- 需要延迟/事务/顺序等业务消息能力，以及更直接的重试、死信治理体验，RocketMQ 往往更顺手。
- RocketMQ 的 Tag 适合 Topic 内的轻量消息分类；复杂场景可让 Producer 写入属性、Consumer 以 SQL92 表达式订阅，Broker 只投递匹配消息。Kafka 没有对应的消息属性 Broker 过滤，若 Consumer 自行筛掉消息，网络与反序列化成本已经发生。
- 过滤不是把所有不相关业务硬塞进一个 Topic 的理由：Topic 仍应按稳定业务边界划分；同一 Consumer Group 内的订阅/过滤条件必须一致，避免部分消息无人消费。
- Kafka 也可以支撑在线业务；RocketMQ 也能承载数据流。区别是默认抽象和生态重心，不是“一个只能离线、一个只能在线”。
- 两者都不替代业务状态表：消息是推进信号，DB 状态才是事实判断。

---

## 4. 幂等：把至少一次执行变成业务上只生效一次

### 4.1 幂等键是什么

`idempotency_key` 表示“同一业务动作”的稳定身份，而不是每次 HTTP 请求随机生成的 request id。

```text
用户 U 对订单 O 的“支付确认”
  → idempotency_key = hash(U, O, operation_type)
```

同一个 key 在合理的业务作用域和保留期内重复到达，应返回第一次的处理结果，或返回“处理中”，而不是再次创建订单/扣款/发送邮件。

### 4.2 三层机制分别解决什么

| 机制 | 适合解决 | 局限 |
| --- | --- | --- |
| Redis `SETNX` | 短窗口去重、削峰、避免并发重入 | TTL 到期会失效，不能作为永久业务事实 |
| DB 唯一索引 | 最终裁决、跨进程/重启仍有效 | 冲突后需要读取既有结果返回 |
| DB 乐观锁 / 条件更新 | 状态机 CAS、只允许合法迁移 | 需要建模好状态与版本 |

一个常见组合：

```text
请求到达
  → Redis SETNX 降低短时并发重入
  → DB unique(idempotency_key) 作为最终裁决
  → status/version 条件更新推进状态机
```

Redis 抢锁失败不应机械报错：先查询 DB 是否已有成功结果、是否仍处理中，再选择复用结果、提示稍后重试或进入等待。

### 4.3 Timeout 不等于没有执行：状态未知最危险

只读操作如 `GET`、Search、QueryStatus，网络超时后通常可以重试。写操作如支付、部署、发邮件、创建工单则可能已经在服务端成功，只是响应丢失。

```text
写请求 Timeout
    ↓
查询 operation_id / 幂等 Ledger
    ├─ SUCCESS      → 复用已有结果
    ├─ NOT_STARTED  → 可安全重试
    └─ UNKNOWN      → 不自动重放，人工或异步核对
```

> 可靠系统不是把所有超时都重试，而是识别“是否可安全重放”。至少一次重试必须配合幂等语义。

---

## 5. 异步系统：状态机、事件和定时兜底

### 5.1 先持久化状态，再异步执行

任何可能跨进程、跨分钟或跨服务的任务，都不应只靠“一个挂起协程”保存进度。最小任务实体：

```text
task_id           任务身份
status            PENDING / RUNNING / SUCCEEDED / FAILED / TIMEOUT / CANCELLED
version           乐观锁版本
attempt           当前尝试次数
deadline          最晚完成时间
next_retry_at     下一次允许重试的时间
idempotency_key   同一逻辑动作的去重身份
result_ref        大结果或产物引用
lease_until       Worker 领取后的租约截止
```

状态表是 Source of Truth；MQ 完成事件只负责通知“值得重新检查该任务”，不替代状态判定。

### 5.2 主流程：DB 状态 + MQ 事件 + Worker 状态推进

```text
Client
  → DB: 创建 PENDING task
  → MQ: TaskRequested(task_id)

Worker
  ← MQ: 收到 TaskRequested
  → DB: CAS 领取 PENDING → RUNNING，写 lease / attempt
  → Execute
  → DB: 写 SUCCEEDED / FAILED / TIMEOUT + result_ref
  → MQ: TaskCompleted(task_id)

Coordinator / API
  ← TaskCompleted
  → DB: 查询状态，决定返回、推进后继任务或通知用户
```

Worker 的核心不是“收到消息就执行”，而是“先在 DB 中原子抢到合法状态再执行”。重复事件、重复消费、重平衡后重新投递，都会在条件更新处自然收敛。

```sql
UPDATE task
SET status = 'RUNNING',
    attempt = attempt + 1,
    lease_until = NOW() + INTERVAL 5 MINUTE,
    version = version + 1
WHERE task_id = :task_id
  AND status = 'PENDING'
  AND version = :version;
```

### 5.3 DB 与 MQ 双写：Outbox 是更严谨的生产方案

直接执行：

```text
DB commit → publish MQ
```

会遇到 DB 成功、进程在 publish 前宕机，导致**漏事件**；反过来先 publish 又可能 DB 回滚，导致**幽灵事件**。

严谨做法是同一 DB 事务内写业务状态与 Outbox：

```sql
BEGIN;
  INSERT INTO task (..., status) VALUES (..., 'PENDING');
  INSERT INTO outbox(event_id, topic, payload, status)
  VALUES (..., 'task.requested', ..., 'NEW');
COMMIT;
```

独立 Publisher/CDC 可靠地发布 Outbox，成功后标记已投递。这样“业务状态存在”与“待投递事件存在”原子一致。

如果系统暂时不引入 Outbox，必须把 `PENDING 且未投递` 作为可扫描状态，让 Scanner 补发事件；这不是完美替代，但能缩小漏事件窗口并形成恢复路径。

### 5.4 Scanner：不是正常轮询，是异常恢复器

定时任务只处理异常或到期状态：

```text
PENDING  且长时间未投递        → 补投 TaskRequested
RUNNING  且 lease/deadline 到期 → 标记 TIMEOUT 或回收为可重试
FAILED   且 next_retry_at 到期  → 条件更新为 PENDING 并投重试事件
Outbox   状态 NEW              → 发布后标记 SENT
```

关键点：Scanner 也只能通过条件更新推进状态，不能直接“看见任务就再执行”。因此多个 Scanner 实例同时运行仍是安全的。

### 5.5 异步系统的失败分支

| 失败 | 处理 |
| --- | --- |
| 消息重复 | 幂等键、唯一索引、状态 CAS 收敛 |
| Worker 执行到一半崩溃 | lease 到期后由 Scanner 回收；写操作需查外部 operation 状态 |
| MQ 暂不可用 | Outbox/待投递状态保留，恢复后补发 |
| 下游持续故障 | 有上限重试、退避、熔断、DLQ/人工处理 |
| 大结果无法放消息 | 写 Artifact/Object Storage，状态表只保存 `result_ref` |
| 完成事件丢失 | Coordinator 重启或 Scanner 重新读状态表推进后续 |

面试收束：

> 异步设计的核心不是“把函数扔到队列里”，而是把任务抽象为可持久化、可幂等推进的状态机；MQ 提供事件驱动，Scanner 提供漏事件和超时恢复，状态表始终负责最终裁决。
