# Personal Agent

从零构建的多平台 AI Agent 系统。支持飞书/Telegram/微信，多 LLM Provider，完整安全管线，多 Agent 编排引擎，语义记忆，MCP 集成。

## 快速开始

```bash
uv sync
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

uv run python -m personal_agent --cli "1+1等于几？"
uv run python -m personal_agent                # 启动服务
```

## 架构一览

```
飞书 / Telegram / 微信
    │  BasePlatformAdapter 自注册
    ▼
Gateway ── Auth（白名单+配对码）/ 会话管理 / Agent 调度
    │  LRU 128 agent cache + generation 自动失活
    ▼
Agent 循环 ── LLM 调用 → 解析 → 工具执行管线 → 继续
    │  3 层消息：history → ctx.messages → api_messages (injections)
    │  6 种重试 + 上下文压缩 + 中断式 LLM 调用
    ▼
工具执行管线 ── pre_check → scope gate → checkpoint → dispatch → post
```

## 核心特性

### 安全管线（8 层防御）

工具通过 5 段管道，每段有独立的安全检查：

```
tool_call 到达
  ├─ ① pre_check — 硬拦截，永不问用户
  │   bash: 硬黑名单 (rm -rf /, mkfs, dd, fork bomb, shutdown)
  │         白名单 (42 命令) + 危险模式 + 链接检测 + 网络隔离
  │   write: 扩展名白名单 + 大小限制 + 路径遍历
  │   read: 敏感文件拦截 (.env, id_rsa, .ssh/config, .netrc)
  │   web: SSRF 防护 (127.0.0.1, 169.254.169.254, 内网 IP)
  │
  ├─ ② scope gate — 可能问用户
  │   destructive 工具需 /allow write|bash|all
  │   独立 destructive 配额 (3/轮)
  │
  ├─ ③ env sanitization — 子进程启动前 strip API key
  │
  ├─ ④ checkpoint — write/edit 前备份原文件
  │
  └─ ⑤ audit + redact — 每次操作记 JSONL 审计日志，API key 自动掩码
```

### 多 Agent 编排引擎

CC 风格的三原语系统，Orchestrator 动态组合：

| 原语 | 行为 | 示例 |
|------|------|------|
| `sub_agent` | 单子 Agent（parallel-safe）| 调 3 次 = 并行研究 |
| `sub_parallel` | 并行 + 屏障 | 三个数据库对比，等全部再继续 |
| `sub_pipeline` | 流水线无屏障 | 15 个发现逐一验证，不等 |

```
主 Agent（唯一决策者）
  ├─ sub_agent("研究 A", allowed_tools=["write"])  → 可写
  ├─ sub_agent("研究 B")                            → 只读（默认）
  └─ sub_agent("研究 C")                            → 只读

workflow_run("review")  → parallel finders → pipeline verify → report
```

**worktree 隔离**：子 Agent 写入时自动创建独立 git worktree，不冲突，主 Agent 控制 merge。

**子 Agent 安全**：默认只读，主 Agent 通过 `allowed_tools` 显式授权。走完整 scope gate 管线。

### MCP 集成

手写 JSON-RPC 2.0 over stdio，零外部依赖。MCP Server 工具自动注册为 ToolEntry，走 tool_search 桥接通路。

```yaml
mcp:
  servers:
    - name: "filesystem"   # 14 工具，C:\Users\MR 沙箱
    - name: "github"       # 26 工具 (issue/PR/仓库)
    - name: "memory"       # 9 工具（知识图谱）
    - name: "sequential-thinking"  # 复杂推理
```

### 工具系统

25+ 内置工具，自注册机制。桥接工具（tool_search/describe/call）让 LLM 通过 BM25 搜索发现 deferrable 工具：

| 分组 | 工具 |
|------|------|
| 文件 | read, write, edit, grep, glob |
| Web | web_search (Bing), web_fetch (+ SSRF) |
| 终端 | bash (白名单), execute_code (沙箱 Python) |
| 任务 | todo (CC 风格全量替换), task (SQLite 持久化) |
| 交互 | clarify (CC 风格结构化提问), confirm (操作前确认) |
| 多 Agent | sub_agent, sub_parallel, sub_pipeline, workflow_run |
| 隔离 | worktree_create, worktree_merge, worktree_cleanup |
| 进程 | process_list, process_kill, process_wait |
| 记忆 | memory, memory_ingest |
| 技能 | skill_search, skill_load |

### 记忆系统

双层架构，不依赖向量数据库：

```
data/system/SOUL.md     → 系统提示素材（手写，注入 system prompt）
data/system/AGENT.md    → 行为规则
data/system/MEMORY.md   → 用户画像
data/system/USER.md     → 用户偏好

data/memory/            → 外部语义记忆
  fastembed + bge-small-zh-v1.5 (512 维)
  cosine 检索，prefetch 每轮自动注入
  支持 .txt .md .pdf .docx 摄取
```

### 多 Provider

| Provider | API 模式 | 说明 |
|----------|---------|------|
| DeepSeek | Anthropic Messages | 默认 |
| OpenAI | Chat Completions | |
| Anthropic | Anthropic Messages | |
| OpenRouter | Chat Completions | ranking headers |

自动检测 api_mode。HTTP 层指数退避重试（429/5xx/connection）。上下文窗口自适应检测。

### 平台适配

| 平台 | 连接方式 | 特性 |
|------|---------|------|
| 飞书 | WebSocket v2 | 去重/防抖/@ 检测/健康检查重连 |
| Telegram | PTB polling | Markdown 降级重试 |
| 微信 | iLink API | QR 扫码登录，长轮询 |

## 配置

```yaml
# config.yaml — 行为配置
agent:
  max_iterations: 30
  max_tool_calls_per_turn: 20

toolsets:
  enabled: ["all"]

memory:
  external_provider: "embedding"
  review_interval: 10

security:
  bash_allow_network: false
  file_max_write_bytes: 100000
  audit_enabled: true

mcp:
  enabled: true
  servers: [...]

auth:
  enabled: true
```

## 测试

```bash
uv run pytest tests/ -v    # 161 tests, 0 failures
```

## 技术栈

Python 3.12+ / uv / asyncio / httpx / aiohttp / aiosqlite /
lark-oapi (飞书) / python-telegram-bot / iLink API (微信) /
fastembed (语义记忆) / pymupdf + python-docx (文件摄取) /
tiktoken (token 计数) / pydantic-settings

不依赖 LangChain、CrewAI 等重框架。
