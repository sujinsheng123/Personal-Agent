# Personal Agent

多平台 AI Agent 系统，参考 Hermes 架构从零构建。

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 LLM API Key 和平台凭据

# 3. CLI 测试
uv run python -m personal_agent --cli "你好，1+1等于几？"

# 4. 启动服务（飞书 / Telegram）
uv run python -m personal_agent
```

## 架构

```
用户消息（飞书/Telegram）
    │
    ▼
适配器（BasePlatformAdapter）──→ 平台注册表自发现
    │ create_task（非阻塞）
    ▼
Gateway（中央调度器）
    ├─ 认证 → 命令检测 → 忙检查 → Agent 调度
    │
    ▼
Agent 引擎（while 循环）
    ├─ LLM 调用 → 解析 tool_calls → 执行工具 → 继续
    ├─ 3 层消息模型：history → messages → api_messages
    └─ 3 种重试 / 压缩 / 记忆注入
```

## 技术栈

- Python 3.12+ / uv / asyncio
- DeepSeek Anthropic-compatible API
- SQLite + aiosqlite
- lark-oapi（飞书）/ python-telegram-bot
- pydantic-settings / httpx / structlog

## 内置工具

| 工具 | 功能 |
|---|---|
| `calculator` | 安全数学表达式求值 |
| `web_search` | DuckDuckGo 搜索 |
| `web_fetch` | URL 抓取转 Markdown |
| `datetime` | 日期/时间/时区 |
| `file_read` | 读本地文件 |
| `file_write` | 写本地文件 |
| `memory` | 用户记忆增删查 |
| `todo` | 待办事项管理 |

## 配置

```env
# LLM
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com/anthropic
LLM_MODEL=deepseek-chat

# 飞书
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx

# Telegram（可选）
TELEGRAM_BOT_TOKEN=xxx
```

哪个平台有凭据就启用哪个。

## 扩展点

- **多 Provider**：`BaseTransport` 抽象基类 + `ProviderProfile`，加 OpenAI 只加一个 `ChatCompletionsTransport`
- **多平台**：`PlatformRegistry` 自注册，加 Discord 只写两个文件
- **多工具/Skill**：`ToolRegistry` / `SkillRegistry`，import 即注册
- **Agent 缓存**：`_get_or_create_agent` 接口预留，LRU + 哨兵模式
- **钩子**：6 个挂载点（消息/LLM/工具/发送），简单 callback 列表
- **压缩/Memory/Cron**：抽象基类 + config 驱动策略选择

## 测试

```bash
uv run pytest tests/ -v
```

## 目录结构

```
src/personal_agent/
├── main.py              # 入口
├── config.py            # 配置
├── models/              # 数据模型
├── db/                  # SQLite 持久化
├── llm/                 # LLM 传输层（BaseTransport + Provider）
├── tools/               # 工具系统（注册表 + 管道 + 内置工具）
├── skills/              # Skill 系统（三层披露）
├── agent/               # Agent 引擎（while 循环 + 压缩 + 钩子）
├── memory/              # 记忆系统（Provider + Manager）
├── compression/         # 上下文压缩
├── adapters/            # 平台适配器（飞书 + Telegram）
├── cron/                # 定时任务（预留）
└── gateway/             # 中央调度器 + SessionStore
```
