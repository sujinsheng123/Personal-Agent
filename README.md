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

## 飞书 Bot 配置

1. 进入[飞书开发者后台](https://open.feishu.cn/app)，创建企业自建应用
2. 添加应用能力 → 开启「机器人」
3. 事件订阅 → 添加 `im.message.receive_v1`（接收消息）事件
4. 权限管理 → 添加 `im:message:read_as_bot`、`im:message:send_as_bot`
5. 发布版本 → 管理员审核通过
6. 在飞书中搜索你的 Bot 名称，发消息测试

## 测试

```bash
# 单元测试（mock，不花 API 费）
uv run pytest tests/ -v

# CLI 测试（真实 API 调用）
uv run python -m personal_agent --cli "你好，1+1等于几？"

# 启动飞书 Bot（WebSocket 长连接）
uv run python -m personal_agent

# 启动后给 Bot 发消息，观察日志输出
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
