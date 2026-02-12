# Nanobot Dashboard

Nanobot 可视化面板 - 基于 FastAPI + HTMX + Alpine.js 的 Web 管理界面。

## 功能特性

- **仪表板**: 系统概览、统计数据、最近执行记录
- **技能管理**: 查看、启用/禁用、手动触发技能
- **调度任务**: Cron 任务时间线、下次执行时间、手动执行
- **实时日志**: SSE 实时日志流、按会话/级别筛选、关键词搜索

## 技术栈

- **后端**: FastAPI (Python 异步框架)
- **模板**: Jinja2
- **前端交互**: HTMX + Alpine.js
- **样式**: 原生 CSS (深色主题)

## 安装

### 1. 创建虚拟环境

```bash
cd nanobot-dashboard
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
python main.py
```

或使用 uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## 使用

启动后访问: http://localhost:8080

### 页面说明

| 页面 | 路径 | 功能 |
|------|------|------|
| 仪表板 | `/` | 系统概览、统计卡片、最近执行 |
| 技能管理 | `/skills` | 技能列表、启用/禁用、手动触发 |
| 调度任务 | `/schedules` | 时间线视图、下次执行、手动运行 |
| 实时日志 | `/logs` | 日志流、筛选、实时更新 |

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/stats` | GET | 获取统计数据 |
| `/api/skills` | GET | 获取技能列表 |
| `/api/skills/{id}/toggle` | POST | 切换技能状态 |
| `/api/skills/{id}/trigger` | POST | 手动触发技能 |
| `/api/schedules` | GET | 获取调度任务列表 |
| `/api/schedules/{id}/toggle` | POST | 切换调度状态 |
| `/api/schedules/{id}/run` | POST | 手动执行调度 |
| `/api/logs` | GET | 获取日志 (支持筛选) |
| `/api/logs/stream` | GET | SSE 实时日志流 |
| `/api/logs/sessions` | GET | 获取会话列表 |

## 项目结构

```
nanobot-dashboard/
├── main.py              # FastAPI 入口
├── api/                 # API 路由
│   ├── __init__.py
│   ├── skills.py        # 技能管理 API
│   ├── schedules.py     # 调度管理 API
│   └── logs.py          # 日志管理 API
├── static/              # 静态文件
│   └── css/
│       └── style.css    # 样式表
├── templates/           # Jinja2 模板
│   ├── base.html        # 基础模板
│   ├── dashboard.html   # 仪表板
│   ├── skills.html      # 技能管理
│   ├── schedules.html   # 调度任务
│   └── logs.html        # 实时日志
├── requirements.txt     # 依赖
└── README.md           # 说明文档
```

## 数据源

面板直接读取 nanobot 的数据文件:

- **技能**: `~/.nanobot/skills/*/` 目录
- **调度任务**: `~/.nanobot/cron/jobs.json`
- **日志**: `~/.nanobot/sessions/*.jsonl`

## 开发

### 添加新页面

1. 在 `templates/` 创建 HTML 模板
2. 在 `main.py` 添加路由
3. 如需 API，在 `api/` 创建路由文件

### 样式修改

编辑 `static/css/style.css`，使用 CSS 变量可快速调整主题色。

## 配置

默认端口: 8080

修改 `main.py` 中的 uvicorn 配置:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
```

## 注意事项

- 面板以只读模式访问 nanobot 数据，不会修改原始配置
- 技能启用/禁用通过修改 SKILL.md frontmatter 实现
- 日志实时更新使用 SSE，无需刷新页面
- 默认绑定 localhost，如需远程访问请修改 host

## License

MIT
