# multi-agent-hub

> 多模型、多平台的智能 AI 中台系统（Multi‑Agent Hub）。  
> 统一封装模型与消息渠道，支持 Agent 协作与 RAG，帮助快速搭建企业级 AI 应用。

---

## ✨ 特性概览

- **多模型聚合**：通过统一抽象层接入/切换不同大模型供应商与模型版本。  
- **多渠道接入**：`bots/` 适配外部 IM/客服/工单等消息入口。  
- **Agent 编排**：`core/` 提供角色、会话与工具调用机制，可扩展为多智能体协作。  
- **RAG 检索增强**：`rag/` 覆盖文档切分、嵌入、索引、检索与（可选）重排。  
- **可插拔基础设施**：`infrastructure/` 抽象模型、向量库、日志、缓存等实现。  
- **持久化与会话**：`storage/` 管理对话历史、索引与任务状态，便于审计与追踪。  
- **静态资源**：`web/static/` 存放前端演示页或控制台的静态文件。  
- **应用入口**：`app/` 提供 API/Runner/CLI 等装配与启动代码。

> 以上根据仓库根目录当前可见结构整理，细节以源码为准。

---

## 📦 目录结构

```
.
├─ app/                 # 应用入口（API/服务进程/调度或 CLI）
├─ bots/                # 外部平台/IM 机器人接入
├─ core/                # 会话、Agent、工具、路由、中间件
├─ domain/              # 领域模型与业务实体（DTO/UseCase 等）
├─ infrastructure/      # 模型/向量库/日志/缓存等适配层
├─ rag/                 # 文档切分/嵌入/索引/检索/重排
├─ storage/             # 数据与会话持久化（ORM/DAO/抽象）
├─ web/static/          # 前端静态资源
├─ config_template.json # 配置模板（复制为 config.json 后使用）
├─ requirements.txt     # Python 依赖
└─ .gitignore
```

---

## 🚀 快速开始

### 1) 环境要求
- Python 3.10+
- （可选）向量数据库：如 Chroma / Qdrant / Milvus 等
- （可选）持久化组件：如 Redis / PostgreSQL / SQLite 等

### 2) 安装

```bash
git clone https://github.com/yacciiyao/multi-agent-hub-yaccii.git
cd multi-agent-hub-yaccii

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 3) 配置

```bash
cp config_template.json config.json
```

在 `config.json` 中按需填写（字段以模板为准）：

- `model_providers`：各模型供应商（如 OpenAI/Azure/Qwen 等）的 `api_key`、`base_url`、`model`。  
- `embedding`：嵌入模型与维度。  
- `vector_store`：类型与连接参数（库名/主机/端口/集合名）。  
- `storage`：数据库/文件存储配置。  
- `bots`：各渠道的 token、回调 URL、签名/加密参数。  
- `routing`：模型选择策略、负载与降级。  
- `rag`：分片大小、重叠、召回数、重排开关、知识库路径等。  

> 请将密钥放入环境变量或独立的密钥管理方案，避免写入代码库。

### 4) 运行

> 具体入口以 `app/` 实现为准，可尝试：

```bash
# 方式 A：模块启动（如果 app 定义了 __main__）
python -m app

# 方式 B：直接运行主脚本（若存在）
python app/main.py

# 方式 C：若提供 Web API（FastAPI/Flask 等），例如：
# uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

启动后根据控制台日志访问健康检查或 Swagger（若提供）。

---

## 🧠 工作流与架构

```
入站消息 ──> bots/* → app/*（路由/中间件/依赖注入）
                         ↓
                      core/*（会话 & Agent 编排 & 工具调用）
                    ┌───────┴─────────┐
                    │                 │
                  rag/*           infrastructure/*（LLM/Embeddings）
                    │                 │
                 向量库/索引         storage/*（DB/文件/缓存）
```

- **core**：维护会话状态、角色与工具调用协议；可扩展多智能体流程。  
- **rag**：文档 → 切分 → 嵌入 → 写入向量库；查询时召回/重排并注入上下文。  
- **infrastructure**：对第三方 LLM/Embedding/向量库的统一适配（鉴权、超时、重试、观测）。  
- **storage**：记录消息、检索日志与执行轨迹，支持审计与回溯。

---

## 🛠️ 开发指引

- **新增模型提供方**：在 `infrastructure/` 新增适配器，实现统一接口（鉴权/超时/重试/指标）。  
- **新增机器人渠道**：在 `bots/` 实现接入/验签，统一消息格式。  
- **新增工具**：在 `core/` 注册工具，声明参数校验与幂等；结合 RAG/搜索等能力。  
- **观测与日志**：建议为链路打点（入站→LLM→工具→存储）并接入指标看板。  
- **配置与密钥**：使用环境变量或密钥管理服务，不提交到 Git。

---

## 🧪 测试与示例（建议）
- 提供最小可运行示例：本地对话、RAG 检索、Agent 协作流程。  
- 对关键模块（适配器、召回/重排、工具协议）补充单元/集成测试。  

---

## 🔒 安全与合规（建议）
- 开启请求/响应脱敏与审计日志；对敏感字段做加密存储。  
- 按场景配置速率限制与成本预算；区分开发/测试/生产配置。  

---

## 🗺️ 路线图（可选）
- [ ] 预置更多模型与向量库适配器  
- [ ] 内置多 Agent 模板（任务分解、评审、执行器）  
- [ ] 完整 FastAPI 管理台及 OpenAPI 文档  
- [ ] 评测/对齐与成本看板  
- [ ] 示例 Bots：飞书/企微/Slack/Telegram 等  

---

## 📄 许可证

如仓库未附带 LICENSE，建议补充（例如 MIT/Apache-2.0）。请根据业务需求选择。

---

**欢迎提交 Issue/PR 共同完善！**
