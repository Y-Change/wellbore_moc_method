# NotebookLM 深度研究辅助工作流方案

> 版本：v1.0 | 2026-07-12
> 工具链：notebooklm-py CLI + Claude Code + 本地 Markdown 整理

---

## 一、总览

```
本地分析文档 → [Source Add] → NotebookLM 笔记本
                                  ├─ [Deep Research] → 网络文献源 + 研究报告
                                  ├─ [Chat Q&A]     → 结构化问答（含文献引证）
                                  └─ [History Save]  → 问答记录存为笔记
                                         ↓
                              本地 Markdown 整理 → 最终交付文档
```

核心思路：**NotebookLM 负责"广撒网"（深度搜索网络文献 + 生成研究报告 + 带引证的问答），本地负责"精加工"（公式推导、实验数据对照、格式化排版）。**

---

## 二、前置条件

### 2.1 环境准备

```bash
pip install notebooklm-py
notebooklm login          # 浏览器 OAuth 认证（仅首次）
notebooklm auth check     # 验证认证有效
```

### 2.2 认证状态文件

| 文件 | 作用 |
|------|------|
| `~/.notebooklm/profiles/default/storage_state.json` | Google OAuth Cookies（26 个），由 `login` 写入 |
| `~/.notebooklm/profiles/default/context.json` | 当前活跃笔记本 ID + 对话 ID |

---

## 三、五步工作流

### Step 1：新建笔记本

```bash
notebooklm create "研究主题名称" --json
# → 返回 notebook_id (UUID)
notebooklm use <notebook_id>   # 设为当前上下文
```

**命名规范**：`{领域}-{子课题}-{视角/方法}`，如 `水击倒谱全参数链-从物理仿真到裂缝分辨率`。

### Step 2：导入本地分析文档

```bash
notebooklm source add "./path/to/analysis.md" --json
notebooklm source wait <source_id> --timeout 120
```

**建议导入顺序**：
1. 已有自研分析文档（如 THEORETICAL_ANALYSIS.md）
2. 上一轮 NotebookLM 研究报告（如 NOTEBOOKLM_RESEARCH_REPORT.md）
3. 实验数据摘要（如 ANALYSIS.md）

### Step 3：启动深度研究

```bash
notebooklm source add-research "研究查询语句" \
    --mode deep \
    --no-wait \
    --notebook <notebook_id>

# 等待完成（15–30 min）
notebooklm research wait --import-all --timeout 1800
```

**查询语句编写原则**：
- 用英文写（Google 搜索网络文献效果更好）
- 覆盖所有关键参数和技术术语
- 明确要求包含数学推导（`Include mathematical derivations`）

**示例**：
```
Comprehensive parameter chain analysis for water hammer cepstrum fracture
identification: (1) MOC discretization and CFL constraint... (10) final
fracture spacing discrimination capability Δd_min=a/(2*B_coh). Include
mathematical derivations connecting all parameters.
```

### Step 4：结构化多层问答

```bash
# 首轮（自动创建对话）
notebooklm ask "Layer 0-2 的问题..." --notebook <id>

# 后续轮次（续接同一对话）
notebooklm ask "Layer 3-5 的问题..." -c <conversation_id>

# 保存完整对话为笔记
notebooklm history --save --note-title "七层参数链问答记录"
```

**问答设计原则**：
- 按逻辑层拆分（一次 2–3 层，问题太长回答质量下降）
- 要求 LaTeX 公式 + 文献引证
- 每层结尾要求说明"输出到下一层的内容"
- 使用 `-c <conversation_id>` 续接同一对话，保持上下文连贯

### Step 5：保存 & 本地整合

```bash
# 保存研究报告全文
notebooklm source fulltext <report_source_id> -o "./research_report.md"

# 保存问答历史
notebooklm history --save --note-title "问答记录"

# 本地整理为最终文档
```

**最终文档结构**（以 PARAMETER_CHAIN.md 为例）：

```
1. 元信息头（时间、方法、数据源）
2. Mermaid 参数依赖总图
3. Layer 0–7 逐层展开（公式 + 数值示例 + 表格）
4. 工程速查表
5. 参数敏感度矩阵
6. 文件索引
```

---

## 四、笔记本生命周期管理

```
创建 → 导入源 → 深度研究 → 问答 → 导出 → [保留/删除]
                                              ↓
                              保留：作为知识库累积，下次可直接复用源
                              删除：notebooklm notebook delete <id>
```

**建议**：每个研究主题保留一个笔记本，累积多轮研究的源和笔记，形成该主题的知识网络。

---

## 五、与本地分析的分工边界

| 维度 | NotebookLM | 本地 (Claude Code + Python) |
|------|-----------|---------------------------|
| 网络文献搜索 | **负责** — Deep Research 搜索 AIP/MDPI/ResearchGate 等 | — |
| 研究报告生成 | **负责** — 自动综述 + 引证 | — |
| 带引证的结构化问答 | **负责** — Chat with sources | — |
| 公式严格推导 | 辅助（可能有符号错误） | **负责** — 从代码物理参数出发的严格推导 |
| 实验数据对照 | 辅助（基于导入的文档） | **负责** — 直接运行 MOC 仿真 + 解析 metrics.json |
| Mermaid/图表绘制 | — | **负责** |
| 格式化排版 | — | **负责** — GFM 表格、LaTeX 块公式、交叉引用 |
| 工程速查表 | — | **负责** — 代入参数的具体数值计算 |

### 分工原则

> **NotebookLM = 广度引擎**：搜索文献、生成综述、提供引证、验证思路是否与学界一致。
> **本地 = 深度引擎**：严格公式推导、实验数据驱动验证、精确数值计算、格式化交付。

---

## 六、常见问题

### Q1: 深度研究超时或导入失败？
```bash
# 部分源导入超时时，CLI 会自动重试并跳过已导入的源
# 若仍有遗漏，手动检查 notebooklm source list
```

### Q2: 问答质量不够深入？
- 将问题拆分为更小的层（每次 1–2 个子问题）
- 明确要求 "Present all derivations in LaTeX"
- 使用 `-c` 保持对话连续，让模型基于前面的回答深化

### Q3: 如何复用已有的源？
```bash
notebooklm source list --json   # 查看所有源
notebooklm ask "..." -s <src_id1> -s <src_id2>  # 指定源回答
```

### Q4: 多项目并行时如何隔离？
```bash
# 方式 A：不同笔记本（推荐）
notebooklm create "项目A"  # → id_A
notebooklm create "项目B"  # → id_B
notebooklm ask "..." --notebook id_A  # 显式指定

# 方式 B：不同 profile
export NOTEBOOKLM_PROFILE=project_a
notebooklm login
```

---

## 七、本次"水击倒谱全参数链"案例复盘

| 步骤 | 耗时 | 产出 |
|------|------|------|
| Step 1: 创建笔记本 | < 1 min | `e274ce4b` |
| Step 2: 导入 2 篇本地分析 | ~1 min | 2 sources |
| Step 3: 深度研究（13 源） | ~20 min | 1 report + 12 web sources |
| Step 4: 3 轮结构化问答（7 层） | ~15 min | 完整对话记录 + 笔记 |
| Step 5: 本地整合 PARAMETER_CHAIN.md | ~10 min | 最终交付文档 |
| **合计** | **~50 min** | **1 笔记本 + 1 研究综述 + 1 综合参数链文档** |
