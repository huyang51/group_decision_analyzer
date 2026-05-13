# 群决策分析多智能体系统

Multi-Agent Group Decision Analyzer

基于社会选择理论、博弈论和行为决策理论的多智能体群决策分析系统。通过计算引擎 + LLM 智能体辩论的方式，对群决策场景进行多维度分析。

## 功能特性

### 核心计算引擎（core/）

| 模块 | 方法 | 说明 |
|------|------|------|
| `condorcet.py` | 孔多塞分析 | 两两配对比较、循环检测、优势有向图 |
| `borda.py` | Borda 计票 | 排序投票计分 |
| `copeland.py` | Copeland 方法 | 胜场减负场计分 |
| `arrow_analysis.py` | Arrow 不可能定理 | 四条件检验：无限制域、帕累托效率、IIA、非独裁性 |
| `ahp.py` | AHP 层次分析法 | 特征值法权重计算、CR 一致性检验 |
| `game_theory.py` | 纳什均衡 | 纯策略均衡检测、最优响应计算 |
| `behavioral.py` | 群体思维诊断 | Janis 八症状检查 + 前景理论风险域分类 |

### LLM 智能体（agents/）

| 智能体 | 职责 |
|--------|------|
| 分析师 | 解读技术分析结果 |
| 建模者 | 识别利益相关者、建议方案和准则 |
| 叙事者 | 生成案例叙述和历史类比 |
| 批判者 | 从 Arrow 定理、策略性投票等角度质疑分析 |
| 协商者 | 模拟各方立场变化和妥协空间 |
| 协调者 | 规划分析流程、综合辩论生成最终建议 |

**辩论机制**：5 个智能体按顺序发言，每人可见前人发言并可引用/反驳，最终由协调者总结。

### 案例库（cases/）

- **孔多塞悖论** — 经典 3 人 3 方案循环
- **OpenAI 政变事件** — 封闭/开放两场景对比
- **Arrow 定理演示** — IIA 违反
- **2002 法国大选** — 多候选人票分裂
- **2000 美国大选** — 搅局者效应

### 可视化

- 两两配对热力图
- 优势有向图（含循环高亮）
- Borda/Copeland 分数柱状图
- 动态推演时间线
- AHP 层次结构图
- 群体思维八症状雷达图
- 前景理论风险域四象限图

## 快速开始

### 环境要求

- Python 3.11+
- conda（推荐）

### 安装

```bash
# 创建 conda 环境
conda create -n gda python=3.11 -y
conda activate gda

# 安装依赖
cd group_decision_analyzer
pip install -r requirements.txt
```

### 配置（可选）

辩论功能需要 DeepSeek API Key：

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

无 API Key 时，计算和图表功能仍可正常使用。

### 运行

```bash
conda activate gda
cd /path/to/class_project
streamlit run group_decision_analyzer/app.py
```

浏览器访问 `http://localhost:8501`。

## 使用流程

1. **建模** — 选择预设案例或自定义投票偏好
2. **核心分析** — 查看孔多塞、Borda、Copeland 结果和理论检验
3. **辩论** — 启动多智能体辩论，获取多视角分析
4. **动态推演** — 增删投票人，观察结果变化
5. **敏感性分析** — 修改单个投票人偏好，测试稳健性
6. **案例库** — 浏览经典案例，一键加载到分析

## 目录结构

```
group_decision_analyzer/
├── app.py                    # Streamlit 入口
├── requirements.txt
├── .env.example
├── core/                     # 计算引擎
│   ├── models.py             # 数据模型
│   ├── condorcet.py          # 孔多塞分析
│   ├── borda.py              # Borda 计票
│   ├── copeland.py           # Copeland 方法
│   ├── arrow_analysis.py     # Arrow 定理检验
│   ├── ahp.py                # AHP 层次分析
│   ├── game_theory.py        # 纳什均衡
│   ├── behavioral.py         # 群体思维 + 前景理论
│   ├── simulation.py         # 动态推演
│   └── visualization.py      # Plotly 图表
├── agents/                   # LLM 智能体
│   ├── base_agent.py         # DeepSeek API 基类
│   ├── coordinator_agent.py  # 协调智能体
│   ├── analyst_agent.py      # 分析智能体
│   ├── modeler_agent.py      # 建模智能体
│   ├── narrative_agent.py    # 叙事智能体
│   ├── critic_agent.py       # 批判智能体
│   ├── deliberation_agent.py # 协商智能体
│   └── debate.py             # 辩论管理器
├── cases/                    # 案例库
│   ├── case_library.py       # 案例管理
│   └── data/                 # JSON 案例文件
└── ui/                       # Streamlit 页面
    ├── sidebar.py
    ├── page_modeling.py
    ├── page_analysis.py
    ├── page_simulation.py
    ├── page_sensitivity.py
    ├── page_cases.py
    └── page_theory.py
```

## 理论基础

- **Arrow 不可能定理** — 没有完美的投票规则
- **孔多塞方法** — 两两对决的多数决
- **Borda 计票** — 排序投票的积分制
- **Copeland 方法** — 胜负场计分
- **AHP 层次分析法** — 多准则决策
- **纳什均衡** — 博弈论稳定策略
- **群体思维** — Janis 的八症状诊断
- **前景理论** — Kahneman-Tversky 损失域/收益域

## License

MIT
