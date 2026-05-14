# 群决策分析多智能体系统

Multi-Agent Group Decision Analyzer（GDA）

基于社会选择理论、多准则决策（MCDM）、博弈论和行为决策理论的多智能体群决策分析系统。通过 **计算引擎 + LLM 智能体辩论** 的方式，对复杂群决策场景进行从建模到建议的全流程分析。

## 项目流程

本系统覆盖群决策分析的完整闭环，分为四个阶段：

```
前分析（建模）  →  核心计算  →  智能体辩论  →  后分析（建议）
```

### 阶段一：前分析——问题建模

建模智能体（ModelerAgent）接收场景描述后，自动完成：
- **利益相关方识别**：从场景中提取关键参与方，评估其角色、利益和影响力
- **备选方案生成**：建议覆盖决策空间的方案集
- **决策准则构建**：提炼可量化的评估维度
- **偏好引导**：在用户输入偏好排序 / 评分矩阵时追问依据，确保输入质量

### 阶段二：核心计算——多方法并行分析

纯 Python 计算引擎（无需 LLM API）并行运行多种方法：

| 方法类别 | 具体方法 | 分析内容 |
|----------|----------|----------|
| 社会选择 | 孔多塞、Borda、Copeland | 偏好排序聚合、循环检测、赢家识别 |
| 理论检验 | Arrow 四条件、G-S 定理 | 投票方法的数学边界 |
| 多准则决策 | TOPSIS、AHP | 基于评分矩阵和准则权重的精细评估 |
| 博弈论 | 纳什均衡检测 | 策略稳定性分析 |
| 行为决策 | 群体思维诊断、前景理论 | 认知偏差与风险偏好 |
| 敏感性分析 | 准则权重遍历 | 排名稳健性检验 |

### 阶段三：智能体辩论——多视角交锋

六个专用 LLM 智能体以圆桌讨论模式运行（非串行流水线），彼此可质疑、反驳、补充：

1. **AnalystAgent** 解读计算数据，呈现关键发现
2. **CriticAgent** 从 Arrow 定理、策略性投票等角度质疑
3. **NarrativeAgent** 引入历史案例类比
4. **DeliberationAgent** 模拟利益相关方言论交换和立场变化
5. **CoordinatorAgent** 综合各方观点，诊断问题，给出方向建议
6. **ModelerAgent**（前分析阶段）识别利益相关方、建议方案和准则

### 阶段四：后分析——决策建议

- **协商模拟**：基于分析结果模拟各方立场变化
- **敏感性分析**：自动遍历准则权重，检验结论稳健性
- **决策建议**：协调智能体综合辩论和方法对比，生成结构化建议

## 核心计算引擎（core/）

| 模块 | 方法 | 说明 |
|------|------|------|
| `condorcet.py` | 孔多塞分析 | 两两配对比较、循环检测、赢家识别 |
| `borda.py` | Borda 计票 | 排序投票积分制 |
| `copeland.py` | Copeland 方法 | 胜场 − 负场计分 |
| `arrow_analysis.py` | Arrow 不可能定理 | 四条件检验（无限制域、帕累托效率、IIA、非独裁性） |
| `topsis.py` | TOPSIS 方法 | 距正/负理想解距离，相对接近度排序 |
| `ahp.py` | AHP 层次分析法 | 特征值法权重计算、CR 一致性检验 |
| `mcdm_sensitivity.py` | MCDM 敏感性分析 | 准则权重遍历、排名稳定性、龙卷风图 |
| `game_theory.py` | 纳什均衡 | 纯策略均衡检测、最优响应计算 |
| `behavioral.py` | 行为决策理论 | Janis 八症状群体思维诊断、前景理论风险域分类 |
| `consensus.py` | 共识度分析 | 投票人偏好相似度矩阵、共识度计算 |
| `power_index.py` | 权力指数 | Shapley-Shubik 与 Banzhaf 投票权力指数 |
| `non_cooperative.py` | 非合作行为检测 | 偏离度雷达图、疑似策略性投票者标记 |
| `simulation.py` | 动态推演 | 增删投票人、偏好变化下的结果追踪 |
| `visualization.py` | 可视化图表 | 15+ 种 Plotly 交互图表 |
| `models.py` | 数据模型 | Ballot、Voter、DecisionMatrix 等核心数据结构 |

## LLM 智能体（agents/）

| 智能体 | 角色 | 核心职责 |
|--------|------|----------|
| `modeler_agent.py` | 问题架构师 | 识别利益相关方、建议备选方案、构建决策准则 |
| `analyst_agent.py` | 数据解读员 | 解读孔多塞矩阵、Borda/Copeland 分数、TOPSIS/AHP 排名 |
| `critic_agent.py` | 魔鬼代言人 | 检验 Arrow 四条件、诊断群体思维、策略性投票分析 |
| `narrative_agent.py` | 案例考古学家 | 历史案例类比（法国大选、美国大选等） |
| `deliberation_agent.py` | 协商模拟器 | 模拟利益相关方立场变化和论据交换 |
| `coordinator_agent.py` | 总指挥 | 编排全流程、管理辩论秩序、生成综合建议 |
| `debate.py` | 辩论管理器 | 圆桌讨论调度、发言顺序管理、观点追踪 |

**辩论机制**：智能体以圆桌讨论模式运行（非串行流水线 A→B→C→D），每个智能体可见前人发言并可以引用/反驳/补充。最终由协调智能体总结各方观点，形成多视角综合判断。

## 案例库（cases/data/）

| 案例文件 | 场景 | 核心教学点 |
|----------|------|------------|
| `openai_coup.json` | OpenAI 董事会政变（9方×6方案×4准则） | 多属性群决策、TOPSIS/AHP、敏感性分析 |
| `condorcet_paradox.json` | 经典孔多塞悖论 | 循环偏好最基本形式 |
| `arrow_demo.json` | Arrow 定理演示 | IIA 违反实例 |
| `french_election_2002.json` | 2002 法国大选 | 多数决失败、极端候选人进入决选 |
| `us_electoral_2000.json` | 2000 美国大选 | 选举人团 vs 普选票、制度设计影响 |

## 可视化图表

系统提供 15+ 种 Plotly 交互图表，覆盖社会选择、MCDM、行为分析和 GDM 四个类别：

**社会选择 / 投票**
- 两两配对比较热力图
- Borda 分数柱状图
- Copeland 得分分组柱状图
- 优势关系有向图（循环路径高亮标红）
- 动态推演时间线折线图

**多准则决策（MCDM）**
- 决策矩阵热力图（方案 × 准则评分）
- 准则权重雷达图（多投票人对比）
- MCDM 排名对比热力图
- 敏感性龙卷风图
- AHP 方案综合得分柱状图

**行为 / 心理**
- 群体思维八症状雷达图
- 前景理论风险域四象限散点图

**大规模群决策（GDM）**
- 投票人偏好相似度热力图
- Shapley-Shubik / Banzhaf 权力指数对比
- 有界信任意见动力学收敛图
- 非合作行为偏离度雷达图

## 环境配置

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

### 配置 LLM API（可选）

智能体辩论功能需要 DeepSeek API Key：

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=your_key_here
```

未配置 API Key 时，计算引擎和可视化图表仍可正常使用，仅智能体辩论功能降级。

## 运行指南

### 启动 Web 界面

```bash
conda activate gda
cd group_decision_analyzer
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

### 界面页面

| 页面 | 功能 |
|------|------|
| 决策建模 | 选择预设案例或自定义偏好、查看利益相关方与方案配置 |
| 分析结果 | 查看孔多塞、Borda、Copeland、Arrow 结果及智能体辩论记录 |
| 动态推演 | 增删投票人、修改偏好，观察结果变化 |
| 敏感性分析 | 遍历准则权重，检验 MCDM 排名稳健性 |
| 历史案例 | 浏览 5 个经典案例，一键加载到分析 |
| 理论参考 | 群决策核心理论公式与概念解释 |

### 生成分析报告

项目根目录下的 `generate_report.py` 可生成完整的 .docx 案例分析报告：

```bash
conda activate gda
cd /path/to/class_project
python generate_report.py
```

输出：`案例分析作业方案.docx`

## 项目结构

```
group_decision_analyzer/
├── app.py                    # Streamlit 入口
├── requirements.txt          # Python 依赖
├── .env.example              # API Key 配置模板
├── core/                     # 纯计算引擎（无 LLM 依赖）
│   ├── models.py             # Ballot / Voter / DecisionMatrix 数据模型
│   ├── condorcet.py          # 孔多塞配对比较、循环检测
│   ├── borda.py              # Borda 计票
│   ├── copeland.py           # Copeland 方法
│   ├── arrow_analysis.py     # Arrow 不可能定理四条件检验
│   ├── topsis.py             # TOPSIS 多准则决策
│   ├── ahp.py                # AHP 层次分析法
│   ├── mcdm_sensitivity.py   # MCDM 敏感性分析
│   ├── game_theory.py        # 纳什均衡检测
│   ├── behavioral.py         # 群体思维诊断 + 前景理论
│   ├── consensus.py          # 共识度分析
│   ├── power_index.py        # 投票权力指数
│   ├── non_cooperative.py    # 非合作行为检测
│   ├── simulation.py         # 动态推演
│   └── visualization.py      # 15+ Plotly 图表
├── agents/                   # LLM 多智能体系统
│   ├── base_agent.py         # DeepSeek API 基类
│   ├── modeler_agent.py      # 建模智能体
│   ├── analyst_agent.py      # 分析智能体
│   ├── critic_agent.py       # 批判智能体
│   ├── narrative_agent.py    # 叙事智能体
│   ├── deliberation_agent.py # 协商模拟智能体
│   ├── coordinator_agent.py  # 协调智能体
│   └── debate.py             # 辩论管理器
├── cases/                    # 案例库
│   ├── case_library.py       # 案例加载与管理
│   └── data/                 # 5 个 JSON 案例文件
└── ui/                       # Streamlit 界面
    ├── sidebar.py            # 侧边栏导航
    ├── page_modeling.py      # 决策建模页
    ├── page_analysis.py      # 分析结果页
    ├── page_simulation.py    # 动态推演页
    ├── page_sensitivity.py   # 敏感性分析页
    ├── page_cases.py         # 历史案例页
    └── page_theory.py        # 理论参考页
```

## 理论基础

- **Arrow 不可能定理**（1951）—— 方案数 ≥ 3 时不存在完美的社会选择函数
- **Gibbard-Satterthwaite 定理**（1973）—— 任何非独裁投票规则均可被策略性操纵
- **孔多塞方法**（1785）—— 两两对决多数决，赢家击败所有对手
- **Borda 计票**（1770）—— 排序投票的位置积分制
- **Copeland 方法**（1951）—— 基于胜场 − 负场的计分
- **TOPSIS 方法**（1981）—— 距理想解最近、距负理想解最远的多准则排序
- **AHP 层次分析法**（Saaty, 1980）—— 判断矩阵特征值法，CR 一致性检验
- **纳什均衡**（1950）—— 无人可通过单方面改变策略获益
- **群体思维**（Janis, 1972）—— 凝聚力过高导致批判性思考丧失的八症状
- **前景理论**（Kahneman-Tversky, 1979）—— 损失域风险寻求、收益域风险规避
- **Habermas 理想协商条件** —— 包容性、理由说明、无强制的程序正义标准

## License

MIT
