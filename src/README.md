# RepoSummary - 代码仓库分析与总结工具

## 项目简介

RepoSummary 是一个智能代码仓库分析工具，能够自动分析 Java 和 Python 项目，提取代码结构、进行语义聚类，并使用大语言模型生成代码功能描述和用户故事。该工具通过多层次的分析和聚类，帮助开发者快速理解大型代码库的功能和结构。

## 主要功能

### 1. 代码结构分析
- **Java 项目分析**：支持解析 Java 文件的导入依赖和方法结构
- **Python 项目分析**：支持解析 Python 文件的函数和方法结构
- **依赖关系提取**：自动构建文件间的依赖关系图和函数调用关系图

### 2. 语义向量化
- 使用 Sentence Transformers (`all-mpnet-base-v2`) 生成代码描述的语义向量
- 支持文件和函数级别的语义表示

### 3. 智能聚类
- **文件聚类**：基于语义相似度和依赖关系对文件进行聚类
- **函数聚类**：在文件聚类基础上，进一步对函数进行特征聚类
- 使用 Leiden 算法进行社区检测，自动寻找最优聚类分辨率

### 4. 特征提取与描述生成
- 使用大语言模型（支持 OpenAI API 兼容接口）生成函数功能描述
- 生成用户故事（User Story）格式的特征描述
- 提取非功能性需求（性能、安全性等）

### 5. 结果导出
- 将分析结果导出为 CSV 格式
- 生成邻接矩阵（文件依赖、函数调用关系）
- 输出特征列表和聚类信息

## 项目结构

```
RepoSummary/
├── src/
│   ├── main.py                          # 主程序入口
│   ├── model/
│   │   └── models.py                    # 数据模型定义
│   ├── structure_analsis/               # 代码结构分析模块
│   │   ├── java/
│   │   │   ├── java_import_analyzer.py  # Java 导入分析器
│   │   │   └── java_method_analyzer.py  # Java 方法分析器
│   │   └── python/
│   │       └── python_analsis.py       # Python 分析方法
│   ├── utils/                           # 工具模块
│   │   ├── file_operations.py          # 文件操作工具
│   │   ├── file_clustering.py          # 文件聚类
│   │   ├── function_clustering.py      # 函数聚类
│   │   ├── feature_generation.py       # 特征生成
│   │   ├── method_summary.py           # 方法摘要生成
│   │   ├── clustering_utils.py         # 聚类工具函数
│   │   └── matrix_computation.py       # 矩阵计算
│   ├── out/                             # 输出目录
│   │   ├── features.csv                 # 特征列表
│   │   ├── file_adj_matrix.csv         # 文件邻接矩阵
│   │   └── func_adj_matrix.csv         # 函数邻接矩阵
│   ├── repository/                      # 克隆的仓库存储目录
│   └── requirement.md                   # 依赖列表
└── README.md                            # 项目说明文档
```

## 环境要求

### Python 版本
- Python 3.10.16

### 依赖包
所有依赖包已列在 `src/requirement.md` 文件中，主要依赖包括：

- **代码分析**：`javalang`, `jedi`
- **图分析**：`python-igraph`, `leidenalg`, `python-louvain`
- **机器学习**：`sentence-transformers`, `numpy`, `pandas`
- **大语言模型**：`openai`, `ollama`
- **可视化**：`matplotlib`
- **其他工具**：`neo4j`, `python-dotenv`, `pydantic`, `tiktoken`

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd RepoSummary
```

### 2. 安装依赖
```bash
cd src
pip install -r requirement.md
```

### 3. 环境配置
创建 `.env` 文件（在 `src` 目录下），配置大语言模型 API：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 使用方法

### 基本用法

1. **分析本地项目**：
```python
from main import main

project_root = "/path/to/your/project"
output_dir = "./out"

main(
    project_root=project_root,
    output_dir=output_dir
)
```

2. **分析 GitHub 项目**：
```python
project_root = "https://github.com/username/repo.git"
# 工具会自动克隆到 repository/ 目录
```

### 运行主程序

直接运行 `main.py`：

```bash
cd src
python main.py
```

注意：需要修改 `main.py` 中的 `project_root` 和 `output_dir` 参数。

### 配置选项

#### 函数描述生成策略
在 `main.py` 中可以选择三种策略：
- `function_name`：使用函数名作为描述（默认，最快）
- `code_t5`：使用 CodeT5 模型生成描述
- `llm`：使用大语言模型生成详细描述

#### 聚类参数
- **文件聚类**：可调整 `gamma_min`、`gamma_max`、`min_clusters` 等参数
- **函数聚类**：可调整 `weight_parameter`、`consensus_tau` 等参数

## 输出说明

### 输出文件

1. **features.csv**：包含所有提取的特征信息
   - 特征 ID、描述、流程、非功能性需求
   - 关联的函数列表和文件信息

2. **file_adj_matrix.csv**：文件依赖关系的邻接矩阵

3. **func_adj_matrix.csv**：函数调用关系的邻接矩阵

### 输出内容示例

```
Cluster ID: 0, 5 Files: ['File1.java', 'File2.java', ...]
Feature ID 0: cluster_id 0 {file1.py, file2.py}
```

## 技术特点

### 1. 多层次聚类
- 第一层：基于语义相似度的文件聚类
- 第二层：基于函数调用关系和语义的函数聚类

### 2. 自适应分辨率选择
- 自动在多个分辨率参数下测试聚类效果
- 选择最优的分辨率参数（gamma）

### 3. 智能特征描述
- 结合代码上下文和依赖关系生成描述
- 支持提取功能性和非功能性需求
- 生成符合用户故事格式的特征描述

### 4. 灵活的模型支持
- 支持多种大语言模型 API（OpenAI 兼容接口）
- 支持本地模型（Ollama）

## 注意事项

1. **API 配置**：使用 LLM 功能前必须配置 `.env` 文件
2. **并行处理**：默认关闭并行处理，在服务器环境运行时建议保持关闭
3. **内存占用**：大型项目可能需要较大内存，建议分批处理
4. **模型下载**：首次运行会自动下载 Sentence Transformers 模型

## 开发计划

- [ ] 支持更多编程语言（C++、JavaScript 等）
- [ ] 优化大型项目的处理性能
- [ ] 添加可视化界面
- [ ] 支持增量分析
- [ ] 添加更多聚类算法选项


## 贡献

欢迎提交 Issue 和 Pull Request！


