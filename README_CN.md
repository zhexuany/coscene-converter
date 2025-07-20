# coScene 转换器

用于将机器人数据集转换为 MCAP 格式以便在 coScene 和 Foxglove 中使用的工具。

## 概述

coScene 转换器是一个 Python 库，它将机器人数据集（特别是来自 Open-X-Embodiment 集合的数据集）转换为 MCAP 格式，以便在 coScene 中进行可视化。该工具提供了一种灵活的基于模式的方法来处理不同的数据集结构和格式。

## 特点

- 将 Open-X-Embodiment 数据集转换为 MCAP 格式
- 支持图像、深度数据、变换和机器人状态
- 与 coScene 集成实现交互式可视化
- 可扩展的模式系统，支持不同的数据集格式
- 数据集结构探索工具

## 安装

```bash
pip install -e .
```

## 使用方法

### 转换数据集

将数据集片段转换为 MCAP 格式：

```bash
python -m cli --dataset berkeley_autolab_ur5 --episode 1
```

选项：
- `--dataset DATASET`：要转换的数据集名称（例如，berkeley_autolab_ur5, stanford_robocook_converted_externally_to_rlds）
- `--episode EPISODE`：要转换的片段编号（默认：1）
- `--batch`：批处理模式下处理多个片段
- `--start START`：批处理模式的起始片段编号（默认：1）
- `--end END`：批处理模式的结束片段编号（默认：10）
- `--output-dir OUTPUT_DIR`：生成的 MCAP 文件的输出目录（默认：mcap_files）
- `--live`：转换过程中显示实时预览
- `--rate RATE`：实时预览的播放速率，单位为赫兹（默认：5.0）
- `--verbose`：启用详细输出，包含步骤信息

### 探索数据集结构

在转换之前探索数据集的结构：

```bash
python scripts/dataset_structure_explorer.py --dataset stanford_robocook_converted_externally_to_rlds
```

**重要提示**：数据集名称必须与 Open-X-Embodiment 集合中的注册数据集名称完全匹配，因为这些名称用于从 `tensorflow_datasets` 加载数据集。您可以在 [Open-X-Embodiment 数据集电子表格](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0) 的"Registered Dataset Name"列下验证注册的数据集名称。

这将生成一个包含数据集结构的 JSON 文件，可用于创建新的模式。由于不同数据集的性质差异很大，理解每个数据集字段和结构的底层含义对于创建适当的模式至关重要。

## 创建新的数据集模式

要添加对新数据集的支持：

1. 运行数据集结构探索器以了解数据集格式：
   ```bash
   python scripts/dataset_structure_explorer.py --dataset your_dataset_name
   ```
   请记住，`your_dataset_name` 必须与 Open-X-Embodiment 集合中的注册数据集名称完全匹配。

2. 仔细分析生成的 JSON 结构以了解：
   - 数据集中每个字段的语义含义
   - 不同数据元素之间的关系
   - 数据集如何表示机器人状态、传感器数据和动作

3. 在 `common/dataset_schemas/your_dataset_name.py` 中创建新的模式文件

4. 按照现有模式（如 `berkeley_autolab_ur5.py` 或 `stanford_robocook_converted_externally_to_rlds.py`）的模式实现模式类

5. 您的模式类应该：
   - 继承自 `DefaultSchema` 或 `DatasetSchema`
   - 实现 `setup_channels()` 以定义数据集的通道
   - 实现 `process_step()` 以处理数据的每个步骤
   - 可选实现 `print_step_info()` 用于调试

## 项目结构

- `cli.py`：转换器的命令行界面
- `common/`：通用工具和模式定义
  - `schemas.py`：基本模式类和通用模式定义
  - `dataset_schemas/`：特定数据集的模式实现
- `open_x_embodiment/`：用于处理 Open-X-Embodiment 数据集的工具
  - `data_loader.py`：加载数据集的函数
  - `converter.py`：将数据集转换为 MCAP 的函数
- `scripts/`：实用脚本
  - `dataset_structure_explorer.py`：探索数据集结构的工具

## 贡献

要添加对新数据集的支持：

1. 使用探索器工具探索数据集结构
2. 基于现有示例创建新的模式文件
3. 使用转换器测试您的模式

## 许可证

Apache 2 许可证