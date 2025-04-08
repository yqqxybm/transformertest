# Transformer 实现项目

这是一个基于 PyTorch 实现的 Transformer 模型项目。

## 项目结构

```
.
├── src/
│   ├── model/              # 模型相关代码
│   │   ├── attention.py    # 多头注意力机制
│   │   ├── decoder.py      # 解码器实现
│   │   ├── encoder.py      # 编码器实现
│   │   ├── ffn.py          # 前馈神经网络
│   │   ├── mask.py         # 掩码生成
│   │   ├── position.py     # 位置编码
│   │   └── transformer.py  # Transformer 主模型
│   ├── config/             # 配置文件
│   │   └── model_config.py # 模型配置参数
│   ├── utils/              # 工具函数
│   │   └── data_loader.py  # 数据加载器
│   ├── test.py             # 测试脚本
│   └── train.py            # 训练脚本
├── requirements.txt        # 项目依赖
└── environment.yml         # Conda 环境配置
```

## 环境要求

- Python 3.9
- PyTorch >= 1.9.0
- NumPy >= 1.19.2

## 安装

1. 使用 Conda 创建环境：
```bash
conda create -n transformer python=3.9
conda activate transformer
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速测试

目前支持快速测试功能，可以验证模型的基本功能：

```bash
cd src
python test.py
``` 