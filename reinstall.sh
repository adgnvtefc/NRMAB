#!/usr/bin/env bash
set -e

# 1. 删除旧虚拟环境
rm -rf venv                                                      # 删除旧的虚拟环境目录 :contentReference[oaicite:6]{index=6}

# 2. 创建新虚拟环境
python3 -m venv venv                                             # 利用 Python 内置 venv 模块创建隔离环境 :contentReference[oaicite:7]{index=7}

# 3. 激活并升级打包工具
source venv/bin/activate                                         # 激活新环境 :contentReference[oaicite:8]{index=8}
pip install --upgrade pip wheel setuptools                       # 升级 pip、wheel、setuptools :contentReference[oaicite:9]{index=9}

# 4. 安装 PyTorch 2.5.0 + CUDA 12.1 GPU 轮子
pip install \
  torch==2.5.0+cu121 \
  torchvision==0.20.0+cu121 \
  torchaudio==2.5.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121             # 从官方 CUDA 12.1 仓库安装 GPU 版 PyTorch :contentReference[oaicite:10]{index=10}

# 5. 安装其余项目依赖
pip install -r requirements.txt                                  # 安装 requirements.txt 中的其他包 :contentReference[oaicite:11]{index=11}

# 6. 安装 PyTorch Geometric 及其 GPU 扩展
pip install torch_geometric                                      # 安装 PyG 核心库 :contentReference[oaicite:12]{index=12}
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html              # 安装 PyG GPU 高性能扩展 :contentReference[oaicite:13]{index=13}

# 7. 验证安装
python - << 'EOF'
import torch, torch_geometric
print("CUDA 可用:", torch.cuda.is_available())                   # 检查 CUDA 可用性 :contentReference[oaicite:14]{index=14}
print("Torch 版本:", torch.__version__, "CUDA 版本:", torch.version.cuda)
print("PyG 版本:", torch_geometric.__version__)
EOF
