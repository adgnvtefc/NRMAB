#!/usr/bin/env bash
set -e

rm -rf venv                                                      

python3 -m venv venv                                             

source venv/bin/activate                                         
pip install --upgrade pip wheel setuptools                       

pip install \
  torch==2.5.0+cu121 \
  torchvision==0.20.0+cu121 \
  torchaudio==2.5.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121             

pip install -r requirements.txt                                  

pip install torch_geometric                                      
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html              

python - << 'EOF'
import torch, torch_geometric
print("CUDA available:", torch.cuda.is_available())                   
print("Torch version:", torch.__version__, "CUDA version:", torch.version.cuda)
print("PyG version:", torch_geometric.__version__)
EOF
