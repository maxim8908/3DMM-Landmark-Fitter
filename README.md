# 3DMM-Landmark-Fitter

A PyTorch-based implementation of a **3D Morphable Model (3DMM)** landmark fitting pipeline.  
This tool fits PCA-based 3D face models to target meshes using sparse 3D landmark constraints — a core component of facial analysis, scan alignment, and identity modeling workflows.

---

## Features

- Fit 3DMM shape coefficients from sparse 3D landmarks  
- Supports `.obj` meshes, `.npy` vertex data, and offset reconstruction  
- Optional per-landmark weighting and L2 regularization  
- Loss tracking and landmark error reporting  
- Easily scriptable for batch fitting

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/3DMM-Landmark-Fitter.git
cd 3DMM-Landmark-Fitter
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
