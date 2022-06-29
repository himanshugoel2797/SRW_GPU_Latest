## Compilation on Windows

- Install CUDA Toolkit
- Install CuPy for associated CUDA version (https://docs.cupy.dev/en/stable/install.html)
- Open the solution in visual studio as normal
- Set build mode to 'Debug_cuda', architecture to 'x64'
- In 'Solution Explorer' in VS, on SRWLIB, right click, go to Build Dependencies -> Build Customizations, make sure the option for the associated CUDA version is checked (e.g. 'CUDA 11.6(.targets, .props)')
- Build as normal

## Build Steps on gpu-002:

```bash
git clone https://github.com/himanshugoel2797/SRW_GPU_Latest.git
conda activate gcc9-conda-forge
cd SRW_GPU_Latest
. setup.sh
make
```

## Perlmutter Notes
```bash
module load PrgEnv-gnu
module load cudatoolkit
module load gcc/10.3.0
module load python
conda create -n srw-test python-3.9 pip numpy scipy scikit-learn matplotlib h5py
conda active srw-test
pip install cupy-cuda114
git clone https://github.com/himanshugoel2797/SRW_GPU_Latest.git
git checkout paper_only
# Uncomment PYPATH=/usr and set PYFLAGS for python3.6m in cpp/gcc/Makefile
cd SRW_GPU_Latest
. setup.sh
make all
```

Note: module load gcc/10.3.0 necessary to workaround a bug in gcc-11 and conda. Eventually may not need it.

# Running GPU Code
To run with GPU enabled, the 'SRW_ENABLEGPU' environment variable needs to be set.
An easy way to do this is shown in `env/work/srw_python/SRWLIB_Example19_CMD_Batched_FC.py`

```python
from __future__ import print_function
import os
import time

os.environ['SRW_ENABLEGPU'] = "1"

from tabnanny import check #Python 2.7 compatibility
from srwlib import *
from srwl_uti_smp import *
import srwl_uti_smp_rnd_obj3d
import matplotlib.pyplot as plt
from uti_plot import * #required for plotting

if useCuPy:
    import cupy as cp
else:
    import numpy as cp
```