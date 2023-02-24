# نصب و راه اندازی PyTorch

<div align="center">
    <a href="https://github.com/aidinism/deep-learning">
        <img src="https://raw.githubusercontent.com/aidinism/deep-learning/main/images/pytorch.png" width=750 alt="pytorch deep learning">
    </a>
</div>

برای استفاده از PyTorch از 2 روش میتونید استفاده کنید:

1.استفاده از  [Google Colab](https://colab.research.google.com/): اگر دسترسی دارید گزینه راحت تری هست (ممکنه در ایران قابل استفاده نباشه 😔 )

2.نصب محلی: اگر دسترسی ندارید نگران نباشید 😊 این روش هم به راحتی قابل انجام هست و البته امکانات بیشتری بهتون میده



<br/>
<br/>

## 1. استفاده از Google Colab

گوگل کولب یک کامپیوتر آنلاین (با رابط ژوپیتر نوت بوک) هست که به طور رایگان در اختیار همه قرار داره.
از این آدرس میتونید دسترسی پیدا کنید: 

https://colab.research.google.com

---
---
---

## 2. نصب و راه اندازی محلی (Linux , Windows WSL)

این راهنما برای نصب در محیط لینوکس هست (یا WSL در ویندوز). برای نصب در ویندوز یا مکینتاش به [راهنمای نصب رسمی PyTorch](https://pytorch.org/get-started/locally/) مراجعه کنید.

### **نکته:** اگر GPU NVIDIA دارید میتوانید 


اگر از WSL 2 استفاده میکنید از راهنمای زیر میتونید CUDA رو نصب کنید:

https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl


## Setup steps locally for a Linux system with a GPU

1. Check CUDA install: [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
![CUDA](https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png)


2. [Install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (you can use Anaconda if you already have it), the main thing is you need access to `conda` on the command line. Make sure to follow all the steps in the Miniconda installation guide before moving onto the next step.
3. Make a directory for the course materials, you can name it what you want and then change into it. For example:
```
mkdir ztm-pytorch-course
cd ztm-pytorch-course
```
4. Create a `conda` environment in the directory you just created. The following command will create a `conda` enviroment that lives in the folder called `env` which lives in the folder you just created (e.g. `ztm-pytorch-course/env`). Press `y` when the command below asks `y/n?`.
```
conda create --prefix ./env python=3.8
```
5. Activate the environment you just created.
```
conda activate ./env
```
6. Install the code dependencies you'll need for the course such as PyTorch and CUDA Toolkit for running PyTorch on your GPU. You can run all of these at the same time (**note:** this is specifically for Linux systems with a NVIDIA GPU, for other options see the [PyTorch setup documentation](https://pytorch.org/get-started/locally/)):
```
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.3 -y
conda install -c conda-forge jupyterlab torchinfo torchmetrics -y
conda install -c anaconda pip -y
conda install pandas matplotlib scikit-learn -y
```
7. Verify the installation ran correctly by running starting a Jupyter Lab server:

```bash
jupyter lab
```

8. After Jupyter Lab is running, start a Jupyter Notebook and running the following piece of code in a cell.
```python
import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo, torchmetrics

# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())
```



اگر در نصب به مشکل برخوردید میتونید در  [Learn PyTorch GitHub Discussions page](https://github.com/aidinism/deep-learning/discussions/2) سوال بپرسید و یا از [PyTorch setup documentation page](https://pytorch.org/get-started/locally/) استفاده کنید.