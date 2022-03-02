This repository contains two module including residual contextual and subpixel convolution network (RC-SPCNet) and post-processing (lifted multicut), developed by Hainan university and Institute of Automation, Chinese Academy of Sciences.

**Environment**

The code is developed and tested under the following configurations.

Network:

•	Hardware: more than 1 Nvidia GPUs with at least 12GB GPU memory

•	Software: CUDA>=11.0, Python>=3.7, tensorflow-gpu>=1.14.0, keras>=2.2.5

Lifted multicut:

•	Hardware: 128GB CPU memory (change the size of the processed dataset accordingly based on the memory of your machine)

•	Software: Python>=3.7, nifty, elf, vigra, numpy, imageio, napari

**Installation**

We recommend creating new environments independently.

•	Download the repository

•	Install the required python environment

•	Activate your new environment and run the code:

For the post-processing method, we suggest install the package via conda and pip as follow:

conda install -c conda-forge vigra

conda install -c conda-forge nifty   

conda install -c conda-forge python-elf 

python -m pip install imageio

python -m pip install "napari[all]"


**Test Data**

The dataset of lifted multicut can be download in：
https://pan.baidu.com/s/1_fX2FNKDXMMSWe8pQOnh4g
code：ydlo

**How to use (lifted multicut)**

Adjust the data path and run the code “postprocess-lifted multicut.py”, the result will be saved in the specified path. 

**Acknowledgement**

This project is built upon some previous projects. Especially, we'd like to thank the contributors of the following github repositories:
multicut_pipeline: https://github.com/ilastik/nature_methods_multicut_pipeline
elf: https://github.com/constantinpape/elf

**Contact**

If you meet any problem, please contact the author directly.

Email: xiaochi@hainanu.edu.cn
