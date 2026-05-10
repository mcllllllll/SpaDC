# SpaDC

![SpaDC_Overview](https://raw.githubusercontent.com/mcllllllll/SpaDC/master/SpaDC_Overview.png)


## Overview

SpaDC is designed to integrate both spatial location and DNA sequence for comprehensive ATAC data analysis.

SpaDC integrates spatial location and DNA fragment. It employs a deep 1D convolutional neural network to predict chromatin accessibility and generate low-dimensional cell embeddings. Additionally, it incorporates spatial graph constraints and MNN-based batch correction to facilitate feature extraction and support various biological applications, including spatial epigenomics data denoising, spatial epigenomics data integration, GRN inference. 



## Installation 

```
git clone https://github.com/mcllllllll/SpaDC.git
cd SpaDC-main
pip install -r requirement.txt
python setup.py install
```



## Tutorials

Tutorials demonstrating how to use SpaDC on different datasets are included in the Tutorials folder.

- Tutorial 1: [Running experiments on the MISAR dataset](https://github.com/mcllllllll/SpaDC/blob/master/Tutorials/Tutorial_MISAR.ipynb)
- Tutorial 2: [Running experiments on the P22 dataset](https://github.com/mcllllllll/SpaDC/blob/master/Tutorials/Tutorial_P22.ipynb)
- Tutorial 3: [Integrating P22 ATAC and H3K27ac datasets](https://github.com/mcllllllll/SpaDC/blob/master/Tutorials/Tutorial_integrate.ipynb)
- Tutorial 4: [Inferring GRN on P22 dataset](https://github.com/mcllllllll/SpaDC/blob/master/Tutorials/Inferring_GRN_on_P22.ipynb)



## Support

If you have any questions, please feel free to contact us [2023202210190@whu.edu.cn](2023202210190@whu.edu.cn). 



## Citation


