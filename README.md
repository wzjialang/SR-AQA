# Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography

Official PyTorch implementation for "[Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_15)"

[2023-10-09] The code of SR-AQA model is coming soon.

[2023-08-03] Release the SR-TEE dataset.

## Introduction
Automated quality assessment (AQA) in transoesophageal echocardiography (TEE) contributes to accurate diagnosis and echocardiographers’ training, providing direct feedback for the development of dexterous skills. However, prior works only perform AQA on simulated TEE data due to the scarcity of real data, which lacks applicability in the real world. Considering the cost and limitations of collecting TEE data from real cases, exploiting the readily available simulated data for AQA in real-world TEE is desired. In this paper, we construct the first simulation-to-real TEE dataset, and propose a novel Simulation-to-Real network (SR-AQA) with unsupervised domain adaptation for this problem. It is based on uncertainty-aware feature stylization (UFS), incorporating style consistency learning (SCL) and task-specific learning (TL), to achieve high generalizability. Concretely, UFS estimates the uncertainty of feature statistics in the real domain and diversifies simulated images with style variants extracted from the real images, alleviating the domain gap. We enforce SCL and TL across different real-stylized variants to learn domain-invariant and task-specific representations. Experimental results demonstrate that our SR-AQA outperforms state-of-the-art methods with 3.02% and 4.37% performance gain in two AQA regression tasks, by using only 10% unlabelled real data.

## Content
### Architecture
<img src="https://github.com/wzjialang/SR-AQA/blob/main/figure/framework_simple.png" height="500"/>

Fig.1 Overall architecture of the proposed Simulation-to-Real Automated Quality Assessment network (SR-AQA).

### Datasets
The SR-TEE dataset published in our paper could be downloaded [here](https://doi.org/10.5522/04/23699736)

### Setup & Usage for the Code

1. Unzip the SR-TEE file and check the structure of data folders:
```
(root folder)
├── TEE
|  ├── real_cases_data_frames
|  |  ├── XXX.jgp
|  |  ├── ...
|  ├── simulated_data_frames
|  |  ├── XXX.jgp
|  |  ├── ...
|  ├── real_cases_data_frames.csv
|  ├── simulated_data_frames.csv
```

2. Check dependencies:
```
- Python 3.8+
- PyTorch 1.10+
- scikit-learn
- cudatoolkit
- cudnn
- OpenCV-Python
- tlib
```

Simple example of dependency installation:
```
conda creative -n sraqa python=3.8.8
conda activate sraqa
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/thuml/Transfer-Learning-Library.git
cd Transfer-Learning-Library/
python setup.py install
pip install -r requirements.txt
```


## Reference
Appreciate the work from the following repositories:
* [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)

## Cite
If this repository is useful for your research, please cite:
```
@InProceedings{10.1007/978-3-031-43996-4_15,
author="Xu, Jialang
and Jin, Yueming
and Martin, Bruce
and Smith, Andrew
and Wright, Susan
and Stoyanov, Danail
and Mazomenos, Evangelos B.",
editor="Greenspan, Hayit
and Madabhushi, Anant
and Mousavi, Parvin
and Salcudean, Septimiu
and Duncan, James
and Syeda-Mahmood, Tanveer
and Taylor, Russell",
title="Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="154--164",
isbn="978-3-031-43996-4"
}
```
