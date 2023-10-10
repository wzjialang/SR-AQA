# Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography
This repository provides the official PyTorch implementation of the following paper:
> [**Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography**](https://doi.org/10.1007/978-3-031-43996-4_15)<br>
> [Jialang Xu](https://www.researchgate.net/profile/Jialang-Xu), Yueming Jin, Bruce Martin, Andrew Smith, Susan Wright, Danail Stoyanov, Evangelos Mazomenos<br>

[2023-10-09] Release the code of SR-AQA model.

[2023-08-03] Release the SR-TEE dataset.

## Introduction
Automated quality assessment (AQA) in transoesophageal echocardiography (TEE) contributes to accurate diagnosis and echocardiographers’ training, providing direct feedback for the development of dexterous skills. However, prior works only perform AQA on simulated TEE data due to the scarcity of real data, which lacks applicability in the real world. Considering the cost and limitations of collecting TEE data from real cases, exploiting the readily available simulated data for AQA in real-world TEE is desired. In this paper, we construct the first simulation-to-real TEE dataset, and propose a novel Simulation-to-Real network (SR-AQA) with unsupervised domain adaptation for this problem. It is based on uncertainty-aware feature stylization (UFS), incorporating style consistency learning (SCL) and task-specific learning (TL), to achieve high generalizability. Concretely, UFS estimates the uncertainty of feature statistics in the real domain and diversifies simulated images with style variants extracted from the real images, alleviating the domain gap. We enforce SCL and TL across different real-stylized variants to learn domain-invariant and task-specific representations. Experimental results demonstrate that our SR-AQA outperforms state-of-the-art methods with 3.02% and 4.37% performance gain in two AQA regression tasks, by using only 10% unlabelled real data.

## Content
### Architecture
<img src="https://github.com/wzjialang/SR-AQA/blob/main/figure/framework_simple.png" height="500"/>

Fig.1 Overall architecture of the proposed Simulation-to-Real Automated Quality Assessment network (SR-AQA).

### Dataset
The SR-TEE dataset published in our paper could be downloaded [here](https://doi.org/10.5522/04/23699736).

### Setup & Usage for the Code
1. Unzip the dowloaded SR-TEE dataset and check the structure of data folders:
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
- cudatoolkit
- cudnn
- tlib
```

- Simple example of dependency installation:
```
conda create -n sraqa python=3.8
conda activate sraqa
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/thuml/Transfer-Learning-Library.git
cd Transfer-Learning-Library/
python setup.py install
pip install -r requirements.txt
```

3. Train & test SR-AQA model:
- For CP task
```
python main_ours.py /path/to/your/dataset \
        -d TEE -s S -t R --task_type cp_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/SR-AQA/TEE_cp --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10
```
- For GI task
```
python main_ours.py /path/to/your/dataset \
        -d TEE -s S -t R --task_type gi_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/SR-AQA/TEE_gi --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10
```

*'--fs_layer 1 1 1 0 0'* means replacing the $1- 3^{rd}$ batch normalization layers of ResNet-50 with the UFS.<br>
*'--lambda_scl'* means the lambda for SCL loss, if *'--lambda_scl'* > 0, then using SCL loss.<br>
*'--lambda_tl'* means the lambda for TL loss, if *'--lambda_tl'* > 0, then using TL loss.<br>
*'--t_data_ratio X'* means using X-tenths of unlabeled real data for training.

## Acknowledge
Appreciate the work from the following repositories:
* [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
* [suhyeonlee/WildNet](https://github.com/suhyeonlee/WildNet)

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
