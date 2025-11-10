<!-- # Our code is currently being organized and will be open-sourced soon -->

# HFGDet

Official PyTorch implementation of "HFGDet: High Frequency Guided UAV Small Object Detection"

## Coming Soon

We will release of pretrained model weights.

## Getting started

我们在YOLO和D-FINE上分别实现了HFGDet并在无人机数据集VisDrone2019和UAVDT，以及通用数据集MS-COCO上进行实验

### Environments

HFGDet is developed based on ``torch==2.4.1`` and ``CUDA Version==12.1``

### Prepare VisDrone2019 Dataset for YOLO

Download and extract [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) dataset in the following directory structure:

```
├── VisDrone2019
    ├── VisDrone2019-DET-train
        ├── images
            ├── 0000002_00005_d_0000014.jpg
            ├── ...
        ├── labels
            ├── 0000001_02999_d_0000014.txt
            ├── ...
    ├── VisDrone2019-DET-val
        ├── images
            ├── 0000001_02999_d_0000005.jpg
            ├── ...
        ├── labels
            ├── 0000001_02999_d_0000005.txt
            ├── ...

```

## Train in YOLO

```shell
cd HFG-YOLO

python train.py
```

## Train in D-FINE

Note: 在D-FINE上训练需要coco格式的VisDrone2019

Train with 2 GPUs:

```shell
cd HFG-D-FINE

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py -c configs/dfine/dfine_hgnetv2_n_visdrone.yml --use-amp --seed=3407 -t weight/dfine/dfine_n_coco.pth &> n_visdrone.log 2>&1 &

# resume
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train.py -c configs/dfine/visdrone/dfine_hgnetv2_n_visdrone.yml --use-amp --seed=3407 -r output/dfine_hgnetv2_n_visdrone/last.pth
```

## Acknowledgements

We thank but not limited to following repositories for providing assistance for our research:

[//]: # (- [TIMM]&#40;https://github.com/rwightman/pytorch-image-models&#41;)

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [D-FINE](https://github.com/Peterande/D-FINE)
