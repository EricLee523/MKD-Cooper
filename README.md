# MKD-Cooper
The code will be made available after the acceptance of the paper (MKD-Cooper: Cooperative 3D Object Detection for Autonomous Driving via Multi-teacher Knowledge Distillation)

## Installation

I recommend you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install MKD-Cooper. The installation is totally the same as OpenCOOD.


## Training 训练
### Single GPU Training

```
python opencood/tools/train.py -y YAML_FILE [--model_dir MODEL_FOLDER]
```

For example:

(1) Single Teacher

```
python opencood/tools/train_w_kd.py -y opencood/hypes_yaml/opv2v/lidar_only_with_noise/MKD_Cooper_Single_Teacher.yaml
```

(2) Multi-Teacher

```
python opencood/tools/train_w_kd_mt.py -y opencood/hypes_yaml/opv2v/lidar_only_with_noise/MKD_Cooper_Multi_Teacher.yaml
```

### Multiple GPU Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp.py -y YAML_FILE [--model_dir MODEL_DIR]
```

For example:

(1) Single Teacher

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_w_kd.py -y opencood/hypes_yaml/opv2v/lidar_only_with_noise/MKD_Cooper_Single_Teacher.yaml
```

(2) Multi-Teacher

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_w_kd_mt.py -y opencood/hypes_yaml/opv2v/lidar_only_with_noise/MKD_Cooper_Multi_Teacher.yaml
```

### Testing

```
python opencood/tools/inference.py --model_dir MODEL_DIR
```


## Acknowlege

This project is impossible without the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), and [CoAlign](https://github.com/yifanlu0227/CoAlign)!

