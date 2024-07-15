This repository contains PyTorch implementation for __Refined Diffusion Posterior Sampling For Seismic Denoising And Interpolation__.

Run the following command the install the guided-diffusion package:
```
pip install -e .
```


## Download Checkpoints and Data

Download dataset F3 and Kerry3D dataset:
- [F3 dataset](https://wiki.seg.org/wiki/F3_Netherlands).
- [Kerry-3D dataset](https://wiki.seg.org/wiki/Kerry-3D).

Download pretrained DDPMs on F3 and Kerry3D dataset from [this page](https://drive.google.com/drive/folders/1PZ2cMMUPTPTxr82JGYg3sFbfneVZtv22?usp=sharing). 

## Sampling 

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule cosine --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

MODEL_PATH="--model_path <pretrained_model_path> --mask_model_path model_050.pth"

SAMPLE_FLAGS="--mask_ratio 0.1 --noise_scale 0.15 --gradient_scale 0.5"
python3 sample.py $MODEL_FLAGS $MODEL_PATH $SAMPLE_FLAGS --cuda_devices '0' --batch_size <batch_size> --mode test_valid --method rdps --data_dir <data_directory> --dataset 'F3,Kerry3D'
```

This implementation is based on / inspired by:
- [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion) (the Guided Diffusion repo),
- [https://github.com/UCSB-NLP-Chang/CoPaint](https://github.com/UCSB-NLP-Chang/CoPaint) (the CoPaint repo),
- [https://github.com/Fayeben/GenerativeDiffusionPrior](https://github.com/Fayeben/GenerativeDiffusionPrior) (the GDP repo), and
- [https://github.com/DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling) (the DPS repo).
