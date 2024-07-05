# Volumetric-Conditional-Score-based-Residual-Diffusion-Model-for-PET-MR-Denoising

This repository contains the source code associated with our paper titled "Volumetric-Conditional-Score-based-Residual-Diffusion-Model-for-PET-MR-Denoising" which has been accepted at MICCAI 2024.

## Requirements

Ensure all the necessary packages listed.

numpy matplotlib scikit-learn scikit-image click requests psutil tqdm imageio imageio-ffmpeg pyspng pillow nibabel
Pytorch Version: torch torchvision --index-url https://download.pytorch.org/whl/cu118

## Sub folders

Diffusion implementation : PatchDiffusion_dnPET

U-net implementation : PatchDiffusion_simpleunet

Transformer implemntation : PatchDiffusion_transformer

## Running the Training/Experiments

To conduct experiments, please build adn run docker image using the command below. Note that you should adjust the paths and hyperparameters according to your specific requirements:

1. move to the location of source code (where dockerfile is located).
2. Build docker image

```bash
sudo docker build -f ./dockerfile_train -t pet_train ./
```

3. Run docker image
```bash
sudo docker run --shm-size=8G --rm --gpus all -v /home/example/:/external/ pet_train
```
Note; here "/home/example/" is where source code and dockerfile are located in your GPU server.


4. To perform experimentsn, please build adn run docker image using the command below. 
```bash
sudo docker build -f ./dockerfile_test -t pet_test ./
```
```bash
sudo docker run --shm-size=8G --rm --gpus all -v /home/example/:/external/ pet_test
```



Pretrained model weights :
https://drive.google.com/drive/folders/1jbyC63eMJE51Pz-bRBl1D28D7dqgNjfT?usp=sharing
Please contact to author or leave the issue in github, if you have any question on model weights. 
