# sdxl-trainer
Most likely this is not for you.
If you want an easy to use & excellent stable diffusion trainer, try EveryDream2Trainer:
https://github.com/victorchall/EveryDream2trainer/

I am a novice programmer.
I spent 6 months learning python in my spare time, starting from zero programming skills, to be able to write my first function sdxl trainer.
I spent a few more months to create what is here.
Do not expect a professional programmer's quality, or you will be sorely disappointed.

I have tried my best to ensure that functions work as they should, tested all the features, checked the outputs, and everything seems good to go.  But there are going to be bugs.  By using this script, you're basically volunteering for bug finding duty.

Real-ESRGAN & HPSv2 code was referenced in this code.  Specific information regarding code used & license can be found in the relevant code sections.


## Welcome
This script was written by me, for me, for my specific goal: Using consumer grade gpus, fine-tune an sdxl model that is of equal quality of that trained by professional/enterprise grade gpus

This is a script meant to do a professional job, using consumer parts.  That's it.  It's not designed to be friendly, convenient, or popular.

With that said, apologies for the state of this readme. This has been a long project and I'm kinda burntout.  If there is significant interest in this repo, I'll take the time to write a better readme.  If you have questions, put in the discussions, not issues.  If something is broken, and you are using an environment similar to below, put it in issues.  Otherwise, put it discussions.


Training Environment:

- Ubuntu 22.04
- python 3.10.12
- 2 rtx3090s, 2 rtx4090s, 1 rtx2060 12GB
- 256GB ram (though 64 probably will suffice)


This script has not been tested in any other environment, and there are no plans to.


Now that it uses deepspeed, it probably doesn't work in windows.  If you know how to use accelerate FSDP 1 gpu w/ cpu offset, it is very easy to convert this script to FSDP, which may work with windows.

##IMPORTANT WARNING
Validation_image has a memory leak, most likely from HPSv2.  Therefore do not use it (set config: validation_image = null).
Validation_image can be run after traing completes, using saved checkpoints.  I'll upload instructions on how to do so later.


## Basic script info:
Uses accelerate with deepspeed zero stage 1, fine-tunes an sdxl model with a minimum of 1 9GB GPU (theoretical, tested with 12GB). Also, can train with multi-gpu

Further info: sdxl_trainer
- designed for large scale, large data set training
- it's assumed that you're willing to do what needs to be done to train an awesome model
- tested on Ubuntu 22.04 with rtx3090, rtx4090, & briefly testd with rtx2060 12GB
- at 1024 resolution, with rtx3090, 1.5-2.0imgs/sec per rtx3090

Features:
- accelerate deepspeed zero stage 1 with cp offset
- mixed precision fp16 (accelerate.autocast(fp16))
- gradient_checkpointing, gradient_accumulation, DDPMScheduler
- Optimizer: AdamW8bit
	- Adagrad8bit, Lion8bit: could be added with simple code change
        - Adafactor: should work with deepspeed
        	- FSDP initial tests showed Cuda OOM, required 1/2 batch size to not OOM
- save_state & load_state
- save pipeline as fp16
- sample image generation = 1st row is base_model + new sample_images appended below
- progress bar - it/sec, imgs/sec, loss
	- imgs/sec: true average over entire train, only measures during actual training batches, so looks slow in the beginning
- tensorboard logging: most items set to log per gradient update or per epoch
- each epoch's leftover training items are appended to next epoch
- use convert_diffusers_to_original_sdxl to convert saved diffusers pipeline to safetensors
- aspect ratio bucketing: multiple aspect ratio buckets per training resolution
- multi-resolution: set a training resolution range
	- image > max_resolution: downscale image to max resolution
	- image < min_resolution: skip image
	- training resolution range: min to max resolution
- upscale low res training images: avoids learning upscale artifacts via original size
	- currently uses Real_ESRGAN & GFPGAN.
		- if you know how to implement 4x_foolhardy_Remacri.pth for upscaling, please let me know
	- image >= min_resolution and image < upscale to resolution:
		- image is upscaled to upscale_to_resolution
	- training resolution range: upscale resolution to max resolution
- original_size: upscales images prior to training, avoids learning upscale artifacts via original size
	- see sdxl micro-conditioning
- standard assortment of learning rate schedulers
- gpu vram usage logging
	- logged usage may be incorrect, logs gpu 0 vram.  will be fixed in future update
- use --set_seed for deterministic training
	- if no set_seed, sample/validation will use seed 123.
        - For consistency in samples/validation between deterministic/random training runs: --set_seed 123
- transparent images merged with white background for caching
- conditional_dropout: default = 0.1, % of captions to replace with empty captions

Loss & Validation
	- default uses 10% of data set for validation_loss/validation_image

- loss:
	- loss: loss from training set
	- validation_loss: loss from validation set
- validation_image scores:
	- inception score: range: 1-infinity, low: 2.0 mid: 5.0 high: 8.0. Close to 1 = generated images are of high quality (high confidence in classifying) and diversity. 
	- LPIPS: range: 0-1, low: 0.05 mid: 0.3 high: 0.7.  0 = perfect image match, 1 = maximum dissimilarity between images
	- FID: range: 0-infinity, low: 5, mid: 25, high: 50. typically lower score = generated images & real images are closer in feature space, suggesting higher quality and better fidelity in generated images
	- KID: range: 0-infinity, low: 0.001 mid: 0.1 high: 1.0. Similar to FID, but less sensitive to outliers
	- HPSv2.1: higher is better, only useful when comparing generated images of same prompt


Suggested Training Parameters (though not tested), assuming a very large data set
- learning rate: 1e-5
- effective batch size: ~2000
- resolution: 1024


Important
- data set caching resolution range & training resolution range must be the same!
- site-packages/basicsr/data/degradations.py
	- change torchvision.transforms.functional_tensor to torchvision.transforms.functional

Known Bugs/Issues
- *Fixed*: It is suggested to use absolute paths.  - - cache_dir: only works with relative paths
- upscale/original_image Vs not upscale/original image training quality comparison test not conducted yet


Other
- data set caching script runs on 1 gpu.  Run multiple processes to use more gpus.
- dynamic batch size based on resolution with goal of maximizing batch size was tested with deepspeed zero stage 2
	- had insignificant impact on batch sizes of differing resolutions, so it was dropped
- torch compile lead to 5% decrease in initial training speed, long-term training speed probably same as without torch compile. so we don't use torch.compile()
- this script does not lora dreambooths


## Installation and usage
  1) git clone this repo
  2) create venv (create_venv.sh)
  3) install requirements (update_req.sh)
     - GFPGAN/BasicSR/RealESRGAN will complain about tb-nightly not installed.  Installing tb-nightly causes two versions of tensorboard to be installed, thus breaking tensorboard.  If you know the proper way of preventing tb-nightly from being installed, let me know in discussions.  Currenly GFPGAN/BasicSR/RealESRGAN installed without dependencies, then need dependencies installed.
  4) in venv/lib/python3.10/site-packages/basicsr/data/degradations.py
     - change torchvision.transforms.functional_tensor to torchvision.transforms.functional
     - basicsr uses deprecated call to torchvision
  5) download and put in base directory:
     - GFPGANv1.3.pth : https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
     - RealESRGAN_x4plus.pth : https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
     - HPS_v2.1_compressed.pt: https://huggingface.co/xswu/HPSv2/blob/main/HPS_v2.1_compressed.pt
  6) download bpe_simple_vocab_16e6.txt.gz: https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz
     - put in: venv/lib/python3.10/site-packages/hpsv2/src/open_clip/
  7) cache image-caption.txt pair database (sdxl_process_data_dir.sh)
     - choose gpu to use via export CUDA_VISIBLE_DEVICES=X in sdxl_process_data_dir.sh
  8) train (train.sh)
      - multi-gpu: choose gpus to use via changing gpu_ids: DS_01.yaml (set for gpus 0,1)
      - single-gpu: choose gpus to use via changing --gpu_ids X (set for gpus 2) in train_single_2.sh


Each python script has detailed information at the top of the script, same for each .sh file.
They have more info than this readme.
