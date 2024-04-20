#export CUDA_VISIBLE_DEVICES=0 #don't use this, use the single_gpu.yaml to select gpu
#export TRANSFORMERS_OFFLINE=1 #enable to run huggingface offline, using cache

#arguments:
#	--cache_dir				save cached latents to dir
#	--data_dir				image-caption.txt pair dir
#	--pretrained_model_name_or_path		sdxl model to train
#	--max_resolution			max training image resolution	
#	--min_resolution			mix training image resolution
#	--upscale_to_resolution			resolution to upscale images to 
#	--upscale_use_GFPGAN			use GFPGAN - for photos
#	--save_upscale_samples			saves copy of upscaled images

#	--data_dir "/mnt/storage/datasets/line art/" \
#	--basename "line art" \
#	in data_dir, line art is the folder representing the dataset's basename
#	think of it as the basename folder for the dataset

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=2
source venv/bin/activate
python sdxl_process_data_dir_06.py \
	--data_dir "/mnt/storage/datasets/line art/" \
	--basename "line art" \
	--cache_dir /mnt/storage/cache/ \
	--max_resolution 1024 \
	--min_resolution 256 \
	--upscale_to_resolution 1024 \
	--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0
