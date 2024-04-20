#FSDP
#2 gpus adam8bit 1024 resolution batch size 10

#additonal commands
#	--cached_dataset_lists None \ #use .list instead if cached dir, not tested
#	--output_dir \ 	#if not used, script will create output dir name base on training parameters
#	--cached_dataset_dirs /mnt/storage/cache/ \
#	--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \ #change model to your base model
#	--save_state \ #save unet when saving pipeline
#	--load_saved_state \ #dir where unet is saved
#	--weight_dtype torch.float16 \ #script is 100% designed for fp16, don't change this unless you know what you're doing
#	--polynomial_lr_end 1e-8 \ #lr_end for polynominal lr scheduler
#	--polynomial_power 1 \ #polynominal lr scheduler power
#	--upscale_use_GFPGAN \	#use GFPGAN when upscaling, useful for photos, not useful for non-photos
#	--save_upscale_samples \ #saves org & upscaled images, useful to check if upscaling meets your quality standard
#	--save_samples \ #save sample_images during training
#	--validation_image #use validation_image (IS/FID/KID/LPIPS/HPSv2) scoring
#	--verify_cached_dataset_hash_values #verify cached dataset integrity before training
#	--validation_loss #use validation_loss
#	--load_saved_state \ #load saved unet
#	--validation_loss \
#	--validation_image \
#	--save_samples \
#	--save_state \

#see sdxl_deepspeed_train.py for full list of arguments

#consumer gpus can't use P2P or IB
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

#uncomment to train offline
#export TRANSFORMERS_OFFLINE=1

source venv/bin/activate

accelerate launch --gpu_ids 2 \
	--config_file DS_single_2.yaml \
	sdxl_DS-Zs2_train_55_sample_len.py \
	--config_json config_single_cotton_doll_1e8.json
