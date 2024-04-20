#deepspeed
#2 gpus, zero stage 1 w/ cpu offset, adam8bit, 1024 resolution, batch size 14

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

#see sdxl_deepspeed_train.py for full list of arguments

#load_saved_state options:
#path/to/saved_state = path/to/saved_state (usually "output/project_name/epoch")
#"highest_epoch" = from output_dir, loads highest numbered folder
#"newest_folder" = from output_dir, loads folder with newest creation date


#uncomment to train offline
export TRANSFORMERS_OFFLINE=1

#increase nccl timeout
export NCCL_SOCKET_TIMEOUT=2000000

#consumer gpus can't use P2P or IB
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1

#num of threads per process for torch.distributed.run, default is one
#increase to see if performance improves
#optimal value: OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node. for training server/boxes not running other important processes
export OMP_NUM_THREADS=1

#potentially other useful things
#https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html?highlight=device
#export OMP_SCHEDULE=STATIC
#export OMP_PROC_BIND=CLOSE
#export GOMP_CPU_AFFINITY="N-M"


source venv/bin/activate
accelerate launch \
	--config_file DS_01.yaml \
	sdxl_deepspeed_train_55.py \
	--config_json config_4090_1e-5.json
