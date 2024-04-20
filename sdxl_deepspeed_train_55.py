#sdxl_FSDP_train
    #designed for large scale, large dataset training
    #it's assumed that you're willing to do what needs to be done to train an awesome model
    #tested on Ubuntu 22.04 with rtx3090s & rtx4090s

#Features:
    #accelerate FSDP FULL_SHARD
    #trains in fp32 with accelerate.autocast(fp16)
    #Defaults: tf32, gradient_checkpointing, gradient_accumulation, DDPMScheduler
    #Optimizer: AdamW8bit
        #Adagrad8bit, Lion8bit: could be added with simple code change
        #Adafactor: initial tests showed Cuda OOM, required 1/2 batch size to not OOM
    #save & load unet: save state is non-fuctional due to BNB incompatibility with FSDP, wait for BNB to issue fix
    #save pipeline as fp16
    #sample image generation = 1st row is base_model + new sample_images appended below
    #progress bar - it/sec, imgs/sec, loss
        #imgs/sec: true average over entire train, only measures during actual training batches, so looks slow in the beginning
    #tensorboard logging: most items set to log per gradient update or per epoch
    #each epoch's leftover training items are appended to next epoch
    #use convert_diffusers_to_original_sdxl to convert saved diffusers pipeline to safetensors
    #aspect ratio bucketing: multiple aspect ratio buckets per training resolution
    #multi-resolution: set a training resolution range
        #image >= max_resolution: downscale image to max resolution
		#image < min_resolution: skip image
        #training resolution range: min to max resolution
    #upscale low res training images: avoids learning upscale artifacts via original size
        #currently uses Real_ESRGAN & GFPGAN.
            #if you know how to implement 4x_foolhardy_Remacri.pth, let me know
        #image >= min_resolution and image < upscale to resolution
			#image is upscaled to upscale_to_resolution
		#training resolution range: upscale resolution to max resolution
    #original_size: upscales images prior to training, avoids learning upscale artifacts via orginal size
        #see sdxl micro-conditioning
    #standard assortment of learning rate schedulers
    #logged usage maybe incorrect, logs gpu 0 vram.  will be fixed in future update
    #use --set_seed for deterministic training
        #if no set_seed, sample/validation will use seed 123.
        #For consistency in samples/validation between deterministic/random training runs: --set_seed 123
    #transparent images merged with white background for caching
    #conditional_dropout: default = 0.1, % of captions to replace with empty captions
    #default uses 10% of dataset for validation_loss/validation_image

#loss:
    #loss: loss from training set
    #validation_loss: loss from validation set
#validation_image scores:
    #inception score: range: 1-infinity, low: 2.0 mid: 5.0 high: 8.0. Close to 1 = generated images are of high quality (high confidence in classifying) and diversity. 
        #lower = better
    #LPIPS: range: 0-1, low: 0.05 mid: 0.3 high: 0.7.  0 = perfect image match, 1 = maximum dissimilarity between images
        #lower = better
    #FID: range: 0-infinity, low: 5, mid: 25, high: 50. typically lower score = generated images & real images are closer in feature space, suggesting higher quality and better fidelity in generated images
        #lower = better
    #KID: range: 0-infinity, low: 0.001 mid: 0.1 high: 1.0. Similar to FID, but less sensitive to outliers
        #lower = better
    #HPSv2.1: higher is better, only useful when comparing generated images of same prompt
        #higher = better

#Suggested Training Parameters
    #learning rate: 1e-5
    #effective batch size: ~2000
    #resolution: 1024

#Important
    #current script uses linux ram drive to temporally to store unet/pipeline (~10GB) when transferring between processes
    #dataset caching resolution range & training resolution range must be the same!
    #site-packages/basicsr/data/degradations.py
        #change torchvision.transforms.functional_tensor to torchvision.transforms.functional

#Known Bugs/Issues
    #cache_dir: only works with relative paths
    #upscale/original_image Vs not upscale/original image test not conducted yet

#Other
    #dataset caching script runs on 1 gpu.  Run multiple processes to use more gpus.
    #dynamic batch size based on resolution with goal of maximizing batch size was tested with deepspeed zero stage 2
        #had non-significant impact on batch sizes of differing resolutions
    #torch compile lead to 5% decrease in initial training speed, long-term training speed probably same as without torch compile
    #this script does not lora dreambooths

#needed code modifications for accelerate/FSDP:
    #accelerate.print #to print on on main process
    #class TorchTracemalloc: &  with TorchTracemalloc() as tracemalloc: #tracks peak memory usage of the processes #i nfsdp_with_peak_mem_tracking.py
    #total_loss += loss.detach().float() #for tracking loss for logging
    #Accelerator.gather_for_metrics #gathers across processes, drops duplicates
    #load from most recent checkpoint: in checkpointing.py
    #dropout: neurons/layers: change instances of Dropout(p=0.0, inplace=False) to Dropout(p=0.1, inplace=False)
    #global minimum_learning_rate:
            #for param_group in optimizer.param_groups:
                #param_group['lr'] = max(param_group['lr'], minimum_learning_rate)
    #what is enforce_zero_terminal_snr?
    #min SNR Gamma
    #manually adjust loss scale before training begins
    #use HPSv2 to compare generated image and original, then use for DPO
    #DPO
    #use_ema
    #timestep_bias_strategy
    #theory: Adam uses more memory during 1st epoch
        #first epoch placeholder optimizer: torch.optim.SGD([])  # Empty optimizer
        #accelerate: unet, placeholder_optimizer, train_dataloader = accelerator.prepare(unet, placeholder_optimizer, train_dataloader)
        #training loop:
            #for epoch in range(1, num_epochs + 1):
                #if epoch > 1:
                    # Create the actual optimizer from the second epoch onward
                    #optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
                    # Replace the placeholder optimizer with the actual optimizer
                    #accelerator.replace_optimizer(placeholder_optimizer, optimizer)
                # Training loop continues with accelerated training
                #for batch in train_dataloader:

#learning rate 1e-5 w/ batch size ~2000, 200 epochs
    #potential learning rates 4e-7, for batch size 1, bsz 2000 1e-5
    #512: 1e-6 over 7000 steps with a batch size of 64


import argparse
from collections import defaultdict
import json
import gc
import logging
import os
from pathlib import Path
import random
import time
#import shutil #not currently used, but probably will be used later

from accelerate import Accelerator
from accelerate.utils import set_seed
import bitsandbytes as bnb
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from PIL import Image
import pynvml
import torch
import torch.nn.functional as F #for F.mse_loss: mean squared error (MSE)
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from sdxl_data_functions_23 import cache_image_caption_pair, cached_file_integrity_check, CachedImageDataset, BucketBatchSampler
from sdxl_validation_functions_24 import make_sample_images, calculate_validation_image_scores, calculate_validation_loss


def main():


    ##########
    #begin script initial configuration & setup
    ##########


    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="config.json", help='training configuration file')
    '''
    #dataset & training resolution
    parser.add_argument("--cached_dataset_dirs", nargs='+', type=str, help="path/to/cache : has cached_dataset.list(s), accepts multiple dirs")
    parser.add_argument("--cached_dataset_lists", nargs='+', type=str, help="path/to/cache_dataset.list, accepts multiple files")
    parser.add_argument("--max_resolution", type=int, default=1024, help="maximum image resolution to use for training")
    parser.add_argument("--min_resolution", type=int, default=512, help="minimum image resolution to use for training")
    parser.add_argument("--upscale_to_resolution", type=int, help="upscale image to resolution for caching, use original_size parameter")
    parser.add_argument("--upscale_use_GFPGAN", action='store_true', help="after upscale image, use GFPGAN to fix face (use for photos only)")
    parser.add_argument("--save_upscale_samples", action='store_true', help="after upscale image, save_upscale_samples")
    parser.add_argument("--verify_cached_dataset_hash_values", action='store_true', help="before training, verify integrity of cached dataset")
    #training parameters
    parser.add_argument("--conditional_dropout_percent", type=float, default=0.1, help="percent of captions to replace with empty captions.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=60, help="number of gradient_accumulation_steps")
    parser.add_argument("--learning_rate_scheduler", type=str, default="constant_with_warmup", help='Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--num_train_epochs", type=int, default=200, help="number of epochs to train")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="path/to/model_index.json or huggingface repo name")
    parser.add_argument("--set_seed", type=int, help="seed number (int) : fixed seed for deterministic training")
    parser.add_argument("--train_batch_size", type=int, default=10, help="batch size per gpu for training")
    parser.add_argument("--weight_dtype", type=str, default="torch.float16", help="torch.float16 or torch.float32")
    #learning rate & optimizer
    parser.add_argument("--lr_warm_up_epochs", type=float, default=0.02, help="percent of total steps to warm-up learning rate for")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="batch size per gpu for training")
    parser.add_argument("--polynomial_lr_end", type=float, default=1e-8,help="polynomial lr scheduler lr_end: ending learning rate for polynomial")
    parser.add_argument("--polynominal_power", type=float, default=1.0, help="polynomial lr scheduler power: polynomial power")
    #saving & loading
    parser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model every n epochs")
    parser.add_argument("--save_state", action='store_true', help="save training state when saving model")
    parser.add_argument("--project_name", type=str, default="ABC", help="name of the project")
    parser.add_argument("--output_dir", type=str, help="path/to/output_dir, where to store samples/saved_models")
    parser.add_argument("--load_saved_state", type=str, help="path/to/saved_state, for continuing training - check output/epoch/model")
    parser.add_argument("--start_save_model_epoch", type=int, default=0, help="epoch to begin saving models")
    #logging
    parser.add_argument("--log_dir", type=str, default="logs", help="logs_dir location")
    #parser.add_argument("--log_interval", type=int, default=1, help="gradient updates per log update") #script uses gradient updates or epochs to log
    #sample_images
    parser.add_argument("--save_samples", action='store_true', help="save sample images")
    parser.add_argument("--num_sample_images", type=int, default=8, help="number of sample images per save sample images")
    parser.add_argument("--save_samples_every_n_epochs", type=int, default=10, help="save samples every n epochs")
    parser.add_argument("--start_save_samples_epoch", type=int, default=0, help="epoch to begin saving samples (base model samples always saved)")
    #validation
    parser.add_argument("--validation_image", action='store_true', help="use validation_image: IS, LPIPS, FID, KID")
    parser.add_argument("--validation_image_percent", type=float, default=0.10, help="percent of len(dataset) to use for image validation")
    parser.add_argument("--validation_image_every_n_epochs", type=int, default=10, help="image validation every n epochs")
    parser.add_argument("--validation_loss", action='store_true', help="use validation_loss")
    parser.add_argument("--validation_loss_percent", type=float, default=0.10, help="percent of len(dataset) to use for loss validation")
    parser.add_argument("--validation_loss_every_n_epochs", type=int, default=1, help="validation_loss every n epochs")
    '''
    args = parser.parse_args()

    #load config
    config_json = args.config_json
    config_json = os.path.abspath(config_json)
    with open(config_json, "r") as f:
        metadata = json.load(f)
        metadata["config_json_train"] = config_json

    #dtype:
    if metadata["weight_dtype"] == "fp16":#torch.float16 for gpu, torch.float32 for cpu
        dtype = torch.float16
    elif metadata["weight_dtype"] == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float32

    #accelerate stuff
    accelerator = Accelerator(
        gradient_accumulation_steps=metadata["gradient_accumulation_steps"],
        mixed_precision=metadata["weight_dtype"], #"fp16", "bf16", or "fp8"
    )
    device = accelerator.device

    
    #deepspeed stuff
    deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
    #accelerate expects batch_size in dataloader, but we use BucketBatchSampler which handles batch_size
    #so we need to pass deepspeed batch_size this way
    deepspeed_config["train_micro_batch_size_per_gpu"] = metadata["train_batch_size"]
    #deepspeed needs to handle gradient_accumulation, so we pass it directly to deepspeed here
    #if also in accelerate/deepspeed config yaml, may cause problem, so remove from yaml
    #all other gradient_accumulation_steps passes are null, only used for calculations, only this value is used
    deepspeed_config["gradient_accumulation_steps"] = metadata["gradient_accumulation_steps"]
    
    
    #welcome message
    accelerator.print("\nsdxl_train: initializing")


    ##set some core parameters
    #seed
    
    if metadata["set_seed"] != None: #set a fixed seed for deterministic training
        set_seed(metadata["set_seed"])
        seed = metadata["set_seed"]
        random.seed(seed)
        generator = torch.Generator("cuda").manual_seed(seed) #set the seed
    else:
        generator = torch.Generator("cuda").manual_seed(123)

    #tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    
    #prevent cuDNN benchmarks during the first forward pass, which may increase VRAM usage during first forward pass
    torch.cuda.benchmark = False #
    


    #PIL - prevent DecompressionBombError for large sample image grid
    Image.MAX_IMAGE_PIXELS = None


    ##variables, dirs, and such

    #key hyperparameters used during training
    num_train_epochs = metadata["num_train_epochs"]  #epochs
    num_processes = accelerator.num_processes #num_gpus used, set via accelerate config or accelerate/deepspeed.yaml
    metadata['num_processes'] = accelerator.num_processes
    train_batch_size = metadata["train_batch_size"] #Batch size
    gradient_accumulation_steps = metadata["gradient_accumulation_steps"]
    effective_batch_size = num_processes * train_batch_size * gradient_accumulation_steps
    metadata['effective_batch_size'] = effective_batch_size
    num_workers = 0 #num_workers for dataloader, leave at 0, throws errors with other numbers.
    metadata['num_workers'] = num_workers
    conditional_dropout_percent = metadata["conditional_dropout_percent"]

    #learning rate
    learning_rate = metadata["learning_rate"] #learning rate
    learning_rate_scheduler = metadata["learning_rate_scheduler"]
    learning_rate_warm_up_epochs = metadata["learning_rate_warm_up_epochs"]
    polynominal_power = metadata["polynominal_power"]
    polynomial_lr_end = metadata["polynomial_lr_end"]

    #dataset stuff
    verify_cached_dataset_hash_values = metadata["verify_cached_dataset_hash_values"]
    cached_dataset_dirs = metadata["cached_dataset_dirs"]
    cached_dataset_lists = metadata["cached_dataset_lists"]
    #ensure cached dataset was processed with same resolution values, these are only used to repair cached dataset
    max_resolution = metadata["max_resolution"] #image max_resolution
    min_resolution = metadata["min_resolution"] #image min_resolution
    upscale_to_resolution = metadata["upscale_to_resolution"] #images will be upscaled to this resolution
    save_upscale_samples = metadata["save_upscale_samples"] #saves copy of org & upscaled images
    upscale_use_GFPGAN = metadata["upscale_use_GFPGAN"]

    #model, vae, saving, & loading
    pretrained_model_name_or_path = metadata["pretrained_model_name_or_path"]
    pretrained_vae_model_name_or_path = metadata["pretrained_vae_model_name_or_path"]
    save_model_every_n_epochs = metadata["save_model_every_n_epochs"]
    start_save_model_epoch = metadata["start_save_model_epoch"]
    save_state = metadata["save_state"] #whether to save state when saving models
    load_saved_state = metadata["load_saved_state"] #path to saved state, for resuming training
    resume_epoch = None

    #dirs & names
    training_script_filename = os.path.basename(os.path.abspath(__file__)) #this file's name
    metadata["training_script_filename"] = training_script_filename
    project_name = metadata["project_name"] #training project name
    output_dir_prefix = metadata["output_dir_prefix"]
    train_name = f"{project_name}"
    metadata['train_name'] = train_name
    output_dir = os.path.join(output_dir_prefix, train_name)
    os.makedirs(output_dir, exist_ok=True)
    metadata['output_dir'] = output_dir

    log_dir_prefix = metadata["log_dir_prefix"]
    log_dir = metadata.get("log_dir", None)
    if log_dir is None:
        log_dir = os.path.join(log_dir_prefix, train_name)
        log_dir = os.path.abspath(log_dir)
        metadata["log_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    #log_interval = args.log_interval #script uses gradient updates or epochs to log

    #samples & validation
    dataset_initial_length = 0 #used to compute percent of dataset, value added after dataset verified
    #samples
    save_samples = metadata["save_samples"]
    save_samples_every_n_epochs = metadata["save_samples_every_n_epochs"]
    start_save_samples_epoch = metadata["start_save_samples_epoch"]
    sample_image_prompts = [] #empty list to start with
    sample_image_prompts_txt =  metadata["sample_image_prompts_txt"]
    save_sample_captions_txt = os.path.join(output_dir, "sample_prompts.txt") #stores copy of captions used for sample generation in output folder
    metadata["save_sample_captions_txt"] = save_sample_captions_txt
    sample_count = 0 #track samples per page (default 10)
    sample_page = 0
    #validation_image
    validation_image = metadata["validation_image"]
    validation_image_percent = metadata["validation_image_percent"]
    validation_image_jsons = []
    validation_image_jsons_txt = "validation_jsons.txt"
    validation_image_every_n_epochs = metadata["validation_image_every_n_epochs"]
    save_validation_image_jsons_txt = os.path.join(output_dir, "validation_jsons.txt")
    metadata["save_validation_image_jsons_txt"] = save_validation_image_jsons_txt
    #validation_loss
    validation_loss = metadata["validation_loss"]
    validation_loss_percent = metadata["validation_loss_percent"]
    validation_loss_every_n_epochs = metadata["validation_loss_every_n_epochs"]
    validation_loss_jsons_txt = os.path.join(output_dir, "validation_loss_jsons.txt")
    metadata["validation_loss_jsons_txt"] = validation_loss_jsons_txt
    #save_piplines list
    pipelines_txt = os.path.join(output_dir, "pipelines.txt")
    metadata["pipelines_txt"] = pipelines_txt

    #save metadata
    save_json_path = os.path.join(output_dir, os.path.basename(config_json))
    metadata["config_json_saved_copy"] = save_json_path
    with open(save_json_path, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
    


    ##logging

    #basic logging
    logging.basicConfig(
        filename="error_log.txt",  #specify the log file name
        level=logging.ERROR,  #set the logging level to ERROR
        format="%(asctime)s - %(levelname)s - %(message)s",  #format for log messages
    )
    
    #tensorboard
    writer = SummaryWriter(log_dir)

    #GPU VRAM logging #current logs gpu 0, will be fixed later
    if accelerator.is_main_process:
        pynvml.nvmlInit()
        def get_gpu_memory_usage(device_id):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return meminfo.used
        device_id = accelerator.local_process_index


    ##########
    #end script initial configuration & setup
    #begin dataset processing/verifying, train/val split, sample prompts list generations, and dataloaders
    ##########


    ##prepare cached_datasets

    #have dataset sanity check  #need second check for it train items == 0
    if accelerator.is_main_process:
        if cached_dataset_dirs == None and cached_dataset_lists == None:
            print("Error: no dataset provided:\n  use either --cached_dataset_dir or --cached_dataset_file to supply dataset")

        print("checking cached datasets")
        print("...")

    #ensure cached_dataset_lists is list, 
    if cached_dataset_lists == None or len(cached_dataset_lists) == 0:
        cached_dataset_lists = [] 
    else:
        #print items in cached_dataset_lists
        for item in cached_dataset_lists:
            if os.path.exists(item):
                if item[-5:] == ".list":
                    if accelerator.is_main_process:
                        print(f"found: {item}")
            else:
                if accelerator.is_main_process:
                    print(f"Error: {item} not found")

    #Scan combined cached_dataset_dirs & cached_dataset_files
    if cached_dataset_dirs != None or len(cached_dataset_dirs) != 0:
        for dir in cached_dataset_dirs:
            dir_contents = os.listdir(dir)
            for item in dir_contents:
                if item[-5:] == ".list":
                    item_path = os.path.join(dir, item)
                    cached_dataset_lists.append(item_path)
                    if accelerator.is_main_process:
                        print(f"found: {item_path}")

    #read cached_dataset_lists
    if accelerator.is_main_process:
        print("\nprocessing cached_dataset_lists:")
    cached_json_list = []
    for item in cached_dataset_lists:
        if accelerator.is_main_process:
            print(f"processing {item}:")
        with open(item, "r") as f:
            cached_json_list_temp = f.readlines() #each line to an item in list
            cached_json_list_temp  = [line.strip() for line in cached_json_list_temp ]  #remove newlines
            cached_json_list_temp  = list(set(cached_json_list_temp )) #remove duplicates
            cached_json_list_temp  = [item for item in cached_json_list_temp  if item] #remove empty items
            
            #append cached_json_list_temp to cached_json_list
            cached_json_list = cached_json_list + cached_json_list_temp
            print("\n  --complete")
    print(f"processed: {len(cached_json_list)} cached image-caption pairs found")


    ##dataset integrity check
    if verify_cached_dataset_hash_values == True:

        #open and read json_files
        if accelerator.is_main_process:
            print("\nbeginning verify cached files integrity")
        failed_hash_check = []
        count = 0
        for json_file in cached_json_list:
            count += 1
            #check hash_values
            result = cached_file_integrity_check(json_file)
            if accelerator.is_main_process:
                print(f"\r{[count]}: {json_file}: {result}", end="") #just looking busy
            #failed hash check
            if result != "pass":
                if accelerator.is_main_process:
                    print(f"{json_file} : {result}")
                failed_hash_check.append(json_file) #add to bad list
                cached_json_list.remove(json_file) #remove from good list
        #print initial pass check hash result
        if accelerator.is_main_process:
            if len(failed_hash_check) == 0:
                if accelerator.is_main_process:
                    print("\n --success : all files passed verification")
            else:
                if accelerator.is_main_process:
                    print(f"\rOh, No. {len(failed_hash_check)} files failed verification.")
                    print(" --attempting re-caching failed files")


        #process failed hash files --> try re-caching
        #create failed_image_files & failed_caption_files lists
        for json_file in failed_hash_check:
            #open and read json_file metadata, append to failed captions list
            with open(json_file, "r") as f: 
                metadata = json.load(f)
            basename = metadata["basename"]
            cache_dir = metadata["cache_dir"]
            data_dir = metadata["data_dir"]
            image_file = metadata["image_file"]
            caption_file = metadata["caption_file"]
            #to tuple list
            image_caption_files_tuple_list = []
            failed_pair = (image_file, caption_file)
            image_caption_files_tuple_list.append(failed_pair)
            #try re-caching image-caption.txt pair
            recached_json_list = cache_image_caption_pair(
                    image_caption_files_tuple_list,
                    pretrained_model_name_or_path,
                    pretrained_vae_model_name_or_path,
                    cache_dir,
                    data_dir,
                    basename,
                    device,
                    max_resolution,
                    min_resolution,
                    upscale_to_resolution,
                    upscale_use_GFPGAN,
                    save_upscale_samples
                )
            #verify re-cached hash values
            for json_file_recached in recached_json_list:
                cached_file_integrity_check(json_file_recached)
                #pass
                if result == True:
                    cached_json_list.append(json_file_recached)
                    failed_hash_check.remove(json_file)
                    if accelerator.is_main_process:
                        print(f"{json_file_recached} re-cached successfully.  It's nice to be back.")
                #fail
                else:
                    error_message = f"{json_file} double failed hash verification.  We tried, it's your responsibility now."
                    if accelerator.is_main_process:
                        print(error_message)
                    logging.error(error_message)

        #completed dataset integrity check
        if accelerator.is_main_process:				
            print("completed dataset integrity check")
            print(f"--{len(cached_json_list)} passed hash check")
            print(f"--{len(failed_hash_check)} failed hash check")
            if len(failed_hash_check) > 0:
                print("    -see error_log.txt for details")
    
    dataset_initial_length = len(cached_json_list)


    ##create sample_prompt_list after dataset completely finalized
    
    #read sample_prompts.txt
    accelerator.print("\collecting sample_prompts")

    #if load_saved_state, use saved samples caption list
    if load_saved_state != None:
        sample_image_prompts_txt = save_sample_captions_txt

    #open sample caption txt file
    accelerator.print("sample_prompts.txt: exists")
    with open(sample_image_prompts_txt, "r") as f:
        prompts_txt = f.readlines() #each line to an item in list
        prompts_txt = [line.strip() for line in prompts_txt]  # Remove newlines
        prompts_txt = list(set(prompts_txt)) #remove duplicates
        prompts_txt = [item for item in prompts_txt if item] #remove empty items
        sample_image_prompts = sample_image_prompts + prompts_txt
    sample_image_prompts.sort()

    #save copy of sample captions to output
    accelerator.print(f"  --len_sample_image_prompts: {len(sample_image_prompts)}")
    if accelerator.is_main_process: #save with main process
        with open(save_sample_captions_txt, "w") as file: 
            for item in sample_image_prompts:
                file.write(item + "\n")


    ##create loss_validation_list
    accelerator.wait_for_everyone()
    if validation_loss:

        #if load_saved_state load validation_loss_jsons_txt
        if load_saved_state != None:
            accelerator.print("\nvalidation loss resumed training:")
            accelerator.print("  --loading validation_loss_jsons_txt")
            with open(validation_loss_jsons_txt, "r") as f:
                validation_loss_jsons = f.readlines() #each line to an item in list
                validation_loss_jsons = [line.strip() for line in validation_loss_jsons]  # Remove newlines
                validation_loss_jsons = [item for item in validation_loss_jsons if item] #remove empty items
            #remove validation_loss_jsons from cached_json_list
            cached_json_list = [file for file in cached_json_list if file not in validation_loss_jsons]

            accelerator.print(f"  --finished")
            accelerator.print(f"validation_loss_jsons: {len(validation_loss_jsons)} files")

        #if new_training
        else:
            accelerator.print("\npreparing validation_loss list")
            #shuffle cached_json_list to ensure randomness
            random.shuffle(cached_json_list)
            
            #organize json files by closest_bucket
            bucket_files = defaultdict(list)
            validation_loss_list_of_lists = []  #loss validation list of lists

            #read json files metadata
            for file_path in cached_json_list:
                try:
                    with open(file_path, 'r') as file:
                        metadata = json.load(file)
                except Exception as e:
                    metadata = None
                    accelerator.print(f"create validation_loss list of lists: Error reading {file_path}: {e}")
                    
                #loss_validation is created to exactly fit buckets & batch_size
                if metadata:
                    closest_bucket = tuple(metadata["closest_bucket"])  # Convert list to tuple for dict key
                    bucket_files[closest_bucket].append(file_path)
                    #using train_batch_size * num_processes = closest_bucket list that exactly fills all gpus with 1 batch
                    #this ensures no leftovers, or other problems
                    if len(bucket_files[closest_bucket]) == train_batch_size * num_processes:
                        validation_loss_list_of_lists.append(bucket_files.pop(closest_bucket))
                    
                    #check if target percent of database reached
                    if sum(len(files) for files in validation_loss_list_of_lists) > dataset_initial_length * validation_loss_percent:
                        break
            
            #remove validation_loss file from cached_json_list
            validation_loss_jsons = [item for sublist in validation_loss_list_of_lists for item in sublist]
            cached_json_list = [file for file in cached_json_list if file not in validation_loss_jsons]
            with open(validation_loss_jsons_txt, "w") as file: 
                    for item in validation_loss_jsons:
                        file.write(f"{item}\n")

            del bucket_files

            accelerator.print(f" --finished")
            accelerator.print(f"validation_loss_jsons: {len(validation_loss_jsons)} files")


    
    #no validation loss
    else:
        validation_loss_jsons = []


    ##create validation_image_json_list after datasets completely finalized
    if validation_image:

        #if can use same validation set for both validation_loss & validation_image
        if validation_image and validation_loss:
            if validation_image_percent == validation_loss_percent:
                validation_image_jsons = validation_loss_jsons
                accelerator.print("\nusing validation_loss jsons for validation_image")

        else:
        #check validation_image_prompts.txt
            accelerator.print("\ncollecting validation_image_jsons")
            if os.path.exists(validation_image_jsons_txt):
                accelerator.print("validation_image_jsons.txt: exists")
                with open(validation_image_jsons_txt, "r") as f:
                    validation_jsons_txt = f.readlines() #each line to an item in list
                    validation_jsons_txt = [line.strip() for line in validation_jsons_txt]  # Remove newlines
                    validation_jsons_txt = list(set(validation_jsons_txt)) #remove duplicates
                    validation_jsons_txt = [item for item in validation_jsons_txt if item] #remove empty items
                    validation_image_jsons = validation_image_jsons + validation_jsons_txt
            
            #calculate number jsons needed
            num_validation_jsons = int(dataset_initial_length * validation_image_percent)
            num_validation_jsons = (num_validation_jsons // 3) * 3
            num_needed_validation_jsons = num_validation_jsons - len(validation_image_jsons)
            
            #collect random_validation_image_jsons
            accelerator.print("\ncollecting random_validation_image_jsons for validation_image_jsons:\n...")
            if num_needed_validation_jsons > 0: #add needed jsons
                for i in range(num_needed_validation_jsons):
                    #first use flattened_validation_loss_jsons
                    if len(validation_loss_jsons) > 0:
                        json_file = random.choice(validation_loss_jsons).pop()
                        validation_image_jsons.append(json_file)
                    #then use cached_json_list
                    else:
                        json_file = random.choice(cached_json_list)
                        validation_image_jsons.append(json_file)

            #if too many jsons, remove excess jsons
            if num_needed_validation_jsons < 0:
                while num_needed_validation_jsons < 0:
                    json_file = random.choice(validation_image_jsons)
                    validation_image_jsons.remove(json_file)
                    num_needed_validation_jsons += 1
        
        #finish
        accelerator.print(f"  --len_validation_image_jsons: {len(validation_image_jsons)}")

        #save copy to output dir
        with open(save_validation_image_jsons_txt, "w") as file:
            sorted_list = validation_image_jsons
            sorted_list.sort()
            for item in sorted_list:
                file.write(item + "\n")
            del sorted_list


    ##setup train_dataset
    accelerator.print("\ntrain dataset setup:")
    train_dataset = CachedImageDataset(cached_json_list, conditional_dropout_percent) 
    accelerator.print(f"len_train_dataset: {len(train_dataset)}")

    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(train_dataset, batch_size=train_batch_size, drop_last=True)

    #initialize the DataLoader with the bucket batch sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        num_workers=num_workers
    )


    ##setup validation_dataset
    accelerator.print("\nvalidation dataset setup:")
    validation_conditional_dropout = 0.0
    validation_loss_dataset = CachedImageDataset(validation_loss_jsons, validation_conditional_dropout) 
    accelerator.print(f"len_validation_loss_dataset: {len(validation_loss_dataset)}")

    #create bucket batch sampler
    bucket_batch_sampler = BucketBatchSampler(validation_loss_dataset, batch_size=train_batch_size, drop_last=True)

    #initialize the DataLoader with the bucket batch sampler
    validation_loss_dataloader = torch.utils.data.DataLoader(
        validation_loss_dataset,
        batch_sampler=bucket_batch_sampler, #use bucket_batch_sampler instead of shuffle
        num_workers=num_workers
    )


    #variables based on train  dataset & dataloader
    num_train_images = len(train_dataset)
    num_train_epochs = num_train_epochs
    num_steps_per_epoch = num_train_images // (num_processes * train_batch_size)
    num_update_steps_per_epoch = num_steps_per_epoch // gradient_accumulation_steps if num_steps_per_epoch >= gradient_accumulation_steps else 1
    total_num_train_steps = num_train_epochs * num_steps_per_epoch
    total_num_update_steps = num_update_steps_per_epoch * num_train_epochs
    num_warmup_update_steps = int(learning_rate_warm_up_epochs * num_update_steps_per_epoch)

    #variables based on validation dataset & dataloader
    num_validation_images = len(validation_loss_dataset)
    val_batch_size = train_batch_size
    num_val_epochs = num_train_epochs
    num_val_steps_per_epoch = num_validation_images // (num_processes * val_batch_size)
    num_val_update_steps_per_epoch = num_val_steps_per_epoch // gradient_accumulation_steps
    total_num_val_steps = num_train_epochs * num_val_steps_per_epoch
    total_num_val_update_steps = num_val_update_steps_per_epoch * num_val_epochs


    ##########
    #end dataset processing/verifying, train/val split, sample prompts list generations, and dataloaders
    #begin loading model, optimizer, lr_scheduler, etc & accelerate.prepare
    ##########


    ##load and configure training components

    #new training session from base model
    if load_saved_state is None:
        accelerator.print(f"\nloading base model: {pretrained_model_name_or_path})")

        noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )
        unet.train() #sets unet for training
        unet.enable_gradient_checkpointing()
        #unet = accelerator.prepare(unet)

        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)

        #Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        if learning_rate_scheduler == "polynomial":
                lr_scheduler = get_scheduler(
                learning_rate_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_update_steps * num_processes,#warmup steps appear to get divided among processes
                num_training_steps=total_num_update_steps,
                lr_end=polynomial_lr_end,
                power=polynominal_power,
            )
        else:
            lr_scheduler = get_scheduler(
                learning_rate_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_update_steps * num_processes,#warmup steps appear to get divided among processes
                num_training_steps=total_num_update_steps,
            )
        
        #move everything to accelerate
        #optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler) 
        unet, optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler) 

        accelerator.print(" --loaded")

    #restore a training session, load components from saved pipeline & load_saved_state
    
    else:
        accelerator.print(f"\nloading saved_state: {load_saved_state}")
        
        #load saved models
        noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )

        '''
        #temp loading unet solution, until BNB works with save_state #FSDP
        unet_stat_dict_path = os.path.join(load_saved_state, "unet_state_dict.pth")
        unet_state_dict = torch.load(unet_stat_dict_path)
        '''
        unet.train() #sets unet for training
        unet.enable_gradient_checkpointing()
        

        #for FSDP
        #unet = accelerator.prepare(unet)

        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)
        
        if learning_rate_scheduler == "polynomial":
                lr_scheduler = get_scheduler(
                learning_rate_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_update_steps * num_processes,#warmup steps appear to get divided among processes
                num_training_steps=total_num_update_steps,
                lr_end=polynomial_lr_end,
                power=polynominal_power,
            )
        else:
            lr_scheduler = get_scheduler(
                learning_rate_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_update_steps * num_processes,#warmup steps appear to get divided among processes
                num_training_steps=total_num_update_steps,
            )

        #unet.load_checkpoint(load_saved_state, "pytorch_model")
        #accelerator.print("loaded unet")
        
        #move everything to accelerate
        #optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler)
        unet, optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, validation_loss_dataloader, lr_scheduler) 

        accelerator.print(" --loaded")
        
        
        #load state does not work with 8bit optimizers, or deepspeed w/ cpu offset
        #load_state #deepspeed

        #load highest epoch
        if load_saved_state == "highest_epoch":
            dirs = os.listdir(output_dir)
            print(f"output_dir: {output_dir}")
            print(f"dirs: {dirs}")
            epoch_dirs = [d for d in dirs if d.isdigit() and os.path.isdir(os.path.join(output_dir, d))]
            print(f"epoch_dirs: {epoch_dirs}")
            highest_epoch = max(epoch_dirs, key=int)
            load_saved_state = os.path.join(output_dir, highest_epoch)
            print(f"load_saved_state: {load_saved_state}")
            del dirs, epoch_dirs, highest_epoch
        
        #load highest epoch
        if load_saved_state == "newest_folder":
            dirs = os.listdir(output_dir)
            print(f"output_dir: {output_dir}")
            print(f"dirs: {dirs}")
            full_dirs = [os.path.join(output_dir, d) for d in dirs if os.path.isdir(os.path.join(output_dir, d))]
            most_recent_dir = max(full_dirs, key=os.path.getmtime)
            load_saved_state = os.path.join(most_recent_dir)
            print(f"load_saved_state: {load_saved_state}")
            del dirs, full_dirs, most_recent_dir
        
        #load saved_state
        accelerator.load_state(load_saved_state)


        #load additional variables from checkpoint
        checkpoint_path = os.path.join(load_saved_state, "checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        resume_global_step = checkpoint["global_step"]
        resume_global_gradient_update_step = checkpoint["global_gradient_update_step"]
        resume_epoch = checkpoint["epoch"]

        #add lr_scheduler later
        if accelerator.is_main_process:
            print(" --loaded")



    ##########
    #end loading model, optimizer, lr_scheduler, etc & accelerate.prepare
    #begin pre-training loop
    ##########


    ##initiate training loop variables

    #resumable variables
    if load_saved_state != None: #if resume
        global_step = resume_global_step
        global_gradient_update_step = resume_global_gradient_update_step
        epoch = resume_epoch 
    else:
        global_step = 0
        global_gradient_update_step = 0
        epoch = 0

    #variables for tracking
    img_batch_times = [] #step for imgs/sec
    img_sec_per_update_list = []
    gradient_update_loss = 0.0 #tracking loss between gradient updates
    between_gradient_updates_step = 0 #tracking steps between gradient updates
    gradient_update_loss_list = [] #track gradient update loss per epoch, for epoch loss
    epoch_loss = 0.0 #tracking loss between epochs
    #hack: create blank epoch_loss.tmp for tracking loss per epoch, avoids divide by zero error
        #this can probably be fixed now that we're using FSDP
    if accelerator.is_main_process:
        with open("epoch_loss.tmp", "w") as f:
            pass

    #print
    if accelerator.is_main_process:
        print("\n\nTraining session information")
        print("----------------------------")
        print()
        print("#saving & loading")
        print(f"start_save_model_epoch: {start_save_model_epoch}")
        print(f"save_model_every_n_epochs: {save_model_every_n_epochs}")
        print(f"save_state: {save_state}")
        print(f"load_saved_state: {load_saved_state}")
        print()
        print("#dataset & batch size")
        print(f" num_train_images: {num_train_images}")
        print(f" num_processes: {num_processes}")
        print(f" train_batch_size: {train_batch_size}")
        print(f" gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f" effective_batch_size: {effective_batch_size}")
        print()
        print("#epochs and steps")
        print(f" num_epochs: {num_train_epochs}")
        print(f" num_steps_per_epoch: {num_steps_per_epoch}")
        print(f" num_update_steps_per_epoch: {num_update_steps_per_epoch}")
        print(f" total_num_train_steps: {total_num_train_steps}")
        print(f" total_num_update_steps: {total_num_update_steps}")
        print()
        print("#optimizer & learning rate")
        print(f" optimizer: bnb.optim.AdamW8bit")
        print(f" learning_rate: {learning_rate}")
        print(f" polynomial_lr_end: {polynomial_lr_end}")
        print(f" polynominal_power: {polynominal_power}")
        print(f" learning_rate_scheduler: {learning_rate_scheduler}")
        print(f" learning_rate_warm_up_epochs: {learning_rate_warm_up_epochs}")
        print(f" num_warmup_update_steps: {num_warmup_update_steps}")
        print()
        print("#samples & validation")
        print(f" save_samples: {save_samples}")
        print(f" len_sample_image_prompts: {len(sample_image_prompts)}")
        print(f" validation_loss: {validation_loss}")
        print(f" validation_loss_every_n_epochs: {validation_loss_every_n_epochs}")
        print(f" len_validation_loss_jsons: {len(validation_loss_jsons)}")
        print(f" validation_image: {validation_image}")
        print(f" validation_image_every_n_epochs: {validation_image_every_n_epochs}")
        print(f" len_validation_image_jsons: {len(validation_image_jsons)}")
        '''
        #not seeing the loss scaling adjustments that deepspeed had
        #but that that might be because accelerate doesn't show them
        #verify exactly what is going on with loss scaling
        print("#first ~10 gradient updates notice") #maybe can be fixed now that FSDP is used?
        print(" the first ~10 gradient updates are a bust")
        print(" they're spent adjusting loss scaling, because of overflows")
        print(" so far, no method is successful to pre-adjust loss scaling")
        print(" not much I can do, so let it do it's thing\n")
        '''
        print()
        print()

    #pre-train clean-up
    gc.collect()
    torch.cuda.empty_cache()


    ##########
    #end pre-training loop
    #begin training loop stuff
    ##########


    ##training loop code

    #time tracking
    if accelerator.is_main_process:
        start_time = time.time()
        
    #epochs
    for epoch in range(epoch, num_train_epochs + 1):
        start_epoch_step = global_step
        
        ## samples, validations, & saving state & model
        if accelerator.is_main_process:
            if epoch == 0:
                with open(pipelines_txt, "a") as f:
                    f.write(f"{pretrained_model_name_or_path}\n")
                    print(f"   --base model added to pipelines_txt")
        
        #check if just resumed from loaded_saved_state
        if resume_epoch != epoch:

            #sample image generating
            accelerator.wait_for_everyone()

            if save_samples:
                
                if epoch >= start_save_samples_epoch or epoch == 0:
                    if epoch % save_samples_every_n_epochs == 0:

                        #re-build pipeline with trained unet

                        accelerator.print("preparing sample pipeline... ")
                        accelerator.wait_for_everyone()

                        '''
                        FSDP
                        #get_state_dict on all processes
                        unet_state_dict = accelerator.get_state_dict(unet)
                        '''

                        #then switch to main process
                        if accelerator.is_main_process:

                            #load vae   
                            vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)
                            vae = vae.to(dtype)
                            #load pipeline
                            pipeline = StableDiffusionXLPipeline.from_pretrained(
                                    pretrained_model_name_or_path,
                                    vae=vae,
                                    unet=accelerator.unwrap_model(unet),
                                    torch_dtype=dtype, #this saves as fp16, later change to fp32
                                )
                            del vae

                            '''
                            FSDP
                            #pull pipeline.unet, load unet_state_dict, put back in pipeline
                            trained_unet = pipeline.unet
                            trained_unet.load_state_dict(unet_state_dict)
                            trained_unet.eval()
                            pipeline.unet = trained_unet
                            del unet_state_dict, trained_unet
                            '''

                            gc.collect()
                            torch.cuda.empty_cache()

                            print("begin image generation")

                            #sample_page & count
                            sample_page = int(epoch // (save_samples_every_n_epochs * 10))
                            sample_count = int((epoch / save_samples_every_n_epochs) - (sample_page * 10))

                            #go make some samples
                            pipeline.to(device).to(dtype)
                            
                            make_sample_images(
                                sample_count,
                                sample_page,
                                pipeline,
                                generator,
                                accelerator,
                                sample_image_prompts,
                                epoch,
                                output_dir,
                                train_name
                            )

                            del pipeline
                            gc.collect()
                            torch.cuda.empty_cache()


            #calculate_validation_image_scores
            accelerator.wait_for_everyone()
            if validation_image:
                #if load_saved_state == None: #add a check to validation_image vs loaded epoch
                #if epoch == 0:
                if epoch % validation_image_every_n_epochs == 0:
                    #for save pipeline to ram disk (linux '/dev/shm/')
                    pipeline_temp = "/dev/shm/pipeline"
                    if accelerator.is_main_process:
                        os.makedirs(pipeline_temp, exist_ok=True)

                    #re-build pipeline with trained unet
                    accelerator.print("preparing validation_image pipeline... ")
                    accelerator.wait_for_everyone()

                    #get_state_dict on all processes
                    unet_state_dict = accelerator.get_state_dict(unet)

                    #then switch to main process
                    if accelerator.is_main_process:

                        #load vae   
                        vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)
                        vae = vae.to(dtype)
                        #load pipeline
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                                pretrained_model_name_or_path,
                                vae=vae,
                                #unet=test_unet,
                                torch_dtype=dtype, #this saves as fp16, later change to fp32
                            )

                        del vae

                        #pull pipeline.unet, load state, put back in pipeline
                        trained_unet = pipeline.unet
                        trained_unet.load_state_dict(unet_state_dict)
                        trained_unet.eval()
                        pipeline.unet = trained_unet

                        del unet_state_dict, trained_unet

                        #save pipeline to ram disk (linux '/dev/shm/')
                        pipeline.save_pretrained(pipeline_temp)
                        del pipeline
                        gc.collect()
                        torch.cuda.empty_cache()

                    #load pipeline on all processes
                    accelerator.wait_for_everyone()
                    
                    #load pipeline
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        pretrained_model_name_or_path,
                        torch_dtype=dtype, #this saves as fp16, later change to fp32
                    )

                    print("begin image generation")
                    pipeline.to(device).to(dtype)

                    calculate_validation_image_scores(pipeline, generator, device, accelerator, validation_image_jsons, epoch, writer, output_dir)

                    del pipeline
                    gc.collect()
                    torch.cuda.empty_cache()


            #calculate validation_loss
            accelerator.wait_for_everyone()
            if validation_loss: #validation_loss runs every epoch
                if epoch % validation_loss_every_n_epochs == 0:
                    calculate_validation_loss(accelerator, unet, validation_loss_dataloader, val_batch_size, num_val_steps_per_epoch, num_processes, noise_scheduler, writer, epoch)
            
            accelerator.wait_for_everyone()


            ##save_state & save_pipeline
            #save_state does not work with BNB, wait for BNB to fix #FSDP
                #instead saving/loading unet

            #check save model epoch count
            accelerator.wait_for_everyone()
            with torch.no_grad(): #just to be safe
                if epoch >= start_save_model_epoch and epoch != 0:
                    if epoch % save_model_every_n_epochs == 0:
                        #save_path
                        save_path = os.path.join(output_dir, str(epoch))
                        if accelerator.is_main_process:
                            print("\npreparing to save...")
                            print(f"  save_path: {save_path}")
                            os.makedirs(save_path, exist_ok=True)

                        #save_model
                        #re-build pipeline with trained modules and save model.
                        if accelerator.is_main_process:
                            print(" saving pipeline... ")
                        accelerator.wait_for_everyone()

                        '''
                        #FSDP
                        #get_state_dict on all processes
                        unet_state_dict = accelerator.get_state_dict(unet)
                        '''

                        #save_state
                        if save_state == True:
                            
                            #deepspeed
                            accelerator.print(" saving state...")
                            #save_state does not work with BNB, wait for BNB to fix
                            accelerator.save_state(save_path) #save state with accelerator

                            '''
                            #FSDP
                            #save unet here, until save_state is fixed
                            if accelerator.is_main_process:
                                print(" saving state...")
                                unet_state_dict_path = state_path = os.path.join(output_dir, str(epoch), "unet_state_dict.pth")
                                torch.save(unet_state_dict, unet_state_dict_path)
                            '''

                            #save dictionary with other training values
                            state_path = os.path.join(output_dir, str(epoch), "checkpoint.pth")
                            accelerator.save({ 
                                "global_step": global_step,
                                "global_gradient_update_step": global_gradient_update_step,
                                "epoch": epoch,
                            }, state_path)
                            if accelerator.is_main_process:
                                print("   --state saved")

                        #after get_state_dict, switch to main process for saving pipeline
                        if accelerator.is_main_process:

                            #load vae   
                            vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path)
                            vae = vae.to(dtype)
                            #load pipeline
                            pipeline = StableDiffusionXLPipeline.from_pretrained(
                                    pretrained_model_name_or_path,
                                    vae=vae,
                                    unet=accelerator.unwrap_model(unet),
                                    torch_dtype=dtype, #this saves as fp16, later change to fp32
                                )
                            del vae

                            '''
                            #FSDP
                            #pull pipeline.unet, load state, put back in pipeline
                            trained_unet = pipeline.unet
                            trained_unet.load_state_dict(unet_state_dict)
                            pipeline.unet = trained_unet
                            del unet_state_dict, trained_unet
                            '''
                            
                            #save pipeline
                            pipeline.save_pretrained(save_path) #, variant="fp16", safe_serialization=True
                            if accelerator.is_main_process:
                                with open(pipelines_txt, "a") as f:
                                    f.write(f"{os.path.abspath(save_path)}\n")
                                    print(f"   --pipeline saved")
                            del pipeline
                            

                    #finished
                    accelerator.wait_for_everyone()
                    gc.collect()
                    torch.cuda.empty_cache()


        ##pre-forward_pass stuff

        #epoch 1 begins
        epoch += 1

        ##check if is final epoch
        if epoch > num_train_epochs:
            break

        #setup
        accelerator.wait_for_everyone()

        accelerator.print(f"\n\nEpoch # {epoch}/{num_train_epochs}")
        accelerator.print(f"global_gradient_update_step: {global_gradient_update_step}/{total_num_update_steps}")


        if accelerator.is_main_process:
            #track epoch time
            img_batch_times = []
            img_sec_per_update_list = []
            epoch_start_time = time.time()
            #create a new progress bar for each epoch
            progress_bar = tqdm(
                range(num_steps_per_epoch),
                initial=0,
                desc=f"Current Epoch Steps",
                #disable=not accelerator.is_local_main_process, #printout timing seems off
            )


        ##forward_pass & backward_pass

        #batches
        for step, batch in enumerate(train_dataloader):
            img_sec_start_time = time.time() #track img/sec
            with accelerator.accumulate(unet): #this tracks our gradient_accumulation_steps

                #sample noise to add to latents
                noise = torch.randn_like(batch["model_input"]) #create noise in the shape of latent tensor
                bsz = batch["model_input"].shape[0] #bsz = batch size

                #sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=batch["model_input"].device)
                timesteps = timesteps.long() #.long() = convert to 64-bit signed integer data type: -2^63 to 2^63-1

                #forward pass
                #add noise to the latents according to the noise magnitude at each timestep (noise manitude@timestep is set by scheduler)
                noisy_model_input = noise_scheduler.add_noise(batch["model_input"], noise, timesteps)

                #time_ids come in as a list of six tensors, with num_values per tensor = batch_size, then it gets flattened here
                add_time_id = torch.cat(batch["add_time_id"], dim=0).view(-1)
                #add_time_id.shape: torch.Size([66] #6 * batch_size 11

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_id}
                unet_added_conditions.update({"text_embeds": batch["pooled_prompt_embed"]})
                with accelerator.autocast():
                    model_pred = unet(noisy_model_input, timesteps, batch["prompt_embed"], added_cond_kwargs=unet_added_conditions).sample #from sd1.5 train script
                del noisy_model_input

                #we're using "epsilon", so target = noise
                target = noise
                del noise

                #calculate loss - original version
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                del model_pred, target

                #loss for logging
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean() #shouldn't need loss.detach, but double-check anyway
                gradient_update_loss += avg_loss.item()
                between_gradient_updates_step += 1

                #backward pass
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    #accelerator.clip_grad_norm_(unet.parameters(), 1.0) #clip_grad_norm done in config yaml
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                ##post batch logging

                global_step += 1

                #log GPU VRAM
                if accelerator.is_main_process:
                    if global_step % 25 == 0:
                        vram_usage = round(get_gpu_memory_usage(device_id) / (1024 ** 3), 3)
                        writer.add_scalar('Performance/GPU VRAM Usage', vram_usage, global_step + 1)

                gc.collect()
                torch.cuda.empty_cache()

                #calculate imgs/second
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    img_sec_end_time = time.time()
                    img_sec_batch_time = img_sec_end_time - img_sec_start_time
                    img_batch_times.append(img_sec_batch_time)
                    img_sec_per_update_list.append(img_sec_batch_time)
                    while len(img_batch_times) > 10:
                        del img_batch_times[0]
                    imgs = len(img_batch_times) * num_processes * train_batch_size
                    imgs_sec = (imgs / sum(img_batch_times))
                    progress_bar.set_postfix({
                        "imgs/s": f"{imgs_sec:.2f}", # .Xf, where X = num decimal places
                    })
                    progress_bar.update() #update progress bar after each step

            #post batch check for gradient updates
            accelerator.wait_for_everyone()
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    
                    imgs_update = len(img_sec_per_update_list) * num_processes * train_batch_size
                    imgs_sec_update = (imgs_update / sum(img_sec_per_update_list))
                    #log stuff every gradient update
                    #if step % log_interval == 0: #script uses gradient updates or epochs to log
                    writer.add_scalar("Performance/imgs per sec, per update_step", imgs_sec_update, global_gradient_update_step) #log imgs/sec
                    img_sec_per_update_list = []

                    #if no gradient update, don't log
                    if accelerator.optimizer_step_was_skipped:
                        print("optimizer_step_was_skipped: loss values discarded")

                    else:
                        global_gradient_update_step += 1
                        #loss per gradient update
                        gradient_update_loss = gradient_update_loss / between_gradient_updates_step
                        writer.add_scalar("Loss/gradient_update, per update_step)", gradient_update_loss, global_gradient_update_step)
                        #for loss per epoch                        
                        gradient_update_loss_list.append(gradient_update_loss)

                        #learning rate log
                        current_lr = lr_scheduler.get_last_lr()[0]
                        writer.add_scalar("Performance/Learning Rate, per update_step", current_lr, global_gradient_update_step)

                    #reset gradient_update_loss
                    gradient_update_loss = 0.0
                    between_gradient_updates_step = 0


            #batch column
        #epoch column

        ##post epoch logging

        #calculate epoch time
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_time_hours = epoch_time / 3600
            writer.add_scalar("Performance/epoch_time(hours), per epoch", epoch_time_hours, epoch) #log imgs/sec

        #epoch loss
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            num_update_steps = len(gradient_update_loss_list)
            if len(gradient_update_loss_list) != 0:
                    epoch_loss = sum(gradient_update_loss_list) / num_update_steps
                    if epoch_loss != 0:
                        print(f"\nepoch_loss: {epoch_loss}")
                        writer.add_scalar("Loss/epoch", epoch_loss, epoch)
                    else:
                        print(f"\nno epoch_loss: all gradient updates skipped due to loss scaling")
                        print(f"\n  --not a problem if epoch 1 & num_updates_per_epoch is < 10")
        gradient_update_loss_list = [] #reset
        
        #track steps & update_steps per epoch
        end_epoch_step = global_step
        epoch_steps = end_epoch_step - start_epoch_step
        if accelerator.is_main_process:
            writer.add_scalar("Performance/steps per epoch", epoch_steps, epoch)
            writer.add_scalar("Performance/update_steps per epoch", num_update_steps, epoch)




    ##calculate total training time
    if accelerator.is_main_process:
        end_time = time.time()
        total_time = end_time - start_time
        hours = total_time / 3600
        print(f"training complete: total training time {hours:.2f} hours")
        print("\n\n")
    accelerator.end_training()


##this thing
if __name__ == "__main__":
    main()