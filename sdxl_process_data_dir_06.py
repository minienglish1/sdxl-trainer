#process_data_dir.py
	#scan dir for image-caption.txt pairs
	#caches found image-caption.pairs 
	#mirrors data_dir directory structure in cache_dir 
	#per image-caption.txt pair creates needed cache files & json file:
	#per data_dir, creates data_dir.txt: contains json_files list

#upscale to resolution info:
	#no upscale to resolution:
		#image >= max_resolution: downscale image to max resolution
		#image < min_resolution: delete image
		#training resolution range: upscale resolution to max resolution
	#upscale to resolution used:
		#image >= min_resolution and image < upscale to resolution
			#image is upscaled to upscale_to_resolution
		#training resolution range: upscale resolution to max resolution


import argparse
import logging
import os

from accelerate import Accelerator
import torch

from sdxl_data_functions_23 import data_dir_search, cache_image_caption_pair


#welcome message
print("\nprocess_data_dir: initializing")


#initiate accelerator
accelerator = Accelerator(
	mixed_precision="fp16",
)

device = torch.device("cuda")


##arguments
parser = argparse.ArgumentParser()
parser.add_argument("--basename", type=str, default="data", help="The name of the dataset folder: ie '/mnt/storage/comics/' basename would be 'comics'")
parser.add_argument("--data_dir", type=str, default="data", help="'path/to/data_dir' --image-caption.txt directory location")
parser.add_argument("--cache_dir", type=str, default="cache", help="'path/to/cache_dir'data_dir --location to store cached images/captions")
parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="'huggingface model, or path to local model")
parser.add_argument("--max_resolution", type=int, default=1536, help="maximum image resolution")
parser.add_argument("--min_resolution", type=int, default=512, help="maximum image resolution")
parser.add_argument("--upscale_to_resolution", type=int, help="upscale image to resolution for caching, use original_size parameter")
parser.add_argument("--upscale_use_GFPGAN", action='store_true', help="after upscale image, use GFPGAN to fix face (use for photos only)")
parser.add_argument("--save_upscale_samples", action='store_true', help="after upscale image, save_upscale_samples")
args = parser.parse_args()


#tf32
torch.backends.cuda.matmul.allow_tf32 = True


##variables

#dirs
basename = args.basename
data_dir = args.data_dir
cache_dir = args.cache_dir
os.makedirs(cache_dir, exist_ok=True)
#models
pretrained_model_name_or_path = args.pretrained_model_name_or_path
pretrained_vae_model_name_or_path = "madebyollin/sdxl-vae-fp16-fix"
#resolution & upscale
max_resolution = args.max_resolution
min_resolution = args.min_resolution
upscale_to_resolution = args.upscale_to_resolution
save_upscale_samples = args.save_upscale_samples
upscale_use_GFPGAN = args.upscale_use_GFPGAN


##error logging
logging.basicConfig(
	filename="error_log.txt",  # Specify the log file name
	level=logging.ERROR,  # Set the logging level to ERROR
	format="%(asctime)s - %(levelname)s - %(message)s"  # Format for log messages
)


#search for image-caption.txt pairs the directory and subdirectories
#input: data_dir 
#return: image_caption_pair_file tuple list
image_caption_files_tuple_list = data_dir_search(data_dir)


##preprocess images/caption, cache latents/hidden_encoder_states
#input:images & captions
    #hashes & caches files & saves json list to disk
#returns list of json filepaths
json_file_paths_list = cache_image_caption_pair(
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
