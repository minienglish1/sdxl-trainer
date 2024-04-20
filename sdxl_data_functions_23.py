#sdxl_data_functions.py
#provides various dataset related functions for sdxl_train system:
    #data_dir_search - searches a dataset dir, creates image_caption_pair_file tuple list
    #cache_image_caption_pair - caches lists of image-caption.txt pairs, creates .list file for use as cached dataset 
    #cached_file_integrity_check - verifies cached dataset integrity
    #CachedImageDataset - loads cached dataset to be sent to dataloader
    #BucketBatchSampler - creates batch by aspect ratio buckets, batch_size sent here instead dataloader
        #if drop_last=True, leftover_items are appended to next epoch
#place these 2 files in base directory
    #GFPGANv1.3.pth : https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
    #RealESRGAN_x4plus.pth : https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
#if save upscale samples, samples saved in: upscale_dir = "upscale_samples"
#notes:
    #need to re-check/verify random selection of aspect bucket each batch


import gc
import hashlib
import json
import logging
import os
from pathlib import Path #not used, but probably will be used
import random

import joblib
from PIL import Image #pillow
from torch.utils.data import Dataset, Sampler


#group indices by their corresponding aspect ratio buckets before sampling batches.
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.leftover_items = []  # Tracks leftover items
        
        self.bucket_indices = self._bucket_indices_by_aspect_ratio()
        self.prepared_batches = self._prepare_batches()

    def _bucket_indices_by_aspect_ratio(self):
        buckets = {}
        for idx in range(len(self.dataset)):
            closest_bucket = self.dataset.get_closest_bucket(idx)
            closest_bucket_key = tuple(closest_bucket)
            if closest_bucket_key not in buckets:
                buckets[closest_bucket_key] = []
            buckets[closest_bucket_key].append(idx)
        return buckets

    def _prepare_batches(self):
        prepared_batches = []
        all_buckets = self.bucket_indices.items()

        leftover_buckets = {}
        for leftover_idx in self.leftover_items:
            closest_bucket_key = tuple(self.dataset.get_closest_bucket(leftover_idx))
            leftover_buckets.setdefault(closest_bucket_key, []).append(leftover_idx)

        #clear leftover items
        self.leftover_items = []

        for bucket_key, bucket in all_buckets:
            # Merge leftovers with current bucket items
            bucket = leftover_buckets.pop(bucket_key, []) + bucket
            random.shuffle(bucket)

            while len(bucket) >= self.batch_size:
                prepared_batches.append(bucket[:self.batch_size])
                bucket = bucket[self.batch_size:]

            if bucket:
                if not self.drop_last:
                    prepared_batches.append(bucket)
                elif bucket:
                    self.leftover_items.extend(bucket)  # Carry over leftovers to next epoch

        #for buckets not used this epoch, append to next epoch's items
        for leftover_bucket in leftover_buckets.values():
            self.leftover_items.extend(leftover_bucket)

        return prepared_batches

    def __iter__(self):
        self.bucket_indices = self._bucket_indices_by_aspect_ratio()  # Refresh bucket indices
        self.prepared_batches = self._prepare_batches()  # Prepare new batches
        random.shuffle(self.prepared_batches)  # Shuffle batches
        for batch in self.prepared_batches:
            yield batch

    def __len__(self):
        return len(self.prepared_batches)


##input: json_file_list -> output: metadata
#looks like leftover code from leftover_idx, check, then delete
class CachedImageDataset(Dataset):
    def __init__(self, json_file_paths_list, conditional_dropout_percent=0.1): 
        self.json_file_paths = json_file_paths_list
        #for conditional_dropout
        self.conditional_dropout_percent = conditional_dropout_percent
        self.empty_prompt_embed = joblib.load("empty.prompt_embed.pkl")  # Tuple of (empty_prompt_embed, empty_pooled_prompt_embed)
        self.empty_pooled_prompt_embed = joblib.load("empty.pooled_prompt_embed.pkl")

    #returns dataset length
    def __len__(self):
        return len(self.json_file_paths)

    def get_closest_bucket(self, index):
        # Retrieve the closest_bucket for a given index
        json_file_path = self.json_file_paths[index]
        with open(json_file_path, "r") as f:
            metadata = json.load(f)
        return metadata["closest_bucket"]


    #returns dataset item, using index
    def __getitem__(self, index):
        json_file_path = self.json_file_paths[index] 
        with open(json_file_path, "r") as f: 
            metadata = json.load(f)

        #cached files
        model_input = joblib.load(metadata["model_input_file"])
        prompt_embed = joblib.load(metadata["prompt_embed_file"])
        pooled_prompt_embed = joblib.load(metadata["pooled_prompt_embed_file"])

        #conditional_dropout
        if random.random() < self.conditional_dropout_percent:
            prompt_embed = self.empty_prompt_embed
            pooled_prompt_embed = self.empty_pooled_prompt_embed

        #TO-DO, check items for error before returning
            #can't load, hash mismatch, etc

        return {
            #cached files
            "model_input": model_input,
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            #information from metadata
            "add_time_id": metadata["add_time_id"],
            "category_key": metadata["category_key"],
            "closest_bucket": metadata["category_key"],
            "original_image_size": metadata["original_image_size"],
            "cropped_image_size": metadata["cropped_image_size"],
        }


#search for image-caption.txt pairs the directory and subdirectories
#input: data_dir output: image_files & caption_files (lists)
def data_dir_search(data_dir):

    #convert to absolute data_dir
    if os.path.isabs(data_dir):
        abs_data_dir = data_dir
    else:
        abs_data_dir = os.path.abspath(data_dir)
    
    data_dir = abs_data_dir

    
    #begin
    image_ext = ['.png', '.jpg', '.jpeg', ".bmp", ".webp", ".tif"]
    image_caption_files_tuple_list = []
    count = 0

    print("\nbegin data_search")
    print("  note:")
    print("    --If image file or caption.txt has error: image-caption.txt pair will be skipped")
    print("    --If image file OK & no caption.txt: caption = \"\" [empty_caption]")
    print("    --Check error_log.txt for details")
    print(f"data_dir: {data_dir}")
    print("...")
        
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_ext: #check if supported image file
                
                #process image
                image_file = os.path.join(root, file) #construct full image file path
                #test Image.open image file                
                try:
                    check_image = Image.open(image_file)
                except Exception as e:
                    error_message = f"Error: {e}, for {image_file}"
                    print(error_message)
                    logging.error(error_message)
                    continue
                
                #process caption
                caption_file = os.path.splitext(image_file)[0] + ".txt" #check for matching txt file
                #test open.read caption.txt
                if os.path.exists(caption_file):
                    try:
                        caption_file = os.path.splitext(image_file)[0] + ".txt" #check for matching txt file
                        with open(caption_file, "r") as f:
                            caption_string = f.read().strip() #test read file
                    except Exception as e:
                        error_message = f"Error reading {caption_file}: {e}"
                        print(error_message)
                        logging.error(error_message)
                        continue
                else:
                    #dummy caption_file & caption_string
                    caption_string = ""
                    with open(caption_file, "w") as f:
                        pass
                    
                    #error message
                    error_message = f"No caption.txt for {image_file}"
                    print(f"created blank: {caption_file}, caption_string = \"\" [empty]")
                    print(error_message)
                    logging.error(error_message)
                
                #append tuple
                pair = (image_file, caption_file)
                image_caption_files_tuple_list.append(pair)

                #finish
                count += 1
                #running print of processing current file
                print(f"\r[{count}]: {file[:30]}... : {caption_string[:30]}...", end='')

    #check if num_images = num_captions
    print("\ndata_search complete")
    print(f"{len(image_caption_files_tuple_list)} image-caption pairs found")
    gc.collect()
    return image_caption_files_tuple_list


##aspect categories for bucketing images
#current width/height ratios: 2:1 to 1:2. example for 1024: [704, 1408] to [1408, 704]
aspect_categories = {
    256: [[256, 256], [192, 320], [320, 192], [192, 256], [256, 192]],
    320: [[320, 320], [256, 384], [384, 256], [320, 256], [256, 320], [384, 192], [192, 384]],
    384: [[384, 384], [320, 448], [448, 320], [256, 512], [512, 256], [384, 320], [320, 384], [256, 448], [448, 256]],
    448: [[448, 448], [384, 512], [512, 384], [320, 576], [576, 320], [448, 384], [384, 448], [320, 512], [512, 320]],
    512: [[512, 512], [448, 576], [576, 448], [640, 384], [384, 640], [512, 448], [448, 512], [384, 576], [576, 384], [640, 320], [320, 640]],
    576: [[576, 576], [512, 640], [640, 512], [448, 704], [704, 448], [576, 512], [512, 576], [384, 768], [768, 384], [640, 448], [448, 640], [384, 704], [704, 384]],
    640: [[640, 640], [704, 576], [576, 704], [896, 448], [448, 896], [512, 768], [768, 512], [832, 448], [448, 832], [576, 640], [640, 576], [704, 512], [512, 704], [448, 768], [768, 448]],
    704: [[704, 704], [960, 512], [768, 640], [640, 768], [512, 960], [832, 576], [576, 832], [896, 512], [512, 896], [704, 640], [640, 704], [768, 576], [576, 768], [832, 512], [512, 832]],
    768: [[768, 768], [1024, 576], [576, 1024], [704, 832], [832, 704], [896, 640], [640, 896], [960, 576], [576, 960], [768, 704], [704, 768], [832, 640], [640, 832], [1024, 512], [512, 1024], [896, 576], [576, 896]],
    832: [[832, 832], [768, 896], [896, 768], [960, 704], [704, 960], [1152, 576], [576, 1152], [1024, 640], [640, 1024], [768, 832], [832, 768], [896, 704], [704, 896], [1088, 576], [576, 1088], [960, 640], [640, 960]],
    896: [[896, 896], [960, 832], [832, 960], [768, 1024], [1024, 768], [640, 1216], [1216, 640], [704, 1088], [1088, 704], [896, 832], [832, 896], [768, 960], [640, 1152], [1152, 640], [960, 768], [704, 1024], [1024, 704], [640, 1088], [1088, 640]],
    960: [[960, 960], [1024, 896], [896, 1024], [832, 1088], [1088, 832], [704, 1280], [1280, 704], [768, 1152], [1152, 768], [960, 896], [896, 960], [704, 1216], [1216, 704], [832, 1024], [1024, 832], [768, 1088], [1088, 768], [640, 1280], [1280, 640], [704, 1152], [1152, 704]],
    1024: [[1024, 1024], [1088, 960], [960, 1088], [768, 1344], [1344, 768], [1152, 896], [896, 1152], [832, 1216], [1216, 832], [1408, 704], [704, 1408], [768, 1280], [1280, 768], [1024, 960], [960, 1024], [896, 1088], [1088, 896], [832, 1152], [1152, 832], [704, 1344], [1344, 704], [768, 1216], [1216, 768]],
    1088: [[1088, 1088], [1536, 768], [768, 1536], [1152, 1024], [1024, 1152], [832, 1408], [1408, 832], [1216, 960], [960, 1216], [896, 1280], [1280, 896], [1472, 768], [768, 1472], [832, 1344], [1344, 832], [1088, 1024], [1024, 1088], [1152, 960], [960, 1152], [896, 1216], [1216, 896], [1408, 768], [768, 1408], [832, 1280], [1280, 832]],
    1152: [[1152, 1152], [1216, 1088], [1088, 1216], [896, 1472], [1472, 896], [1280, 1024], [1024, 1280], [960, 1344], [1344, 960], [1536, 832], [832, 1536], [896, 1408], [1408, 896], [1152, 1088], [1088, 1152], [1216, 1024], [1024, 1216], [960, 1280], [1280, 960], [832, 1472], [1472, 832], [896, 1344], [1344, 896]],
    1216: [[1216, 1216], [960, 1536], [1536, 960], [1280, 1152], [1152, 1280], [1344, 1088], [1088, 1344], [1408, 1024], [1024, 1408], [1600, 896], [896, 1600], [960, 1472], [1472, 960], [1216, 1152], [1152, 1216], [1280, 1088], [1088, 1280], [1664, 832], [832, 1664], [896, 1536], [1536, 896], [1344, 1024], [1024, 1344], [960, 1408], [1408, 960], [1600, 832], [832, 1600]],
    1280: [[1600, 1024], [1280, 1280], [1024, 1600], [1344, 1216], [1216, 1344], [1408, 1152], [1152, 1408], [1792, 896], [896, 1792], [1472, 1088], [1088, 1472], [1664, 960], [960, 1664], [1536, 1024], [1024, 1536], [1280, 1216], [1216, 1280], [1728, 896], [896, 1728], [1344, 1152], [1152, 1344], [960, 1600], [1600, 960], [1408, 1088], [1088, 1408], [1472, 1024], [1024, 1472], [1664, 896], [896, 1664]],
    1344: [[1344, 1344], [1408, 1280], [1280, 1408], [1472, 1216], [1216, 1472], [960, 1856], [1856, 960], [1024, 1728], [1728, 1024], [1536, 1152], [1152, 1536], [1600, 1088], [1088, 1600], [1792, 960], [960, 1792], [1344, 1280], [1280, 1344], [1408, 1216], [1216, 1408], [1664, 1024], [1024, 1664], [1472, 1152], [1152, 1472], [1536, 1088], [1088, 1536], [1728, 960], [960, 1728]],
    1408: [[1408, 1408], [1472, 1344], [1344, 1472], [1024, 1920], [1536, 1280], [1280, 1536], [1920, 1024], [1088, 1792], [1792, 1088], [1600, 1216], [1216, 1600], [1664, 1152], [1152, 1664], [1024, 1856], [1856, 1024], [1408, 1344], [1344, 1408], [1472, 1280], [1280, 1472], [1728, 1088], [1088, 1728], [1536, 1216], [1216, 1536], [1600, 1152], [960, 1920], [1920, 960], [1152, 1600], [1024, 1792], [1792, 1024], [1664, 1088], [1088, 1664]],
    1472: [[1472, 1472], [1536, 1408], [1408, 1536], [1088, 1984], [1984, 1088], [1600, 1344], [1344, 1600], [1856, 1152], [1152, 1856], [1664, 1280], [1280, 1664], [1728, 1216], [1216, 1728], [1024, 2048], [2048, 1024], [1088, 1920], [1920, 1088], [1472, 1408], [1408, 1472], [1792, 1152], [1536, 1344], [1344, 1536], [1152, 1792], [1600, 1280], [1280, 1600], [1024, 1984], [1984, 1024], [1664, 1216], [1216, 1664], [1088, 1856], [1856, 1088], [1728, 1152], [1152, 1728]],
    1536: [[1152, 2048], [1536, 1536], [2048, 1152], [1600, 1472], [1472, 1600], [1664, 1408], [1408, 1664], [1216, 1920], [1920, 1216], [1728, 1344], [1344, 1728], [1088, 2112], [2112, 1088], [1792, 1280], [1280, 1792], [1984, 1152], [1152, 1984], [1536, 1472], [1472, 1536], [1216, 1856], [1856, 1216], [1600, 1408], [1408, 1600], [1664, 1344], [1344, 1664], [1088, 2048], [2048, 1088], [1728, 1280], [1280, 1728], [1920, 1152], [1152, 1920], [1792, 1216], [1216, 1792]],
    1600: [[1600, 1600], [1664, 1536], [1536, 1664], [1728, 1472], [1472, 1728], [1280, 1984], [1984, 1280], [1792, 1408], [1408, 1792], [1152, 2176], [2176, 1152], [1344, 1856], [1856, 1344], [2048, 1216], [1216, 2048], [1600, 1536], [1536, 1600], [1280, 1920], [1920, 1280], [1664, 1472], [1472, 1664], [1152, 2112], [1728, 1408], [1408, 1728], [2112, 1152], [1216, 1984], [1984, 1216], [1792, 1344], [1344, 1792], [1280, 1856], [1856, 1280], [1088, 2176], [2176, 1088]],
    1664: [[1664, 1664], [1728, 1600], [1600, 1728], [1792, 1536], [1536, 1792], [2048, 1344], [1344, 2048], [1856, 1472], [1472, 1856], [2240, 1216], [1216, 2240], [1408, 1920], [2112, 1280], [1280, 2112], [1920, 1408], [1344, 1984], [1984, 1344], [1664, 1600], [1600, 1664], [1152, 2304], [1728, 1536], [2304, 1152], [1536, 1728], [2176, 1216], [1216, 2176], [1792, 1472], [1472, 1792], [2048, 1280], [1280, 2048], [1408, 1856], [1856, 1408], [1152, 2240], [2240, 1152], [1344, 1920], [1920, 1344], [2112, 1216], [1216, 2112]],
    1728: [[1728, 1728], [1792, 1664], [1664, 1792], [1408, 2112], [2112, 1408], [1856, 1600], [1600, 1856], [2432, 1216], [1216, 2432], [1920, 1536], [2304, 1280], [1536, 1920], [1280, 2304], [2176, 1344], [1344, 2176], [1472, 1984], [1984, 1472], [1408, 2048], [2048, 1408], [2368, 1216], [1216, 2368], [1728, 1664], [1664, 1728], [1792, 1600], [1600, 1792], [2240, 1280], [1280, 2240], [1856, 1536], [1536, 1856], [2112, 1344], [1344, 2112], [1472, 1920], [1920, 1472], [2304, 1216], [1216, 2304], [1408, 1984], [1984, 1408], [2176, 1280], [1280, 2176]],
    1792: [[1792, 1792], [1856, 1728], [1728, 1856], [1472, 2176], [2176, 1472], [1920, 1664], [2496, 1280], [1664, 1920], [1280, 2496], [2368, 1344], [1344, 2368], [1984, 1600], [1600, 1984], [2240, 1408], [1408, 2240], [1536, 2048], [2048, 1536], [2432, 1280], [1280, 2432], [1472, 2112], [2112, 1472], [1792, 1728], [1728, 1792], [2304, 1344], [1344, 2304], [1856, 1664], [1664, 1856], [1920, 1600], [1600, 1920], [1408, 2176], [2176, 1408], [1536, 1984], [1984, 1536], [2368, 1280], [1280, 2368], [1472, 2048], [2048, 1472], [2240, 1344], [1344, 2240]],
    1856: [[1856, 1856], [1920, 1792], [2560, 1344], [1792, 1920], [2240, 1536], [1536, 2240], [1344, 2560], [1984, 1728], [1728, 1984], [2432, 1408], [1408, 2432], [1664, 2048], [2048, 1664], [2304, 1472], [1472, 2304], [1600, 2112], [2112, 1600], [2496, 1344], [1344, 2496], [1536, 2176], [2176, 1536], [2368, 1408], [1408, 2368], [1856, 1792], [1792, 1856], [1920, 1728], [1728, 1920], [1984, 1664], [1664, 1984], [2240, 1472], [1472, 2240], [1280, 2560], [2560, 1280], [1600, 2048], [2048, 1600], [2432, 1344], [1344, 2432], [2304, 1408], [1536, 2112], [1408, 2304], [2112, 1536]],
    1920: [[1920, 1920], [2304, 1600], [1600, 2304], [1984, 1856], [1856, 1984], [2496, 1472], [1472, 2496], [1792, 2048], [2048, 1792], [1728, 2112], [2112, 1728], [2368, 1536], [1536, 2368], [1664, 2176], [2176, 1664], [2688, 1344], [1344, 2688], [2560, 1408], [1408, 2560], [2240, 1600], [1600, 2240], [2432, 1472], [1472, 2432], [1920, 1856], [1856, 1920], [1984, 1792], [1792, 1984], [1728, 2048], [2304, 1536], [1536, 2304], [2048, 1728], [2624, 1344], [1344, 2624], [2496, 1408], [1664, 2112], [1408, 2496], [2112, 1664], [2368, 1472], [1472, 2368], [1600, 2176], [2176, 1600]],
    1984: [[1984, 1984], [1920, 2048], [2560, 1536], [1536, 2560], [2048, 1920], [1856, 2112], [2112, 1856], [1792, 2176], [2176, 1792], [2432, 1600], [1600, 2432], [1408, 2752], [2752, 1408], [1728, 2240], [2240, 1728], [2624, 1472], [1472, 2624], [2496, 1536], [1664, 2304], [2304, 1664], [1536, 2496], [1984, 1920], [1920, 1984], [1856, 2048], [2048, 1856], [2368, 1600], [1600, 2368], [1792, 2112], [1408, 2688], [2112, 1792], [2688, 1408], [2560, 1472], [1472, 2560], [1728, 2176], [2176, 1728], [2432, 1536], [1536, 2432], [1664, 2240], [2240, 1664], [1408, 2624], [2624, 1408]],
    2048: [[2048, 2048], [1984, 2112], [2112, 1984], [1920, 2176], [2176, 1920], [1856, 2240], [2240, 1856], [2496, 1664], [1664, 2496], [2816, 1472], [1472, 2816], [2688, 1536], [1792, 2304], [2304, 1792], [1536, 2688], [2560, 1600], [1600, 2560], [1728, 2368], [2368, 1728], [1984, 2048], [2048, 1984], [1920, 2112], [2112, 1920], [2752, 1472], [1472, 2752], [2432, 1664], [1664, 2432], [1856, 2176], [2176, 1856], [2624, 1536], [1536, 2624], [1792, 2240], [2240, 1792], [2496, 1600], [1600, 2496], [1728, 2304], [2304, 1728], [2816, 1408], [1408, 2816], [1472, 2688], [2688, 1472], [2368, 1664], [1664, 2368]],
}


##preprocess images/caption, cache latents/hidden_encoder_states
#input:image_caption_pair_file tuple list
    #hashes & caches files & saves json list to disk
#returns list of json filepaths
def cache_image_caption_pair(
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
    ):

    #imports
    import math

    from diffusers import AutoencoderKL
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from torchvision.transforms import ToTensor, Resize
    from torchvision import transforms
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    scaler = GradScaler()


    #absolute data_dir
    if os.path.isabs(data_dir):
        abs_data_dir = data_dir
    else:
        abs_data_dir = os.path.abspath(data_dir)

    data_dir = abs_data_dir

    #absolute cache_dir
    if os.path.isabs(cache_dir):
        abs_cache_dir = cache_dir
    else:
        abs_cache_dir = os.path.abspath(cache_dir)

    cache_dir = abs_cache_dir

    #create paths
    basename = basename.lstrip('/')
    cache_dir = os.path.join(cache_dir, basename)

    os.makedirs(cache_dir, exist_ok=True)

    #welcome message
    print("initiating cache_image_caption_pair function")
    print(f"  cache_dir: {cache_dir}")
    
    with torch.no_grad(): #just to be safe

        #filter keys: min_resolution < keys < max_resolution
        filtered_aspect_categories = {key: value for key, value in aspect_categories.items() if max_resolution >= key >= min_resolution}
        sorted_categories_keys = sorted(filtered_aspect_categories)
        print(sorted_categories_keys)

        #variables
        json_file_paths_list = []

        #load individual components (not pipeline)
        #vae
        vae = AutoencoderKL.from_pretrained(
            pretrained_vae_model_name_or_path
        )
        vae.to(device)
        #tokenizers & text_encoders
        tokenizer_one = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder_cls_one = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder_cls_two = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2"
        )
        text_encoder_cls_one.to(device)
        text_encoder_cls_two.to(device)

        #initiate
        count = 0
        print("\nbegin cache_image_caption_pair")
        print(f"  --{len(image_caption_files_tuple_list)} image-caption.txt files")
        print("...")
        for i in range(len(image_caption_files_tuple_list)):

            #image & caption paths & relative paths
            image_file = image_caption_files_tuple_list[i][0]
            image_file_split = image_file.split(basename, 1)[1].lstrip('/')
            image_file_cache_path = os.path.join(cache_dir, image_file_split)
            caption_file = image_caption_files_tuple_list[i][1]
            caption_file_split = caption_file.split(basename, 1)[1].lstrip('/')
            caption_file_cache_path = os.path.join(cache_dir, caption_file_split)
            print(f"\nprocessing [{i}]:\n{image_file}")

            #to cache files' paths
            #os.path.join(cache_dir, data_dir_basename, f"{relative_file...
            json_file_path = f"{image_file_cache_path}.metadata.json"
            model_input_file = f"{image_file_cache_path}.latent.pkl"
            prompt_embed_file = f"{caption_file_cache_path}.prompt_embed.pkl"
            pooled_prompt_embed_file = f"{caption_file_cache_path}.pooled_prompt_embed.pkl"
            os.makedirs(os.path.dirname(model_input_file), exist_ok=True)


            #check if image-caption.txt pair already cached
            #if failed cached_file_integrity_check, will be re-cached
            #TO-DO: re-cache based failure message
            if os.path.exists(json_file_path):
                if cached_file_integrity_check(json_file_path) == "pass":
                    json_file_paths_list.append(json_file_path)
                    count += 1
                    print(f"  --already cached")
                    continue


            ######
            #begin processing image
            ######

            #reset upscaled metadata
            upscaled = False

            #try opening image file
            try:
                image = Image.open(image_file)
            except Exception as e: #failed opening
                error_message = f"Error: {e}, for {image_file}"
                print(error_message)
                logging.error(error_message)
                continue
            
            #hash image_file
            image_file_hash_value = hashlib.sha256(image_file.encode()).hexdigest()

            #calculate image dimensions & aspect ratio
            org_image_width, org_image_height = image.size
            original_aspect_ratio = org_image_width / org_image_height #get original aspect ratio 
            original_image_size = (org_image_width, org_image_height) #actual original_image_size

            #height & width by 64, image_pixels by 64, aspect ratio by 64
            image_width_64 = (org_image_width // 64) * 64
            image_height_64 = (org_image_height // 64) * 64
            image_pixels_64 = image_width_64 * image_height_64
            image_pixels_64_sqrt = int(math.sqrt(image_pixels_64))

            #check if image is too small
            if image_pixels_64 < min_resolution ** 2:
                print(" --too small")
                continue


            #convert image to RGB

            #if has transparency check
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                if image.mode != 'RGBA':
                    try:
                        image = image.convert('RGBA')
                    except Exception as e:
                        error_message = f"Error: {e}, for {image_file}"
                        print(error_message)
                        logging.error(error_message)
                        continue

                #based on alpha value, merge RGB with white background, convert to RGB
                white_bg = Image.new('RGBA', image.size, 'WHITE')
                image = Image.alpha_composite(white_bg, image)
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    error_message = f"Error: {e}, for {image_file}"
                    print(error_message)
                    logging.error(error_message)
                    continue

                print(f"  --converted: RGBA/LA/P w/ transparency --> RGB w/ white background")

            #no transparency
            else:
                try:
                    image = image.convert("RGB")
                except Exception as e:
                    error_message = f"Error: {e}, for {image_file}"
                    print(error_message)
                    logging.error(error_message)
                    continue


            #upscale & image cropping 

            #if upscale_to_resolution
            if upscale_to_resolution is not None:
                #image pixels less than upscale, get category_key,then use category key to verify needs upscale
                if image_pixels_64 <= upscale_to_resolution ** 2: #if image pixels <= upscaled image pixels
                    category_key = next((key for key in sorted_categories_keys if key >= image_pixels_64_sqrt), max_resolution) #category_key for image
                    if category_key < upscale_to_resolution: #image needs upscale

                        #get closest_bucket
                        category_key = upscale_to_resolution
                        aspect_ratios = aspect_categories[category_key]
                        original_aspect_ratio = org_image_width / org_image_height
                        closest_bucket = min(aspect_ratios, key=lambda x: abs((x[0]/x[1]) - original_aspect_ratio))

                        #calculate upscale_factor
                        upscale_factor_width = closest_bucket[0] / org_image_width
                        upscale_factor_height = closest_bucket[1] / org_image_height
                        upscale_factor = max(upscale_factor_width, upscale_factor_height)

                        #upscale image
                        print(f"  --upscaling image: {upscale_factor}")
                        image = Real_ESRGAN(
                            image,
                            upscale_factor,
                            upscale_use_GFPGAN,
                            image_file,
                            save_upscale_samples
                            )

                        #re-calculate_64 for upscaled image
                        up_image_width, up_image_height = image.size
                        up_image_width_64 = (up_image_width // 64) * 64
                        up_image_height_64 = (up_image_height // 64) * 64
                        image_pixels_64 = up_image_width_64 * up_image_height_64
                        image_pixels_64_sqrt = int(math.sqrt(image_pixels_64))

                        #upscaled metadata = True
                        upscaled = True


            #find closest bucket
            #find the largest category that is lower than or equal to image_pixels_64
            category_key = next((key for key in sorted_categories_keys if key >= image_pixels_64_sqrt), max_resolution)
            #print(f"category_key1: {category_key}")
            if category_key is None:
                #do not process images too small
                print("  --Error: category_key is None:")
                continue

            #aspect ratio process
            aspect_ratios = aspect_categories[category_key]
            original_aspect_ratio = org_image_width / org_image_height
            closest_bucket = min(aspect_ratios, key=lambda x: abs((x[0]/x[1]) - original_aspect_ratio))

            #resize and crop
            target_width, target_height = closest_bucket #target width based on bucket
            scaling_factor = min(image.width / target_width, image.height / target_height) #find how to resize
            new_width = int(image.width / scaling_factor)
            new_height = int(image.height / scaling_factor)
            image_resized = image.resize((new_width, new_height), Image.LANCZOS)

            # Calculate cropping coordinates to get the final image
            x1 = (new_width - target_width) // 2
            y1 = (new_height - target_height) // 2
            x2 = x1 + target_width
            y2 = y1 + target_height

            #crop
            image_cropped = image_resized.crop((x1, y1, x2, y2))
            del image_resized
            crop_top_left = (y1, x1)

            cropped_image_width, cropped_image_height = image_cropped.size
            target_size = (cropped_image_height, cropped_image_width) #for use with add_ids
            cropped_image_size = (cropped_image_width, cropped_image_height) #actual cropped_image_size
            cropped_aspect_ratio = cropped_image_width / cropped_image_height #get original aspect ratio 

            #downscale check #might use later
            downscaled = False
            if cropped_image_width * cropped_image_height < image_pixels_64:
                downscaled = True

            #original_size
            if upscaled == True:
                original_size = (image_height_64, image_width_64)
            else:
                original_size = (cropped_image_height, cropped_image_width)

            #add_time_id
            add_time_id = [
                original_size[0],
                original_size[1],
                crop_top_left[0],
                crop_top_left[1],
                target_size[0],
                target_size[1]
                ]

            #continue with cropped_image as image
            image = image_cropped
            del image_cropped


            #process image for caching

            #transform image
            train_transforms = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            ]
            )
            image = train_transforms(image) #returns normalized tensor

            #encode model_input (latent_image)
            
            with autocast(): #mixed-precision (fp16)
                pixel_values = image.to(memory_format=torch.contiguous_format).to(device).unsqueeze(0)
                model_input = vae.encode(pixel_values, return_dict=False)[0].sample() #[0] = remove bsz dimension
                #return_dict false, only returns tensor, so we sample directly. if return_dict=False, .latent_dist.sample() to access
            model_input = model_input[0] * vae.config.scaling_factor
            model_input_shape = model_input.shape
                #print(f"\nmodel_input: {model_input.shape}") #for 1024 image : torch.Size([4, 128, 128])
            del image, pixel_values

            #save model_input (latent_image) #model_input_file path created earlier
            joblib.dump(model_input, model_input_file)
            del model_input

            #latent_file_hash_value 
            model_input_file_hash_value = hashlib.sha256(model_input_file.encode()).hexdigest()

            gc.collect()
            torch.cuda.empty_cache()


            ######
            #finished processing image
            #begin process caption
            ######

            #setup & read caption
            prompt_embeds_list = []
            caption_file_hash_value = hashlib.sha256(caption_file.encode()).hexdigest()
            with open(caption_file, "r") as f:
                caption_string = f.read().strip() #test read file


            ##tokenizers & text_encoders

            #process prompt_embed_1

            #tokenize caption 1 #done on cpu
            tokenized_caption_1 = tokenizer_one(
                caption_string,
                max_length=tokenizer_one.model_max_length, #num_tokens specified by tokenizer
                padding="max_length", #pad tokens to meet model_max_length
                truncation=True, #crop excess tokens to meet model_max_length
                return_tensors="pt", #returns tensor
            ).input_ids #returns a dictionary: input_ids & attention_mask; .input_ids = the actual tokens

            #get prompt_embed
            with autocast(): #diffusers code uses mixed precision
                prompt_embed_1 = text_encoder_cls_one(tokenized_caption_1.to(device), output_hidden_states=True, return_dict=False)
                    #print(f"prompt_embed_1: {prompt_embed_1}") #returns a tuple of tensors

            #[0] = "pooled" output: intended to capture the essence of the entire input sequence in a fixed-size representation
            #"We are only ALWAYS interested in the pooled output of the final text encoder"
            #so pooled_prompt_embed_1 is discarded

            prompt_embed_1 = prompt_embed_1[-1][-2] #clip-skip 2: second-to-last layer
                #print(f"prompt_embed_1: {prompt_embed_1.shape}") #torch.Size([1, 77, 768])
            bs_embed, seq_len, _ = prompt_embed_1.shape #reshape: bsz, sequence_length, dimensions
            prompt_embed_1 = prompt_embed_1.view(bs_embed, seq_len, -1)[0] #remove bsz dimensions
                #print(f"prompt_embed_1: {prompt_embed_1.shape}") #torch.Size([77, 768])
            prompt_embeds_list.append(prompt_embed_1)
            del tokenized_caption_1, prompt_embed_1

            #process prompt_embed_2

            #tokenize caption 2 #done on cpu
            tokenized_caption_2 = tokenizer_two(
                caption_string,
                max_length=tokenizer_two.model_max_length, # num_tokens specified by tokenizer
                padding="max_length", #pad tokens to meet model_max_length
                truncation=True, #crop excess tokens to meet model_max_length
                return_tensors="pt", #returns tensor
            ).input_ids #returns a dictionary: input_ids & attention_mask; .input_ids = the actual tokens

            #get encoder_hidden_state
            with autocast(): #diffusers code uses mixed precision
                prompt_embed_2 = text_encoder_cls_two(tokenized_caption_2.to(device), output_hidden_states=True, return_dict=False)
                #print(f"prompt_embed_2: {prompt_embed_2}\n") #returns a tuple of tensors

            #[0] = "pooled" output: intended to capture the essence of the entire input sequence in a fixed-size representation
            #pooled_prompt_embed_2 is kept
            pooled_prompt_embed_2 = prompt_embed_2[0] #return "pooled" output
                #print(f"pooled_prompt_embed_2: {pooled_prompt_embed_2.shape}") #torch.Size([1, 1280])

            prompt_embed_2 = prompt_embed_2[-1][-2]
                #print(f"prompt_embed_2: {prompt_embed_2.shape}") #torch.Size([1, 77, 1280])
            bs_embed, seq_len, _ = prompt_embed_2.shape
            prompt_embed_2 = prompt_embed_2.view(bs_embed, seq_len, -1)[0] 
                #print(f"prompt_embed_2: {prompt_embed_2.shape}") #torch.Size([77, 1280])
            prompt_embeds_list.append(prompt_embed_2)
            del tokenized_caption_2, prompt_embed_2
            
            #concatenate prompt_embed(TE1+TE2)
            prompt_embed = torch.concat(prompt_embeds_list, dim=-1) #torch.Size([77, 2048])
            prompt_embed_shape = prompt_embed.shape
            del prompt_embeds_list
                #print(f"prompt_embed: {prompt_embed.shape}") #torch.Size([77, 768])

            #re-shape pooled_prompt_embed(TE2)
            pooled_prompt_embed = pooled_prompt_embed_2.view(bs_embed, -1)[0]
            pooled_prompt_embed_shape = pooled_prompt_embed.shape
            del pooled_prompt_embed_2
                #print(f"pooled_prompt_embed: {pooled_prompt_embed.shape}\n") #torch.Size([1280])


            #create embed filename by 1. remove root  2. prepend cache 3. append hash_value.latent.pkl

            #prompt_embed_file & pooled_prompt_embed_file: cache & hash # file paths created earlier
            joblib.dump(prompt_embed, prompt_embed_file)
            prompt_embed_file_hash_value = hashlib.sha256(prompt_embed_file.encode()).hexdigest()
            del prompt_embed
            joblib.dump(pooled_prompt_embed, pooled_prompt_embed_file)
            pooled_prompt_embed_file_hash_value = hashlib.sha256(pooled_prompt_embed_file.encode()).hexdigest()
            del pooled_prompt_embed


            gc.collect()
            torch.cuda.empty_cache()

            ######
            #finished processing caption
            ######


            #setup & save cache json_file
            metadata = {
                "basename": basename,
                "data_dir": data_dir,
                "cache_dir": cache_dir,
                "image_file": image_file,
                "org_image_height": org_image_height,
                "org_image_width": org_image_width,		
                "original_size": original_size,
                "original_image_size": original_image_size,
                "category_key": category_key,
                "closest_bucket": closest_bucket,
                "crop_top_left": crop_top_left,
                "cropped_image_height": cropped_image_height,
                "cropped_image_width": cropped_image_width,
                "target_size": target_size,
                "cropped_image_size": cropped_image_size,
                "downscale": downscaled,
                "upscaled": upscaled,
                "original_aspect_ratio": original_aspect_ratio,
                "cropped_aspect_ratio": cropped_aspect_ratio,
                "image_file_hash_value": image_file_hash_value,
                "model_input_file": model_input_file,
                "model_input_shape": model_input_shape,
                "model_input_file_hash_value": model_input_file_hash_value,
                "caption_file": caption_file,
                "caption_string": caption_string,
                "caption_file_hash_value": caption_file_hash_value,
                "prompt_embed_file": prompt_embed_file,
                "prompt_embed_shape": prompt_embed_shape,
                "prompt_embed_file_hash_value": prompt_embed_file_hash_value,
                "pooled_prompt_embed_file": pooled_prompt_embed_file,
                "pooled_prompt_embed_shape": pooled_prompt_embed_shape,
                "pooled_prompt_embed_file_hash_value": pooled_prompt_embed_file_hash_value,
                "add_time_id": add_time_id,
            }

            with open(json_file_path, "w") as f:
                json.dump(metadata, f, indent=4)


            #cache image-caption.txt pair complete
            json_file_paths_list.append(json_file_path)
            count += 1
            print(f"  --processed: [{category_key}]: {closest_bucket}")

            #clean-up
            del caption_string
            gc.collect()
            torch.cuda.empty_cache() 


        #save json_file_paths_list as data_dir.txt
        json_file_paths_list.sort()
        json_file_paths_list_txt = os.path.join(cache_dir, data_dir.replace(os.sep, "_") + ".list")
        with open(json_file_paths_list_txt, "w") as f: #erase file
            pass
        with open(json_file_paths_list_txt, "a") as f:
            for item in json_file_paths_list:
                f.write(f"{item}\n")

        print("...")
        print(f"\n{count} image-caption.txt pairs cached")
        return json_file_paths_list


#cached file integrity check
def cached_file_integrity_check(json_file_path):
    with open(json_file_path, "r") as f:
        metadata = json.load(f)

    #verify image_image_file
    image_file = metadata["image_file"]
    recalculated_image_file_hash_value = hashlib.sha256(image_file.encode()).hexdigest()
    if os.path.exists(image_file) and recalculated_image_file_hash_value == metadata["image_file_hash_value"]:
        pass
    else:
        return "\nimage_file_fail"
    
    #verify caption_file
    caption_file = metadata["caption_file"]
    recalculated_caption_file_hash_value = hashlib.sha256(caption_file.encode()).hexdigest()
    if os.path.exists(caption_file) and recalculated_caption_file_hash_value == metadata["caption_file_hash_value"]:
        pass
    else:
        return "\ncaption_file_fail"

    #verify model_input_file
    model_input_file = metadata["model_input_file"]
    recalculated_model_input_file_hash_value = hashlib.sha256(model_input_file.encode()).hexdigest()
    if os.path.exists(model_input_file) and recalculated_model_input_file_hash_value == metadata["model_input_file_hash_value"]:
        pass
    else:
        return "\nmodel_input_file_fail"
    
    #verify prompt_embed_file
    prompt_embed_file = metadata["prompt_embed_file"]
    recalculated_prompt_embed_file_hash_value = hashlib.sha256(prompt_embed_file.encode()).hexdigest()
    if os.path.exists(prompt_embed_file) and recalculated_prompt_embed_file_hash_value == metadata["prompt_embed_file_hash_value"]:
        pass
    else:
        return "\nprompt_embed_file_fail"
    
    #verify pooled_prompt_embed_file
    pooled_prompt_embed_file = metadata["pooled_prompt_embed_file"]
    recalculated_pooled_prompt_embed_file_hash_value = hashlib.sha256(pooled_prompt_embed_file.encode()).hexdigest()
    if os.path.exists(pooled_prompt_embed_file) and recalculated_pooled_prompt_embed_file_hash_value == metadata["pooled_prompt_embed_file_hash_value"]:
        pass
    else:
        return "\npooled_prompt_embed_file_fail"

    #success
    gc.collect()
    return "pass"


def Real_ESRGAN(image, outscale, upscale_use_GFPGAN, image_file, save_upscale_samples):
    #Real-ESRGAN is BSD-3-Clause license
    #License: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE 
    #Original code: https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    #The code has been modified to suite the needs of this script

    #Real_ESRGAN & GFPGAN
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
    import numpy as np

    import cv2

    #Adjustable options Real_ESRGAN & GFPGAN.  Just left things at default.
    tile = 0 #type=int, default=0, help='Tile size, 0 for no tile during testing')
    tile_pad = 10 #type=int, default=10, help='Tile padding')
    pre_pad = 0 #type=int, default=0, help='Pre padding size at each border')
    fp32 = False #action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    gpu_id = None #Not needed, each process only runs on 1 gpu, can't see other gpus

    model_name = "RealESRGAN_x4plus"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    #file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    model_path = "RealESRGAN_x4plus.pth"

    #RealESRGAN doesn't use dni_weight, normally use dni to control the denoise strength
    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True,
        gpu_id=gpu_id)

    if upscale_use_GFPGAN == True:
        face_enhancer = GFPGANer(
            #model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            model_path="GFPGANv1.3.pth",
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )

    #convert image to np.array for RealESRGAN & GFPGAN
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #save copies of upscaled images, in case you want to check quality
    #save copy of org_image
    if save_upscale_samples == True:
        upscale_dir = "upscale_samples"
        dir_path = os.path.dirname(image_file)
        upscale_dir_path = os.path.join(upscale_dir, dir_path)
        os.makedirs(upscale_dir_path, exist_ok=True)
        filename = os.path.basename(image_file)
        org_image = f"{filename}_org.jpg"
        org_image_path = os.path.join(upscale_dir_path, org_image)
        image.save(org_image_path, 'JPEG')
    
    #RealESRGAN upscale
    output, _ = upsampler.enhance(img, outscale=outscale)
    
    #save copy of RealESRGAN_image
    if save_upscale_samples == True:
        image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        RealESRGAN_image = f"{filename}_RealESRGAN.jpg"
        RealESRGAN_image_path = os.path.join(upscale_dir_path, RealESRGAN_image)
        image.save(RealESRGAN_image_path, 'JPEG')
    
    #GFPGAN face_enchance
    if upscale_use_GFPGAN == True:
        _, _, output = face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)
    
        #save copy of GFPGAN_image
        if save_upscale_samples == True:
            image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            GFPGAN_image = f"{filename}_GFPGAN.jpg"
            GFPGAN_image_path = os.path.join(upscale_dir_path, GFPGAN_image)
            image.save(GFPGAN_image_path, 'JPEG')

    #convert image back to pil.Image
    image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return image