#sdxl_validation_functions
#validation functions for sdxl_FSDP_train
    #make_sample_images: makes sample images
    #calculate_validation_image_scores: calculates validation image scores
    #calculate_validation_loss: calculates validation loss

import gc
import json
import os
import time
import random

from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F #for F.mse_loss: mean squared error (MSE)
#from torch.utils.data import random_split #validation split based on buckets, and is not 100% random
from tqdm.auto import tqdm


#creates sample images
def make_sample_images(
        sample_count,
        sample_page,
        pipeline,
        generator,
        accelerator,
        sample_image_prompts,
        epoch,
        output_dir,
        train_name
    ):
    
    import textwrap

    accelerator.print("\nbeginning sample generation:")

    #generate sample images
    sample_images = []
    
    for prompt in sample_image_prompts:
        with torch.no_grad(): #just to be safe
            image = pipeline(prompt, num_inference_steps=30, generator=generator).images[0]
        sample_images.append(image)

    del pipeline
    del generator
    
    #create epoch marker image
    image_size = sample_images[0].size
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("LibreBaskerville-DpdE.ttf", 72)
    if epoch == 0:
        text = "Base Model"
    else:
        text = f"Epoch: {epoch}"
    x_position = 10 #10 pixels from left
    y_position = image_size[1] // 2 #centers text
    draw.text((x_position, y_position), text, fill="black", anchor="lm", font=font) #ls: l=start at x_position, s=center on y_position
    sample_images.insert(0, img)

    #create and save grid_image
    grid_image = make_image_grid(sample_images, rows=1, cols=len(sample_images)) #change to based on number of samples generated
    filename = os.path.join(output_dir, f"{train_name}_{sample_page}.jpg")

    if sample_count == 0:
        #create prompt label image
        prompt_labels = []
        #blank image
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        prompt_labels.append(img)
        #sample prompt images
        for prompt in sample_image_prompts:
            img = Image.new('RGB', image_size, color='white')
            draw = ImageDraw.Draw(img)
            text = prompt
            lines = textwrap.wrap(text, width=35)
            font = ImageFont.truetype("LibreBaskerville-DpdE.ttf", 52)
            x_position = 20 #10 pixels from left
            y_position = 20 #10 pixels from top
            for line in lines:
                draw.text((x_position, y_position), line, fill="black", anchor="lt", font=font) #lt: l=start at x_position, t=below y_position
                y_position += 75
            prompt_labels.append(img)
        #make image grid
        prompt_grid_image = make_image_grid(prompt_labels, rows=1, cols=len(prompt_labels))
        #paste images to prompt_grid_image
        new_height = prompt_grid_image.height + grid_image.height
        combined_image = Image.new('RGB', (prompt_grid_image.width, new_height), "white")
        combined_image.paste(prompt_grid_image, (0, 0))
        combined_image.paste(grid_image, (0, prompt_grid_image.height))
        combined_image.save(filename, quality=95)
        print(f"sample image saved: {filename}")

        del prompt_labels, text, lines, prompt_grid_image
    
    else:
        #append new images to bottom of existing image
        existing_image = Image.open(filename)
        new_height = existing_image.height + grid_image.height
        combined_image = Image.new('RGB', (existing_image.width, new_height), "white")
        combined_image.paste(existing_image, (0, 0))
        combined_image.paste(grid_image, (0, existing_image.height))
        combined_image.save(filename, quality=95)
        print(f"sample image appended: {filename}")

        del existing_image


    del image_size, img, draw, font, x_position, y_position, sample_images
    del grid_image, combined_image, new_height

    gc.collect()
    torch.cuda.empty_cache()


#calculates validation_image scores, uses distributed state, not dataset/DataLoader
def calculate_validation_image_scores(pipeline, generator, device, accelerator, validation_image_jsons, epoch, writer, output_dir):

    with torch.no_grad(): #validation, no gradients

        #imports for validation_image
        from accelerate import PartialState
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        import lpips
        import numpy as np
        from scipy.linalg import sqrtm
        import torchvision.models as models
        from torchvision import transforms

        import time #time was acting strange if not imported inside function
        start_time = time.time()
        count = 0
        
        #welcome message
        accelerator.print("\nbeginning image generation for validation_image:")

        #setup distributed_state
        distributed_state = PartialState()
        pipeline.to(distributed_state.device)
        val_image_idx = 0 #tracking jsons, per GPU, initially used to check correct division of jsons among gpus


        ##load validation models & variables
        
        #check if models already downloaded, if not, download on main process
        accelerator.wait_for_everyone()
        lpips_checkpoint = os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
        IS_checkpoint = os.path.expanduser("~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth")
        if accelerator.is_main_process:
            if not os.path.exists(lpips_checkpoint) or not os.path.exists(IS_checkpoint):
                inception_model = models.inception_v3(weights="DEFAULT").eval().to(distributed_state.device)
                lpips_model = lpips.LPIPS(net='vgg').to(distributed_state.device) #depreciated
        accelerator.wait_for_everyone()
        
        #IS: inception score setup
        
        inception_model = models.inception_v3(weights="DEFAULT").eval().to(distributed_state.device)
        softmax = torch.nn.Softmax(dim=1)
        preds_list = []

        #LPIPS setup
        lpips_model = lpips.LPIPS(net='vgg').to(distributed_state.device) #depreciated
        lpips_scores = []

        #FID & KID setup
        def get_features(module, input, output):
            features.append(output.flatten(1))
        features = [] #register forward hook
        hook_handle = inception_model.avgpool.register_forward_hook(get_features)

        gathered_reference_images_features_pt_file = os.path.join(output_dir, "gathered_reference_images_features.pt")
        reference_images_features = [] 
        generated_images_features = []

        
        ##HPSv2 setup
        #HPSv2 is Apache-2.0 license 
        #License: https://github.com/tgxs002/HPSv2/blob/master/LICENSE
        #Original code: https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
        #The code has been modified to suite the needs of this script
            #specifically because importing hpsv2 & using hpsv2.score caused Cuda OOMs on next epoch
        
        #get model?
        model_dict = {}
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        #load checkpoint, to.(device)
        cp = "HPS_v2.1_compressed.pt"
        checkpoint = torch.load(cp, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(distributed_state.device)
        model.eval()
        

        #HPSv2 score list
        hpsv2_scores = []


        ##split validation_image_jsons among GPUs
        with distributed_state.split_between_processes(validation_image_jsons, apply_padding=True) as json_list:

            for json_file in json_list:
                #tracking json per GPU
                val_image_idx += 1 #count idx per json, for tracking to ensure json_list division between GPUs
                accelerator.print(f"\rvalidating: [{val_image_idx}]", end="")
                #read json metadata, get info for reference image & generated image parameters
                with open(json_file, "r") as f: #open and read json file
                    metadata = json.load(f)
                prompt = metadata["caption_string"]
                height = metadata["cropped_image_height"]
                width = metadata["cropped_image_width"]
                reference_image_path = metadata["image_file"]
                reference_image = Image.open(reference_image_path).convert("RGB")

                #generate image
                generated_image = pipeline(prompt, num_inference_steps=30, generator=generator, height=height, width=width).images[0]

                #IS: inception score
                inception_preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                inception_image = inception_preprocess(generated_image.convert("RGB")).unsqueeze(0).to(distributed_state.device)
                inception_output = inception_model(inception_image) #score image
                preds = softmax(inception_output) #softmax probabilities
                preds_list.append(preds)

                #LPIPS
                LPIPS_preprocess = transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                LPIPS_generated_image = LPIPS_preprocess(generated_image.convert("RGB")).unsqueeze(0).to(distributed_state.device)
                LPIPS_reference_image = LPIPS_preprocess(reference_image.convert("RGB")).unsqueeze(0).to(distributed_state.device)
                lpips_score = lpips_model(LPIPS_generated_image, LPIPS_reference_image)
                lpips_scores.append(lpips_score)

                #FID
                #since it uses inception_model, some duplicates
                #generated image
                FID_generated_image = LPIPS_generated_image #IS duplicate
                inception_model(FID_generated_image) #IS duplicate
                generated_images_features.append(features[-1])
                features.clear()
                #reference_image
                if not os.path.exists(gathered_reference_images_features_pt_file):
                    FID_reference_image = LPIPS_reference_image 
                    inception_model(FID_reference_image)
                    reference_images_features.append(features[-1])
                    features.clear()
                    del FID_reference_image #del here, because later epoch don't create this file

                #KID
                #KID uses FID data for calculations

                #HPSv2
                #generated_image & prompt
                image = preprocess_val(generated_image).unsqueeze(0).to(device=distributed_state.device, non_blocking=True)
                text = tokenizer([prompt]).to(device=distributed_state.device, non_blocking=True)
                with accelerator.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                hpsv2_scores.append(torch.tensor(hps_score, dtype=torch.float16).to(device))

                #local_count
                count += 1


        #clean-up
        #image generation
        del pipeline
        del generator
        del generated_image
        #inception score
        del inception_model
        del softmax
        del inception_image
        del inception_output
        del preds
        #LPIPS
        del lpips_model
        del LPIPS_generated_image
        del LPIPS_reference_image
        del lpips_score
        #FID
        del FID_generated_image
        #HPSv2
        del model_dict, preprocess_train, preprocess_val, model, cp, checkpoint, tokenizer
        
        gc.collect()
        torch.cuda.empty_cache()


        ##prepare values/list, gather on main process, calculate& log scores

        #values/lists to tensors, since gather works with tensors
        accelerator.wait_for_everyone()
        #inception score
        preds_tensor = torch.cat(preds_list, dim=0) #convert to tensor,
        #LPIPS
        lpips_tensor = torch.cat(lpips_scores, dim=0)
        #FID
        hook_handle.remove()
        generated_images_features_tensor = torch.cat(generated_images_features, dim=0)
        if not os.path.exists(gathered_reference_images_features_pt_file):
            reference_images_features_tensor = torch.cat(reference_images_features, dim=0)
        #HPSv2
        hpsv2_tensor = torch.cat(hpsv2_scores, dim=0)

        #gather values/lists of tensors
        accelerator.wait_for_everyone()
        #inception score
        gathered_preds = accelerator.gather(preds_tensor) #gather preds
        ##LPIPS score
        gathered_lpips = accelerator.gather(lpips_tensor)
        #FID
        gathered_generated_images_features = accelerator.gather(generated_images_features_tensor)
        if not os.path.exists(gathered_reference_images_features_pt_file):
            gathered_reference_images_features = accelerator.gather(reference_images_features_tensor)
            torch.save(gathered_reference_images_features, gathered_reference_images_features_pt_file)
        else:
            gathered_reference_images_features = torch.load(gathered_reference_images_features_pt_file)
            gathered_reference_images_features = gathered_reference_images_features.to(device)
        #HPSv2
        gathered_hpsv2 = accelerator.gather(hpsv2_tensor)

        #clean-up
        #inception score
        del preds_list
        del preds_tensor
        #LPIPS
        del lpips_scores
        del lpips_tensor
        #HPSv2
        del hpsv2_tensor, hpsv2_scores

        gc.collect()
        torch.cuda.empty_cache()


        #compute global average scores
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:

            #inception score calculate
            marginal_probs = torch.mean(gathered_preds, dim=0)
            kl_div = gathered_preds * (torch.log(gathered_preds + 1e-5) - torch.log(marginal_probs + 1e-5))
            kl_div = torch.mean(torch.sum(kl_div, dim=1))
            inception_score = torch.exp(kl_div).item()
            #inception score log
            print(f"Average Inception Score: {inception_score}")
            writer.add_scalar("Validation/Inception Score per epoch", inception_score, epoch)
            #inception score clean
            del marginal_probs
            del kl_div
            del inception_score

            #LPIPS score calculate
            average_lpips_score = torch.mean(gathered_lpips).item()
            #LPIPS score log
            print(f"Average LPIPS Score: {average_lpips_score}")
            writer.add_scalar("Validation/LPIPS Score per epoch", average_lpips_score, epoch)
            #LPIPS score clean
            del average_lpips_score

            #FID score
            def compute_mean_covariance(features):
                mean_vec = torch.mean(features, dim=0)
                centered_features = features - mean_vec
                cov_matrix = torch.matmul(centered_features.T, centered_features) / (features.shape[0] - 1)
                return mean_vec, cov_matrix
            gen_mean, gen_cov = compute_mean_covariance(gathered_generated_images_features)
            real_mean, real_cov = compute_mean_covariance(gathered_reference_images_features)

            def compute_fid(real_mean, real_cov, gen_mean, gen_cov):
                diff = real_mean - gen_mean
                covmean = sqrtm(real_cov.dot(gen_cov))
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                fid = diff.dot(diff) + np.trace(real_cov + gen_cov - 2 * covmean)
                return fid
            fid_score = compute_fid(
                real_mean.cpu().numpy(), 
                real_cov.cpu().numpy(), 
                gen_mean.cpu().numpy(), 
                gen_cov.cpu().numpy()
            )
            #FID score log
            print(f"FID Score: {fid_score}")
            writer.add_scalar("Validation/FID Score per epoch", fid_score, epoch)
            #FID score clean
            del gen_mean, gen_cov
            del real_mean, real_cov
            del fid_score

            #KID score
            def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
                if Y is None:
                    Y = X
                if gamma is None:
                    gamma = 1.0 / X.size(1)
                K = torch.mm(X, Y.t())
                K.mul_(gamma).add_(coef0).pow_(degree)
                return K
            def compute_kid_score(generated_features, reference_features):
                K_xx = polynomial_kernel(generated_features).mean()
                K_yy = polynomial_kernel(reference_features).mean()
                K_xy = polynomial_kernel(generated_features, reference_features).mean()
                kid_score = K_xx + K_yy - 2 * K_xy
                return kid_score.item()
            kid_score = compute_kid_score(gathered_generated_images_features, gathered_reference_images_features)
            #KID score log
            print(f"KID Score: {kid_score}")
            writer.add_scalar("Validation/KID Score per epoch", kid_score, epoch)

            #HPSv2 avg score
            average_hpsv2_score = torch.mean(gathered_hpsv2).item()
            #HPSv2 score log
            print(f"Average HPSv2 Score: {average_hpsv2_score}")
            writer.add_scalar("Validation/HPSv2 Score per epoch", average_hpsv2_score, epoch)
            #LPIPS score clean
            del average_hpsv2_score


        #clean-up
        accelerator.wait_for_everyone()
        #inception_score
        del gathered_preds
        #LPIPS
        del gathered_lpips
        #HPSv2
        del gathered_hpsv2

        gc.collect()
        torch.cuda.empty_cache()

        #calculate time per image
        if accelerator.is_main_process:
            end_time = time.time()
            print(f"count: [{count}]")
            total_time = end_time - start_time
            total_time = total_time
            print(f"total_time: {total_time / 60:.2f}minutes")
            print(f"images_per_gpu: {count}")
            secs_img = total_time / count
            print(f"seconds per imgs: {secs_img:.2f}seconds/img")


def calculate_validation_loss(accelerator, unet, validation_loss_dataloader, val_batch_size, num_val_steps_per_epoch, num_processes, noise_scheduler, writer, epoch):

    with torch.no_grad():
        if accelerator.is_main_process:
            val_start_time = time.time()

        if accelerator.is_main_process:
            print(f"\nEpoch #{epoch} loss validation")

            #create a new progress bar for each epoch
            if accelerator.is_main_process:
                progress_bar = tqdm(
                    range(num_val_steps_per_epoch),
                    initial=0,
                    desc=f"validation_loss step",
                )
        #variables for local_total_val_loss
        val_loss = 0
        count = 0

        for step, batch in enumerate(validation_loss_dataloader):

            #sample noise to add to latents
            noise = torch.randn_like(batch["model_input"]) #create noise in the shape of latent tensor
            bsz = batch["model_input"].shape[0] #bsz = batch size

            #sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=batch["model_input"].device)
            timesteps = timesteps.long() #.long() = convert to 64-bit signed integer data type: -2^63 to 2^63-1

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
            avg_loss = accelerator.gather(loss.repeat(val_batch_size)).mean()
            val_loss += avg_loss
            count += 1

            #update progress bar
            if accelerator.is_main_process:
                    val_imgs = count * num_processes * val_batch_size
                    val_end_time = time.time()
                    val_img_sec_total_time = val_end_time - val_start_time 
                    val_imgs_sec = val_imgs / val_img_sec_total_time
                    progress_bar.set_postfix({
                        "imgs/s": f"{val_imgs_sec:.2f}", # .Xf, where X = num decimal places
                        "loss": f"{loss.item():.4f}"
                    })
                    progress_bar.update()  #update progress bar after each step

        #calculate avg_val_loss
        avg_val_loss = val_loss / count

        #log avg_val_loss
        if accelerator.is_main_process:
            writer.add_scalar("Validation/loss (epoch)", avg_val_loss, epoch)

        #clean-up
        del loss, avg_loss, val_loss, count
        gc.collect()
        torch.cuda.empty_cache()


'''#not currently used
def split_dataset(dataset, train_percent, seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    total_size = len(dataset)
    train_size = int(train_percent * total_size)
    validate_size = total_size - train_size
    
    return random_split(dataset, [train_size, validate_size])
'''