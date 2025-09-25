import gc
import os
import pickle
import shutil
import subprocess
import sys
import csv
sys.path.append(os.getcwd())
# os.environ["OMP_NUM_THREADS"] = "1"
import time
import traceback
from pathlib import Path
import glob
import cv2
import deepspeed
from utils import get_arguments
import moviepy.editor as mp
import numpy as np
import torch
import torch.distributed
from omegaconf import OmegaConf
import getpass
from scipy.io import wavfile
from torch.amp import autocast
from safetensors.torch import load_file
from tqdm import tqdm
from ovi.distributed_comms.util import get_device, get_world_size, get_local_rank, get_global_rank
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from utils import (
    init_fusion_score_model_ovi, 
    init_text_model, init_vae
)
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info

global_rank = get_global_rank()


def preprocess_video_tensor(video, device, target_dtype, h_w_multiple_of=32, resize_total_area=None):
    """Preprocess video data into standardized tensor format and (optionally) resize area.
    
    Args:
        video: numpy array or tensor. Accepts (C,F,H,W) or (F,H,W,C) or torch with 4/5 dims.
        device: torch device.
        target_dtype: desired torch dtype.
        h_w_multiple_of: snap H and W down to multiples of this after area targeting.
        resize_total_area: Target *area* (H*W). Examples:
            - 589824 (int)
            - (768, 768) or [768, 768]
            - "768x768", "768*768"
    """
    import re
    import numpy as np
    import torch

    def _parse_area(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]) * int(val[1])
        if isinstance(val, str):
            m = re.match(r"\s*(\d+)\s*[x\*\s]\s*(\d+)\s*$", val, flags=re.IGNORECASE)
            if m:
                return int(m.group(1)) * int(m.group(2))
            # if it's a plain integer in a string
            if val.strip().isdigit():
                return int(val.strip())
        raise ValueError(f"resize_total_area={val!r} is not a valid area or WxH.")

    def _best_hw_for_area(h, w, area_target, multiple):
        """
        Find H', W' that:
          - keep aspect ratio ~ h:w,
          - H' = multiple * p, W' = multiple * q for integers p,q >= 1,
          - area close to area_target.
        """
        if area_target <= 0:
            return h, w

        ratio_wh = w / float(h)
        # Work in units of 'multiple'
        area_unit = multiple * multiple
        tgt_units = max(1, area_target // area_unit)  # area in units of (multiple^2)

        # Initial p based on solving p*q ≈ tgt_units and q/p ≈ ratio_wh
        # p ≈ sqrt(tgt_units / ratio_wh), q ≈ ratio_wh * p
        p0 = max(1, int(round(np.sqrt(tgt_units / max(ratio_wh, 1e-8)))))
        candidates = []
        for dp in range(-3, 4):
            p = max(1, p0 + dp)
            q = max(1, int(round(p * ratio_wh)))
            H = p * multiple
            W = q * multiple
            candidates.append((H, W))

        # also consider direct scaled rounding path as a fallback candidate
        scale = np.sqrt(area_target / (h * float(w)))
        H_sc = max(multiple, int(round(h * scale / multiple)) * multiple)
        W_sc = max(multiple, int(round(w * scale / multiple)) * multiple)
        candidates.append((H_sc, W_sc))

        # Pick the candidate whose area is closest to target (tie-breaker: closer aspect)
        def score(HW):
            H, W = HW
            area = H * W
            return (abs(area - area_target), abs((W / max(H, 1e-8)) - ratio_wh))

        H_best, W_best = min(candidates, key=score)
        return H_best, W_best

    # ---- Build tensor (your original logic) ----
    if isinstance(video, np.ndarray):
        if video.shape[0] == 3:
            video_tensor = torch.from_numpy(video).float().unsqueeze(0).to(device, dtype=target_dtype)
        elif video.shape[-1] == 3:
            video_tensor = torch.from_numpy(video).float().permute(3, 0, 1, 2).unsqueeze(0).to(device, dtype=target_dtype)
        else:
            # assume already (C,F,H,W) style
            video_tensor = torch.from_numpy(video).float().to(device, dtype=target_dtype)
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.transpose(0, 1).unsqueeze(0)
    else:
        video_tensor = video.to(device, dtype=target_dtype)
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.transpose(0, 1).unsqueeze(0)

    # Normalize into [-1, 1]
    if video_tensor.max() > 1.0:
        video_tensor = video_tensor / 255.0
    video_tensor = video_tensor * 2.0 - 1.0

    # ---- Compute target size (optional area-based resize) ----
    _, _, f, h, w = video_tensor.shape

    # If a target area is specified, choose (new_h, new_w) to match it (and multiples)
    area_target = _parse_area(resize_total_area)
    if area_target is not None:
        # pick best H',W' under constraints
        target_h, target_w = _best_hw_for_area(h, w, area_target, h_w_multiple_of)
        print(f"\nResizing video from {h}x{w} to {target_h}x{target_w} to target area {area_target}")
    else:
        # no area targeting; we’ll just snap down to multiples from the *current* (h, w)
        target_h = (h // h_w_multiple_of) * h_w_multiple_of
        target_w = (w // h_w_multiple_of) * h_w_multiple_of

    # safety: ensure strictly positive and at least one multiple
    target_h = max(h_w_multiple_of, int(target_h))
    target_w = max(h_w_multiple_of, int(target_w))

    # If nothing changes, skip interpolation
    if (h != target_h) or (w != target_w):
        # Resize spatial dims; retains frames as "channels" exactly like your original code
        video_tensor = torch.nn.functional.interpolate(
            video_tensor.view(-1, f, h, w),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).view(1, -1, f, target_h, target_w)

    return video_tensor

    
def _save_video_frames_as_mp4(video_frames, output_path, fps=24):
    """
    Save video frames as MP4 file using cv2.
    
    Args:
        video_frames: numpy array of shape (f, h, w, c) or (c, f, h, w)
        output_path: path to save the MP4 file
        fps: frames per second
    """
    if video_frames.ndim == 4:
        if video_frames.shape[0] <= 4:  # Assume (c, f, h, w)
            video_frames = video_frames.transpose(1, 2, 3, 0)  # (f, h, w, c)
        # else assume (f, h, w, c)
    
    # Ensure values are in [0, 255] range
    if video_frames.max() <= 1.0:
        video_frames = np.clip(video_frames, -1, 1)
        video_frames = (video_frames + 1) / 2  # Scale to [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
    else:
        video_frames = video_frames.astype(np.uint8)
    
    f, h, w, c = video_frames.shape
    
    # Convert RGB to BGR for OpenCV
    if c == 3:
        video_frames = video_frames[..., ::-1]  # RGB to BGR
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Write frames
    for frame in video_frames:
        out.write(frame)
    
    out.release()

def _combine_video_audio_to_mp4(video_path, audio_path, output_path):
    """
    Combine video file with audio file into a single MP4 using moviepy.
    
    Args:
        video_path: path to video file
        audio_path: path to audio file  
        output_path: path to save combined MP4
    """
    try:
        # Load video and audio
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        
        # Match durations - use the shorter one
        min_duration = min(video_clip.duration, audio_clip.duration)
        video_clip = video_clip.subclip(0, min_duration)
        audio_clip = audio_clip.subclip(0, min_duration)
        
        # Set audio to video
        final_clip = video_clip.set_audio(audio_clip)
        
        # Write result
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
    except Exception as e:
        print(f"Failed to combine video and audio: {e}")
        # If combining fails, just copy the video file
        shutil.copy(video_path, output_path)

def get_text_clip_embeds(texts, start_images, text_clip_model):

    with torch.no_grad():
        ## this will return list of seq, c tensors, no padding
        text_embeddings = text_clip_model.get_text_embeddings(texts)
        
        ## start_images is list of c h w, returns b, 257, ...
        if start_images is not None:
            clip_embeddings = text_clip_model.get_clip_visual_embeddings(start_images)
        else:
            clip_embeddings = None 

    return text_embeddings, clip_embeddings

def get_scheduler_time_steps(sampling_steps, sample_solver='unipc', device=0, config=None):

    torch.manual_seed(4)

    if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            sampling_steps, device=device, shift=config.inference.get("shift", 5.0))
        timesteps = sample_scheduler.timesteps

        print(f"Using unipc! See timesteps: {timesteps}! \n")
        print(f"All index: {[sample_scheduler.index_for_timestep(t) for t in timesteps]}! \n")
        print(f"All (sigma_next - sigma): {[sample_scheduler.sigmas[sample_scheduler.index_for_timestep(t) + 1] - sample_scheduler.sigmas[sample_scheduler.index_for_timestep(t)] for t in timesteps]}! \n")

    elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=config.inference.get("shift", 5.0))
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=device,
            sigmas=sampling_sigmas)
        print(f"Using dpm++! See timesteps: {timesteps}! \n")
        
    elif sample_solver == 'euler':
        sample_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=config.inference.get("shift", 5.0)
        )
        # sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        # sigmas = np.array(sigmas)
        timesteps, sampling_steps = retrieve_timesteps(
            sample_scheduler,
            sampling_steps,
            device,
        )
        print(f"Using euler! See timesteps: {timesteps}! \n")
        print(f"All index: {[sample_scheduler.index_for_timestep(t) for t in timesteps]}! \n")
        print(f"All (sigma_next - sigma): {[sample_scheduler.sigmas[sample_scheduler.index_for_timestep(t) + 1] - sample_scheduler.sigmas[sample_scheduler.index_for_timestep(t)] for t in timesteps]}! \n")
    
    else:
        raise NotImplementedError("Unsupported solver.")
    
    return sample_scheduler, timesteps 

@torch.inference_mode()
def inference(config, score_model, vae_model_video, vae_model_audio, text_model, device, sample_solver='unipc', local_rank=0, global_rank=0, world_size=1, root_write_dir=None, target_dtype=torch.float32):
        
    score_model.eval()
    seed_g = config.inference.get("seed", 100)

    cfg_scale_audio = config.inference.get("cfg_scale_audio", 1.0)
    cfg_scale_video = config.inference.get("cfg_scale_video", 1.0)
    sampling_steps = config.inference.num_steps
    t2v_only = config.inference.get("t2v_only", False)

    # Check for CSV or pickle directory
    csv_path = config.inference.get("csv_path", None)
    assert csv_path is not None, "Please provide a valid csv_path in the config for inference."
    
    # Load CSV data
    all_eval_data = []
    delimiter = ","
    if csv_path.lower().endswith(".tsv"):
        delimiter = "\t"
    with open(csv_path, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            all_eval_data.append(row)
    print(f"Loaded {len(all_eval_data)} entries from CSV")

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group  
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups, not individual ranks
    total_files = len(all_eval_data)
    
    if total_files == 0:
        print(f"WARNING: No evaluation files found")
        this_rank_eval_data = []
        start_idx = 0
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if remainder != 0:
            pad_count = num_sp_groups - remainder
            print(f"Padding eval data with {pad_count} duplicates to match {num_sp_groups} SP groups.")
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        files_per_group = len(all_eval_data) // num_sp_groups
        start_idx = sp_group_id * files_per_group
        end_idx = start_idx + files_per_group
        this_rank_eval_data = all_eval_data[start_idx:end_idx]

    # Only first rank in SP group handles I/O
    is_io_rank = (sp_rank == 0)
    
    # Create local directory for temporary storage
    if is_io_rank:
        os.makedirs(root_write_dir, exist_ok=True)
    
    # Timing statistics
    sample_times = []
    with torch.no_grad():
        for idx, data_entry in enumerate(tqdm(this_rank_eval_data, 
                                             desc=f"Group {sp_group_id} generating", 
                                             disable=not is_io_rank)):
            sample_start_time = time.time()
            name_id = f"csv_{start_idx + idx:04d}"
            print(f"[DEBUG - GLOBAL RANK {global_rank}] Processing sample {idx}")
            try:
                inference_scheduler_video, timesteps_video = get_scheduler_time_steps(
                sampling_steps=sampling_steps,
                device=device,
                sample_solver=sample_solver,
                config=config
                )
                inference_scheduler_audio, timesteps_audio = get_scheduler_time_steps(
                sampling_steps=sampling_steps,
                device=device,
                sample_solver=sample_solver,
                config=config
                )

                # CSV mode: create data from CSV row
                csv_row = data_entry
                
                if not t2v_only:
                    # Load first frame from path
                    first_frame = cv2.imread(csv_row['first_frame'])
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    
                    # Create dummy video by repeating first frame (shape: c, f, h, w)
                    num_frames = 121  # Default frame count
                    original_video = np.stack([first_frame.transpose(2, 0, 1)] * num_frames, axis=1)
                    original_video = original_video.astype(np.float32) / 255.0  # Normalize to [0,1]
                    
                # Create silent audio
                sample_rate = 16000
                
                # Extract text data
                text_audio = csv_row['combined']
                
                # IMAGE-TO-VIDEO GENERATION: Use groundtruth first frame as conditioning
                with autocast('cuda', enabled=target_dtype != torch.float32, dtype=target_dtype):
                    # Preprocess audio and video using helper functions
                    print(f"[DEBUG - GLOBAL RANK {global_rank}] About to preprocess audio and video tensors")
                    if not t2v_only:
                        original_video_tensor = preprocess_video_tensor(original_video, device, target_dtype, h_w_multiple_of=32, resize_total_area=config.inference.get("resize_total_area", None))
                        print(f"[DEBUG - GLOBAL RANK {global_rank}] Preprocessed audio and video tensors - video shape: {original_video_tensor.shape}")
                    
                    print(f"[DEBUG - GLOBAL RANK {global_rank}] About to try to get clip and clap embeds")
                    # Get text embeddings for both conditional and unconditional (CFG) with visual conditioning
                    negative_prompt = config.inference.get("negative_prompt", "")
                    text_embeddings = text_model(
                        [text_audio, negative_prompt], 
                        text_model.device
                    )
                    text_embeddings = [emb.to(target_dtype).to(device) for emb in text_embeddings]

                    # Split embeddings
                    text_embeddings_audio_pos = text_embeddings[0]
                    text_embeddings_video_pos = text_embeddings[0] 

                    text_embeddings_audio_neg = text_embeddings[1]
                    text_embeddings_video_neg = text_embeddings[1]

                    if not t2v_only:              
                        with torch.no_grad():
                            latents_images = vae_model_video.wrapped_encode(original_video_tensor[:,:,:1]).bfloat16().squeeze(0) # c 1 h w 
                        latents_images = latents_images.to(target_dtype)
                        video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
                    else:
                        video_latent_h, video_latent_w = config.inference.get("video_latent_height", 32), config.inference.get("video_latent_width", 62)

                    video_noise = torch.randn((1, config.inference.video_latent_channel, config.inference.video_latent_length, video_latent_h, video_latent_w), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed_g)).squeeze(0)  # c, f, h, w
                    audio_noise = torch.randn((1, config.inference.audio_latent_length, config.inference.audio_latent_channel), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed_g)).squeeze(0)  # 1, l c -> l, c
                    
                    # Calculate sequence lengths from actual latents
                    max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
                    _patch_size_h, _patch_size_w = score_model.video_model.patch_size[1], score_model.video_model.patch_size[2]
                    max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
                    
                    # Sampling loop
                    for i, (t_v, t_a) in enumerate(zip(timesteps_video, timesteps_audio)):
                        # print(f"[DEBUG - GLOBAL RANK {global_rank}] Starting sampling at {i} iteration")
                        timestep_input = torch.full((1,), t_v, device=device)
                        #timestep_input_a = torch.full((1,), t_a, device=device)

                        if not t2v_only:
                            video_noise[:, :1] = latents_images

                        # Positive (conditional) forward pass
                        pos_forward_args = {
                            'audio_context': [text_embeddings_audio_pos],
                            'vid_context': [text_embeddings_video_pos],
                            'vid_seq_len': max_seq_len_video,
                            'audio_seq_len': max_seq_len_audio,
                            'first_frame_is_clean': True if not t2v_only else False
                        }

                        pred_vid_pos, pred_audio_pos = score_model(
                            vid=[video_noise],
                            audio=[audio_noise],
                            t=timestep_input,
                            **pos_forward_args
                        )
                        
                        # Negative (unconditional) forward pass  
                        neg_forward_args = {
                            'audio_context': [text_embeddings_audio_neg],
                            'vid_context': [text_embeddings_video_neg],
                            'vid_seq_len': max_seq_len_video,
                            'audio_seq_len': max_seq_len_audio,
                            'first_frame_is_clean': True if not t2v_only else False,
                            'slg_layers': config.inference.get('slg_layers', False)
                        }
                        
                        pred_vid_neg, pred_audio_neg = score_model(
                            vid=[video_noise],
                            audio=[audio_noise],
                            t=timestep_input,
                            **neg_forward_args
                        )

                        # Apply classifier-free guidance
                        pred_video_guided = pred_vid_neg[0] + cfg_scale_video * (pred_vid_pos[0] - pred_vid_neg[0])
                        pred_audio_guided = pred_audio_neg[0] + cfg_scale_audio * (pred_audio_pos[0] - pred_audio_neg[0])

                        # Update noise using scheduler
                        video_noise = inference_scheduler_video.step(
                            pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                        )[0].squeeze(0)

                        audio_noise = inference_scheduler_audio.step(
                            pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                        )[0].squeeze(0)

                    # Decode final latents
                    # Decode audio
                    print(f"[DEBUG - GLOBAL RANK {global_rank}] Decoding final latents")
                    audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                    generated_audio = vae_model_audio.wrapped_decode(audio_latents_for_vae)
                    generated_audio = generated_audio.squeeze().cpu().float().numpy()
                    
                    # Decode video  
                    video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
                    generated_video = vae_model_video.wrapped_decode(video_latents_for_vae)
                    generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w

                # Only I/O rank saves generated files
                if is_io_rank:
                    # Save generated audio
                    print(f"[DEBUG - GLOBAL RANK {global_rank}] Saving generated")
                    sample_rate = 16000  # Define sample_rate here for CSV mode
                    generated_audio_path = os.path.join(root_write_dir, f"{name_id}_gen.wav")
                    wavfile.write(generated_audio_path, sample_rate, (generated_audio * 32767).astype(np.int16))
                    
                    # Save generated video
                    generated_video_path = os.path.join(root_write_dir, f"{name_id}_gen.mp4")
                    _save_video_frames_as_mp4(generated_video.transpose(1, 2, 3, 0), generated_video_path, fps=config.inference.get("save_fps", 24))  # f, h, w, c
                    
                    # Combine video and audio into final MP4
                    if t2v_only:
                        combined_path = os.path.join(root_write_dir, f"{name_id}_combined_t2v.mp4")
                    else:
                        combined_path = os.path.join(root_write_dir, f"{name_id}_combined.mp4")
                    _combine_video_audio_to_mp4(generated_video_path, generated_audio_path, combined_path)
            
                # Calculate total sample time
                total_sample_time = time.time() - sample_start_time
                sample_times.append(total_sample_time)
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(traceback.format_exc())
                continue
    
    # Log overall statistics (all ranks)
    if sample_times:
        avg_sample_time = sum(sample_times) / len(sample_times)
        print(f"[RANK {global_rank}] Overall Statistics:")
        print(f"  - Processed {len(sample_times)} samples successfully")
        print(f"  - Average time per sample: {avg_sample_time:.2f}s")
        print(f"  - Total generation time: {sum(sample_times):.2f}s")

    # Wait for all ranks to finish uploading
    torch.distributed.barrier()
    

def main(config, args): 
    # init dist and cuda settings
    deepspeed.init_distributed(dist_backend='nccl')

    sp_size = config.inference.get("sp_size", 1)
    if sp_size == "world_size" or sp_size == -1:
        sp_size = get_world_size()
    initialize_sequence_parallel_state(sp_size)
    print(f"Using SP: {get_sequence_parallel_state()}")
    device = get_device()
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    root_write_dir = config.inference.local_output_root
    
    torch.cuda.set_device(device) 

    args.local_rank = local_rank
    args.device = device
    
    sample_solver = config.inference.get("sample_solver", 'unipc')
    target_dtype = torch.bfloat16

    score_model_fusion = init_fusion_score_model_ovi(config, rank=local_rank)
    score_model_fusion.to(dtype=target_dtype).to(device=device).eval()
    
    vae_model_video = init_vae(config.video, rank=device)
    vae_model_video.model.requires_grad_(False).eval()
    vae_model_video.model = vae_model_video.model.bfloat16()

    vae_model_audio = init_vae(config.audio, rank=device)
    vae_model_audio.requires_grad_(False).eval()
    vae_model_audio = vae_model_audio.bfloat16()
    
    if config.text.get("t5_on_cpu", False):
        text_model = init_text_model(config, rank=torch.device('cpu'))
    elif config.text.get("text_model_ds_config", None) is not None:
        text_model_ds_config = dict(config.text.get("text_model_ds_config"))
        print(f"using sharded text model with {text_model_ds_config}...")
        text_model = init_text_model(config, rank=device)
        text_model.model, _, _, _ = deepspeed.initialize(
                                    args=args,model=text_model.model, 
                                    config=text_model_ds_config)
    else:
        text_model = init_text_model(config, rank=device)
    
    if hasattr(config.model, 'checkpoint_path') and config.model.checkpoint_path:
    
        checkpoint_path = config.model.checkpoint_path
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module']
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = score_model_fusion.load_state_dict(df, strict=False)

        print(
            f"Loaded ckpt for `entire fusion score model` on rank 0 from"
            f"{checkpoint_path}. Total missing: {missing}, "
            f"unexpected: {unexpected}"
        )

        del df 
        gc.collect()
    else: 
        raise RuntimeError("No valid checkpoint path found in config. Please specify either 'checkpoint_path'")
    

    inference(
        config=config,
        score_model=score_model_fusion,
        vae_model_video=vae_model_video,
        vae_model_audio=vae_model_audio,
        text_model=text_model,
        device=device,
        sample_solver=sample_solver,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        root_write_dir=root_write_dir,
        target_dtype=target_dtype
    )

if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)