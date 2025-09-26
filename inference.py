import os
import sys
import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_fusion_engine import OviFusionEngine



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.inference.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # validate inputs before loading model to not waste time if input is not valid
    text_prompt = config.inference.get("text_prompt")
    image_path = config.inference.get("image_path", None)
    text_prompts, image_paths = validate_and_process_user_prompt(text_prompt, image_path)
    if config.inference.get("t2v_only", False):
        image_paths = [None] * len(text_prompts)

    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("OVI Fusion Engine loaded!!")

    shard_text_model = config.inference.get("shard_text_model", False)
    shard_fusion_model = config.inference.get("shard_fusion_model", False)

    #TODO: add sharding...
    if shard_text_model or shard_fusion_model:
        raise NotImplementedError("Sharding for text and fusion model is not implemented yet.")

    require_sample_padding = shard_text_model or shard_fusion_model

    
    output_dir = config.inference.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths))

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
        start_idx = 0
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if remainder != 0 and require_sample_padding:
            pad_count = num_sp_groups - remainder
            logging.info(f"Padding eval data with {pad_count} duplicates to match {num_sp_groups} SP groups.")
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        files_per_group = len(all_eval_data) // num_sp_groups
        start_idx = sp_group_id * files_per_group
        end_idx = start_idx + files_per_group
        this_rank_eval_data = all_eval_data[start_idx:end_idx]

    for idx, (text_prompt, image_path) in tqdm(enumerate(this_rank_eval_data)):
        output_path = os.path.join(output_dir, f"{idx + start_idx}_generated.mp4")
        generated_video, generated_audio = ovi_engine.generate(text_prompt=text_prompt,
                                                                image_path=image_path,
                                                                aspect_ratio=config.inference.get("aspect_ratio", "9:16"),
                                                                seed=config.inference.get("seed", 100),
                                                                solver_name=config.inference.get("solver_name", "unipc"),
                                                                sample_steps=config.inference.get("sample_steps", 50),
                                                                shift=config.inference.get("shift", 5.0),
                                                                video_guidance_scale=config.inference.get("video_guidance_scale", 5.0),
                                                                audio_guidance_scale=config.inference.get("audio_guidance_scale", 4.0),
                                                                slg_layer=config.inference.get("slg_layer", 9),
                                                                video_negative_prompt=config.inference.get("video_negative_prompt", ""),
                                                                audio_negative_prompt=config.inference.get("audio_negative_prompt", ""))
        
        save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
    


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)