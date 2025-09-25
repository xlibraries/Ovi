import importlib
import torch 
import os, json
import argparse
from sys import argv

import deepspeed

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        type=str,
                        required=True)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    return parser


def get_arguments(args=argv[1:]):
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args)

    # no cuda mode is not supported
    args.no_cuda = False

    return args

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def init_vae(config, rank: int = 0):
    if 'mmaudio' in config.vae.target:
        return init_mmaudio_vae(config=config, rank=rank)
    elif "Wan2_2_VAE" in config.vae.target:
        return init_wan_vae_2_2(config=config, rank=rank)
    
def init_wan_vae_2_2(config, rank=0):
    vae_model = get_obj_from_str(config.vae.target)

    vae_config = {}
    vae_config['device'] = rank
    vae_config['vae_pth'] = config.vae.path
    vae_model = vae_model(**vae_config)

    return vae_model

def init_mmaudio_vae(config, rank=0):
    vae_model = get_obj_from_str(config.vae.target)
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    vae_config['tod_vae_ckpt'] = config.vae.tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = config.vae.bigvgan_vocoder_ckpt

    vae = vae_model(**vae_config
    ).to(rank)

    return vae


def init_fusion_score_model_wan(config, rank: int = 0):
    fusion_model = get_obj_from_str(config.model.target)

    assert config.model.get("video_config", False) and os.path.exists(
        config.model.video_config
    ), f"model.video_config must exist and be a valid path! {config.model.get('video_config', None)}"

    assert config.model.get("audio_config", False) and os.path.exists(
        config.model.audio_config
    ), f"model.audio_config must exist and be a valid path! {config.model.get('audio_config', None)}"

    with open(config.model.video_config) as f:
        video_config = json.load(f)

    with open(config.model.audio_config) as f:
        audio_config = json.load(f)

    fusion_model = fusion_model(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    print(
        f"Score model (Fusion) all parameters:{params_all}"
    )

    return fusion_model

def init_text_model(config, rank):
    from wan.modules.t5 import T5EncoderModel

    models_path = config.text.models_path
    text_encoder_path = os.path.join(models_path, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(models_path, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        shard_fn=None)


    return text_encoder


