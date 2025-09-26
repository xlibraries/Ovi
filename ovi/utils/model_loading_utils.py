import importlib
import torch 
import os
import json
from safetensors.torch import load_file

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.modules.mmaudio.features_utils import FeaturesUtils
    
def init_wan_vae_2_2(config, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_config['vae_pth'] = config.vae.path
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(config, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    vae_config['tod_vae_ckpt'] = config.vae.tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = config.vae.bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_fusion_score_model_ovi(rank: int = 0):
    video_config = "ovi/configs/model/dit/video.json"
    audio_config = "ovi/configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config

def init_text_model(config, rank):
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


def load_fusion_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.exists(checkpoint_path):
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

        missing, unexpected = model.load_state_dict(df, strict=True)

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError("{checkpoint=} does not exists'")