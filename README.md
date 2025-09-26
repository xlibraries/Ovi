# Ovi
---

## 🚀 Features
- Feature 1
- Feature 2
- Feature 3

---

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/character-ai/Ovi.git

# Navigate into the project directory
cd Ovi

# Install dependencies
virtualenv ovi-env
source ovi-env/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
```

## Download Weights
...

## Run Examples
Inference parameters are controlled via a yaml file, for example `ovi/configs/inference/inference_fusion.yaml`
```
inference:
  output_dir: ./engine_out
  num_steps: 50
  solver_name: unipc
  shift: 5.0
  audio_guidance_scale: 3.0
  video_guidance_scale: 4.0
  shard_text_model: False
  shard_fusion_model: False
  video_negative_prompt: "excessive head movement, unnatural motion, jitter, shaky camera, unstable video, deformed hands, extra fingers, fused fingers, missing fingers, six fingers, mutated hands, malformed hands, distorted hands, unnatural hand pose, bad anatomy, text captions, closed captions, low quality, blurry, pixelated, artifacts, distorted, noisy, poor resolution"
  audio_negative_prompt: "robotic, muffled, distorted, metallic, background noise, echo, clipping, low-quality microphone, mouth clicks"
  seed: 100
  aspect_ratio: "9:16" #["9:16", "16:9", "1:1"]
  text_prompt: "A red-haired woman with fair skin, wearing a maroon cardigan over a tan top, smiles and looks towards someone off-screen to her left. She sits at a dark wooden table covered with numerous colorful puzzle pieces, and her hands are sorting through them. In the blurry foreground to the left, the back of a person's head and shoulder are visible. The background features light-colored louvered shutters. The woman continues to smile as she asks, <S>Was that Scott, your manager?<E> An unseen man responds, <S>Yeah.<E> As the woman maintains her smile, continuing to sort puzzle pieces, the man adds, <S>He just hired Mrs. Appletree to be his new secre-<E> The woman glances down at the puzzle pieces briefly before looking back up with a slight widening of her smile.. <AUDCAP>Soft chatter, clinking sounds of puzzle pieces, distinct male voice, distinct female voice.<ENDAUDCAP>"
  # text_prompt: /home/chetwinlow/Ovi/converted_prompts.tsv
  t2v_only: True
  slg_layer: 9
```
fill in description of each parameter...

### a. single process (for a single GPU, text_prompt can be a single string input or a csv file containing prompts and images)
```
python3 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```

### b. multiprocess (for multiple GPUs, where we can run samples in parallel)
```
torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```




