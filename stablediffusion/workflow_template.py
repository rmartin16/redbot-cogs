from typing import Callable, Dict, List, Tuple

# SDXL base+refiner "ensemble of experts" workflow, using Juggernaut XL v9 as the
# base model and stabilityai's official sd_xl_refiner_1.0 for the final detail pass:
# https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
# https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
#
# Node IDs:
#   3  KSamplerAdvanced (base)     4  CheckpointLoaderSimple (base)
#   5  EmptyLatentImage            6  CLIPTextEncode (base positive)
#   7  CLIPTextEncode (base negative)
#   8  VAEDecode                   9  SaveImage
#   11 CheckpointLoaderSimple (refiner)
#   12 KSamplerAdvanced (refiner)
#   13 CLIPTextEncodeSDXLRefiner (refiner positive)
#   14 CLIPTextEncodeSDXLRefiner (refiner negative)
#
# The base sampler runs steps 0..SWITCH, then hands its (still-noisy) latent to the
# refiner sampler for SWITCH..TOTAL — the classic SDXL base+refiner split, not extra
# steps tacked on top, so total step count (and runtime) stays close to base-only.
REFINER_SWITCH_RATIO = 0.8  # fraction of total steps done by the base model

_TOTAL_STEPS = 32
_SWITCH_STEP = round(_TOTAL_STEPS * REFINER_SWITCH_RATIO)

DEFAULT_WORKFLOW: dict = {
    "3": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["4", 0],
            "add_noise": "enable",
            "noise_seed": 0,
            "steps": _TOTAL_STEPS,
            "cfg": 5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
            "start_at_step": 0,
            "end_at_step": _SWITCH_STEP,
            "return_with_leftover_noise": "enable",
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"batch_size": 1, "height": 1024, "width": 1024},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": "masterpiece, best quality"},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": ""},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["12", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
    "11": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_refiner_1.0.safetensors"},
    },
    "12": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["11", 0],
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": _TOTAL_STEPS,
            "cfg": 5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "positive": ["13", 0],
            "negative": ["14", 0],
            "latent_image": ["3", 0],
            "start_at_step": _SWITCH_STEP,
            "end_at_step": _TOTAL_STEPS,
            "return_with_leftover_noise": "disable",
        },
    },
    "13": {
        "class_type": "CLIPTextEncodeSDXLRefiner",
        "inputs": {
            "clip": ["11", 1],
            "ascore": 6.0,
            "width": 1024,
            "height": 1024,
            "text": "masterpiece, best quality",
        },
    },
    "14": {
        "class_type": "CLIPTextEncodeSDXLRefiner",
        "inputs": {
            "clip": ["11", 1],
            "ascore": 2.5,
            "width": 1024,
            "height": 1024,
            "text": "",
        },
    },
}

# Juggernaut's own docs: heavy default negatives often hurt more than they help,
# so start empty and let users opt in via a negative_prompt: override.
DEFAULT_NEGATIVE_PROMPT = ""

# Maps a "key:value" prompt-override token (A1111-style inline config) to every
# place it needs to land in the node graph (base+refiner pairs, where applicable):
# list of (node_id, input_key, type_caster).
NODE_CONFIG_MAP: Dict[str, List[Tuple[str, str, Callable]]] = {
    "cfg_scale": [("3", "cfg", float), ("12", "cfg", float)],
    "cfg": [("3", "cfg", float), ("12", "cfg", float)],
    "seed": [("3", "noise_seed", int), ("12", "noise_seed", int)],
    "sampler_name": [("3", "sampler_name", str), ("12", "sampler_name", str)],
    "sampler": [("3", "sampler_name", str), ("12", "sampler_name", str)],
    "scheduler": [("3", "scheduler", str), ("12", "scheduler", str)],
    "width": [("5", "width", int), ("13", "width", int), ("14", "width", int)],
    "height": [("5", "height", int), ("13", "height", int), ("14", "height", int)],
    "negative_prompt": [("7", "text", str), ("14", "text", str)],
}
