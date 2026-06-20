from typing import Callable, Dict, Tuple

# SDXL API-format workflow, using Juggernaut XL v9 (RunDiffusionPhoto_v2) in place of
# vanilla sd_xl_base_1.0 for noticeably better general-purpose output quality.
# Settings (sampler/scheduler/cfg/steps) follow that checkpoint's documented
# recommendations: https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
# Node IDs: 3 KSampler, 4 CheckpointLoaderSimple, 5 EmptyLatentImage,
# 6/7 CLIPTextEncode pos/neg, 8 VAEDecode, 9 SaveImage.
# No separate VAELoader: Juggernaut XL has its VAE baked in (per its docs), so
# VAEDecode pulls straight from the checkpoint loader's VAE output.
DEFAULT_WORKFLOW: dict = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 5,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "seed": 0,
            "steps": 32,
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
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
}

# Juggernaut's own docs: heavy default negatives often hurt more than they help,
# so start empty and let users opt in via a negative_prompt: override.
DEFAULT_NEGATIVE_PROMPT = ""

# Maps a "key:value" prompt-override token (A1111-style inline config) to where it
# lands in the node graph: (node_id, input_key, type_caster).
NODE_CONFIG_MAP: Dict[str, Tuple[str, str, Callable]] = {
    "steps": ("3", "steps", int),
    "cfg_scale": ("3", "cfg", float),
    "cfg": ("3", "cfg", float),
    "seed": ("3", "seed", int),
    "sampler_name": ("3", "sampler_name", str),
    "sampler": ("3", "sampler_name", str),
    "scheduler": ("3", "scheduler", str),
    "denoise": ("3", "denoise", float),
    "width": ("5", "width", int),
    "height": ("5", "height", int),
    "negative_prompt": ("7", "text", str),
}
