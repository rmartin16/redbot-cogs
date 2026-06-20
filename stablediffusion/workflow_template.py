from typing import Callable, Dict, Tuple

# SDXL API-format workflow. Node IDs match /home/russell/comfy/sdxl_api_workflow.json:
# 3 KSampler, 4 CheckpointLoaderSimple, 5 EmptyLatentImage, 6/7 CLIPTextEncode pos/neg,
# 8 VAEDecode, 9 SaveImage, 10 VAELoader.
DEFAULT_WORKFLOW: dict = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "seed": 0,
            "steps": 25,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
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
        "inputs": {"clip": ["4", 1], "text": "worst quality, low quality, blurry"},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["10", 0]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
    "10": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "sdxl_vae.safetensors"},
    },
}

DEFAULT_NEGATIVE_PROMPT = "worst quality, low quality, blurry"
DEFAULT_STEPS = 25

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
