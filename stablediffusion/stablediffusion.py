import copy
import io
import os
import uuid
from itertools import islice
from random import choices, randint
from time import time
from typing import Dict, Generator, List, Tuple, TypeVar

import aiohttp
import discord

from discord.http import Route
from redbot.core import commands

from .comfyui_api import ComfyUIClient, GenerationCancelled, GenerationFailure
from .workflow_template import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_WORKFLOW,
    NODE_CONFIG_MAP,
    REFINER_SWITCH_RATIO,
)

T = TypeVar("T")
U = TypeVar("U")

COMFYUI_POST_ENDPOINT = os.environ.get("COMFYUI_POST_ENDPOINT", "http://10.16.8.6:8188")

FULL_WORD_LIST_FILE = os.environ.get("STABLEDIFFUSION_FULL_WORD_LIST", "/data/words_full")
COMMON_WORD_LIST_FILE = os.environ.get("STABLEDIFFUSION_COMMOM_WORD_LIST", "/data/words_common")
try:
    with open(FULL_WORD_LIST_FILE) as f:
        WORDS_FULL = list(set(f.read().splitlines()))
except Exception:
    WORDS_FULL = []
try:
    with open(COMMON_WORD_LIST_FILE) as f:
        WORDS_COMMON = list(set(f.read().splitlines()))
except Exception:
    WORDS_COMMON = []


def chunks(data: Dict[T, U], chunk_size: int) -> Generator[Dict[T, U], None, None]:
    """Iteratively return chunks of a dictionary"""
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k: data[k] for k in islice(it, chunk_size)}


def build_workflow(prompt: str, negative_prompt: str, num_of_images: int, overrides: dict) -> Tuple[dict, List[str]]:
    """Build a ComfyUI API-format workflow from a prompt and A1111-style key:value overrides.

    Unlike A1111's flat JSON payload, ComfyUI's node graph has no free-form namespace to
    dump arbitrary keys into, so unknown/bad-typed overrides are dropped with a warning
    instead of failing the whole generation.
    """
    workflow = copy.deepcopy(DEFAULT_WORKFLOW)
    # Prompt/negative text needs encoding for both the base (6/7) and refiner (13/14)
    # stages, since the refiner has its own CLIP text encoder.
    workflow["6"]["inputs"]["text"] = prompt
    workflow["13"]["inputs"]["text"] = prompt
    negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["14"]["inputs"]["text"] = negative_prompt
    workflow["5"]["inputs"]["batch_size"] = num_of_images
    # ComfyUI has no A1111-style "-1 means randomize" sentinel; roll our own seed,
    # shared by both samplers so the refiner continues the same noise.
    seed = randint(0, 2**32 - 1)
    workflow["3"]["inputs"]["noise_seed"] = seed
    workflow["12"]["inputs"]["noise_seed"] = seed

    overrides = dict(overrides)
    warnings = []

    # "steps" is special: it's the *total* step budget split between base and
    # refiner, so it has to recompute the base/refiner handoff point rather than
    # just being copied onto a single node input.
    raw_steps = overrides.pop("steps", None)
    if raw_steps is not None:
        try:
            total_steps = int(raw_steps)
            switch_step = round(total_steps * REFINER_SWITCH_RATIO)
            workflow["3"]["inputs"]["steps"] = total_steps
            workflow["3"]["inputs"]["end_at_step"] = switch_step
            workflow["12"]["inputs"]["steps"] = total_steps
            workflow["12"]["inputs"]["start_at_step"] = switch_step
            workflow["12"]["inputs"]["end_at_step"] = total_steps
        except (ValueError, TypeError):
            warnings.append(f"ignored bad override: `steps:{raw_steps}`")

    for key, value in overrides.items():
        mapping = NODE_CONFIG_MAP.get(key)
        if mapping is None:
            warnings.append(f"unknown config key ignored: `{key}`")
            continue
        for node_id, input_key, caster in mapping:
            try:
                workflow[node_id]["inputs"][input_key] = caster(value)
            except (ValueError, TypeError):
                warnings.append(f"ignored bad override: `{key}:{value}`")
                break

    return workflow, warnings


class ProgressBar:
    def __init__(self, total: int):
        self.bar_width = 40
        self.completed_char = "#"
        self.remaining_char = "."

        self.current = 0
        self.total = total

    def update(self, completed: int, seconds_remaining: float = 0):
        self.current = completed

        completed_count = int(self.bar_width * completed / self.total)
        bar_completed = self.completed_char * completed_count
        bar_remaining = self.remaining_char * (self.bar_width - completed_count)
        percent_done = int(completed_count * (100 / self.bar_width))

        eta = "" if seconds_remaining < 0.1 else f"ETA: {round(seconds_remaining)}s"

        return f"`{bar_completed}{bar_remaining} {percent_done}% {eta}`"


class Image:
    def __init__(self, config: dict, seed: int, image: discord.File):
        self.config = config
        self.seed = seed
        self.image = image


class StatusMessage:
    def __init__(self, ctx, bot_user_id):
        self.ctx = ctx
        self.msg = None
        self.bot_user_id = bot_user_id

    async def create(self, content="Generating images..."):
        self.msg = await self.ctx.send(content)

    async def update(self, content):
        await self.msg.edit(content=content)

    async def add_cancel_reaction(self):
        """Add ❌ reaction to message so users can cancel image generation."""
        await self.msg.add_reaction("❌")


class StableDiffusion(commands.Cog):
    """Stable Diffusion image generation."""

    def __init__(self, bot):
        self.bot = bot
        self.api = ComfyUIClient(base_url=COMFYUI_POST_ENDPOINT)
        self.status_msg: StatusMessage = None
        self.ctx = None
        self.channels = {}

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    @commands.command(name="stablediffusionrandom")
    @commands.guild_only()
    async def generate_random(
            self,
            ctx: commands.Context,
            num_of_words: str = "4",
            word_list: str = "FULL",
            num_of_images: str = "1",
    ):
        if not WORDS_FULL and not WORDS_COMMON:
            return await ctx.send("Failed to load word list...")
        try:
            num_of_words = int(num_of_words)
        except ValueError:
            num_of_words = 4
        WORDS = WORDS_FULL if word_list.upper() == "FULL" else WORDS_COMMON
        prompt = " ".join(choices(WORDS, k=num_of_words))
        await ctx.send(f"Here's the best i can do with `{prompt}` from {word_list.lower()} word list...")
        await self.generate(ctx, prompt=f"{prompt} {num_of_images}")

    @commands.max_concurrency(1, commands.BucketType.default)
    @commands.command(name="stablediffusion")
    @commands.guild_only()
    async def generate(self, ctx: commands.Context, *, prompt: str):
        """Generate images through Stable Diffusion."""
        embed_links = ctx.channel.permissions_for(ctx.guild.me).embed_links
        if not embed_links:
            return await ctx.send(
                "I need the `Embed Links` permission here before you can use this command."
            )

        self.ctx = ctx
        self.status_msg = StatusMessage(ctx=ctx, bot_user_id=self.bot.user.id)

        async with ctx.typing():
            try:
                await self.status_msg.create()
                workflow, prompt, warnings = self.request_config(prompt)
                if warnings:
                    await ctx.send("\n".join(warnings))
                start = time()
                images = await self.generate_images(workflow)
                gen_time = time() - start
                await self.upload(images, prompt, gen_time)
            except Exception as e:
                import traceback
                await self.ctx.send(
                    "Something went wrong... :(\n"
                    f"`{e}`\n"
                    f"```\n"
                    f"{traceback.format_exc()}\n"
                    f"```")
            finally:
                await self.status_msg.msg.delete()

    async def upload(self, images, prompt, gen_time):
        """Send images to Discord."""
        await self.status_msg.update(content="Uploading to discord...")
        embed = discord.Embed(
            colour=await self.ctx.embed_color(),
            title="Stable Diffusion results",
            url="https://huggingface.co/spaces/stabilityai/stable-diffusion"
        )
        for files_images_chunk in chunks(images, chunk_size=4):
            seeds = " ".join(f"{idx}: {img.seed}" for idx, img in enumerate(files_images_chunk.values()))
            footer = f"{prompt} by {self.ctx.author} in {round(gen_time, 1)}s\n{seeds}"
            embeds = [
                embed.copy().set_image(url=f"attachment://{name}").set_footer(text=footer)
                for name, image in files_images_chunk.items()
            ]

            form = []
            payload = {"embeds": [e.to_dict() for e in embeds]}
            form.append({"name": "payload_json", "value": discord.utils._to_json(payload)})

            for name, image in files_images_chunk.items():
                form.append(
                    {
                        "name": name,
                        "value": image.image.fp,
                        "filename": image.image.filename,
                        "content_type": "application/octet-stream",
                    }
                )

            try:
                await self.ctx.guild._state.http.request(
                    Route("POST", "/channels/{channel_id}/messages", channel_id=self.ctx.channel.id),
                    form=form,
                    files=(f.image for f in files_images_chunk.values())
                )
            except discord.errors.DiscordServerError as e:
                await self.ctx.send(f"Discord is sucking... >:( {e}")

    def request_config(self, prompt) -> Tuple[dict, str, List[str]]:
        prompt, num_of_images = self.extract_count_from_prompt(prompt)
        prompt, prompt_config = self.parse_config_from_prompt(prompt)

        workflow, warnings = build_workflow(
            prompt=prompt,
            negative_prompt="",
            num_of_images=num_of_images,
            overrides=prompt_config,
        )

        return workflow, prompt, warnings

    def extract_count_from_prompt(self, prompt) -> Tuple[str, int]:
        # HACK: Support returning arbitrary number of images
        num_of_images = prompt.split(" ")[-1:]
        try:
            num_of_images = int(num_of_images[0].strip())
            prompt = prompt.strip().rstrip(str(num_of_images))
        except ValueError:
            num_of_images = 1
        return prompt, min(num_of_images, 8)

    def parse_config_from_prompt(self, prompt) -> Tuple[str, Dict]:
        new_prompt = []
        prompt_config = {}
        for piece in prompt.strip().split(" "):
            if ":" in piece:
                prompt_config[piece.split(":")[0]] = piece.split(":")[1]
            else:
                new_prompt.append(piece)

        return " ".join(new_prompt), prompt_config

    async def generate_images(self, workflow: dict) -> Dict[str, Image]:
        """Request and retrieve generated images."""
        images = {}
        progress_bar = ProgressBar(total=100)
        await self.status_msg.update(content=progress_bar.update(0))
        await self.status_msg.add_cancel_reaction()

        client_id = str(uuid.uuid4())
        prompt_id = str(uuid.uuid4())
        self.channels[str(self.ctx.channel.id)] = {"msg_id": self.status_msg.msg.id, "prompt_id": prompt_id}

        async def on_progress(value, max_steps):
            pct = round(100 * value / max_steps) if max_steps else 0
            if pct == 100 or pct >= progress_bar.current + 5:
                await self.status_msg.update(content=progress_bar.update(pct))

        try:
            history_entry = await self.api.run_workflow(
                workflow, prompt_id=prompt_id, client_id=client_id, on_progress=on_progress
            )
            await self.status_msg.update(content=progress_bar.update(100))

            base_seed = workflow["3"]["inputs"]["noise_seed"]
            image_bytes_list = await self.api.fetch_output_images(history_entry)
            for num, img_bytes in enumerate(image_bytes_list):
                name = f"{num}.png"
                images[name] = Image(
                    image=discord.File(io.BytesIO(img_bytes), name),
                    seed=base_seed,
                    config=workflow,
                )
        except GenerationCancelled:
            raise GenerationFailure("Generation was cancelled.")
        except aiohttp.ClientResponseError as e:
            raise GenerationFailure(f"Bad HTTP response... [{e}]")
        except aiohttp.ClientConnectionError as e:
            raise GenerationFailure(f"ComfyUI backend is probably down [{e}]")
        except GenerationFailure:
            raise
        except Exception as e:
            raise GenerationFailure(f"Unknown error: {repr(e)}")
        finally:
            self.channels.pop(str(self.ctx.channel.id), None)

        return images

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        channel_key = str(reaction.message.channel.id)
        if channel_key not in self.channels:
            return
        if self.channels[channel_key]["msg_id"] != reaction.message.id:
            return
        if user.id == self.bot.user.id:
            return
        if str(reaction.emoji) != "❌":
            return

        prompt_id = self.channels[channel_key].get("prompt_id")
        if prompt_id:
            await self.api.interrupt(prompt_id=prompt_id)
