import io
import json
import os
from itertools import islice
from random import choices
from time import time
from typing import Dict, Generator, Tuple, TypeVar

import aiohttp
import discord

import webuiapi
from PIL.PngImagePlugin import PngImageFile
from discord.http import Route
from redbot.core import commands

T = TypeVar("T")
U = TypeVar("U")

STABLEDIFFUSION_POST_ENDPOINT = os.environ.get("STABLEDIFFUSION_POST_ENDPOINT")
DEFAULT_REQUEST_STEPS = 50
CONFIG_PROPERTIES = {
    'iterations', 'steps', 'cfg_scale', 'sampler_name', 'width', 'height', 'seed', 'initimg',
    'strength', 'fit', 'gfpgan_strength', 'upscale_level', 'upscale_strength'
}

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


class ProgressBar:
    def __init__(self, total: int):
        self.bar_width = 30
        self.completed_char = "#"
        self.remaining_char = "."

        self.total = total

    def update(self, completed: int):
        completed_count = int(self.bar_width * completed / self.total)
        bar_completed = self.completed_char * completed_count
        bar_remaining = self.remaining_char * (self.bar_width - completed_count)
        percent_done = int(completed_count * (100 / self.bar_width))
        return f"`{bar_completed}{bar_remaining} {percent_done}%`"


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


class GenerationFailure(Exception):
    """Image generation failed."""


class StableDiffusion(commands.Cog):
    """Stable Diffusion image generation."""

    def __init__(self, bot):
        self.bot = bot
        self.api = webuiapi.WebUIApi(host="10.16.8.5")
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

    @commands.max_concurrency(2, commands.BucketType.default)
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
                request_config = self.request_config(prompt)
                start = time()
                images = await self.generate_images(request_config)
                gen_time = time() - start
                await self.upload(images, request_config['prompt'], gen_time)
            except Exception as e:
                import traceback
                await self.ctx.send(f"Something went wrong... :( [{e}]\n{traceback.format_exc()}")
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

    def request_config(self, prompt) -> Dict:

        prompt, num_of_images = self.extract_count_from_prompt(prompt)
        prompt, prompt_config = self.parse_config_from_prompt(prompt)

        request_config = {
            "enable_hr": False,
            "denoising_strength": 0,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": "",
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            # "hr_sampler_name": "",
            # "hr_prompt": "",
            # "hr_negative_prompt": "",
            "prompt": prompt,
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": "",
            "batch_size": num_of_images,
            "n_iter": 1,
            "steps": DEFAULT_REQUEST_STEPS,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "restore_faces": True,
            "tiling": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "negative_prompt": "",
            "eta": 0,
            # "s_min_uncond": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True,
            "script_args": [],
            "sampler_index": "Euler",
            "script_name": "",
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {}
        }

        request_config.update(prompt_config)

        return request_config

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
            if ":" in piece:  # and piece.split(":")[0] in CONFIG_PROPERTIES:
                prompt_config[piece.split(":")[0]] = piece.split(":")[1]
            else:
                new_prompt.append(piece)
        return " ".join(new_prompt), prompt_config

    async def generate_images(self, request_config: dict) -> Dict[str, Image]:
        """Request and retrieve generated images."""
        images = {}
        current_step = 0
        step_update_size = 8  # step frequency to update status message
        progress_bar = ProgressBar(total=100)
        await self.status_msg.update(content=progress_bar.update(current_step))
        await self.status_msg.add_cancel_reaction()
        self.channels[str(self.ctx.channel.id)] = {"msg_id": self.status_msg.msg.id}
        try:

            response = await self.api.txt2img(use_async=True, **request_config)

            for num, image in enumerate(response.images):
                image: PngImageFile
                num = str(num)

                image_bytes = io.BytesIO()
                image.save(fp=image_bytes)

                images[num] = Image(
                    image=discord.File(image_bytes, f"{num}.png"),
                    seed=response.parameters["seed"],
                    config=response.parameters,
                )

            # async with aiohttp.ClientSession() as session:
            #     async with session.post(STABLEDIFFUSION_POST_ENDPOINT, json=request_config) as response:
            #         response.raise_for_status()
            #
            #         results = []
            #         async for line in response.content:
            #             resp = json.loads(line)
            #             event = resp.get("event", "").lower()
            #
            #             if event.startswith("upscaling"):
            #                 await self.status_msg.update(content="Upscaling images...")
            #
            #             elif event == "result":
            #                 results.append(resp)
            #
            #             elif event == "step":
            #                 current_step += 1
            #                 if current_step == progress_bar.total or current_step % step_update_size == 0:
            #                     await self.status_msg.update(content=progress_bar.update(current_step))
            #
            #         for result in results:
            #             async with session.get(STABLEDIFFUSION_POST_ENDPOINT + "/" + result["url"]) as image:
            #                 name = result["url"].split("/")[-1]
            #                 images[name] = Image(
            #                     image=discord.File(io.BytesIO(await image.content.read()), name),
            #                     seed=result["seed"],
            #                     config=result["config"],
            #                 )

        except json.decoder.JSONDecodeError as e:
            raise GenerationFailure(f"This isn't JSON... [{e}]")
        except aiohttp.ClientResponseError as e:
            raise GenerationFailure(f"Bad HTTP response... [{e}]")
        except aiohttp.ClientConnectionError as e:
            raise GenerationFailure(f"Stable Diffusion backend is probably down [{e}]")
        except Exception as e:
            raise GenerationFailure(f"Unknown error: {repr(e)}")

        return images

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        if str(reaction.message.channel.id) not in self.channels:
            return
        if self.channels[str(reaction.message.channel.id)]["msg_id"] != reaction.message.id:
            return
        if user.id == self.bot.user.id:
            return

        async with aiohttp.ClientSession() as session:
            await session.get(f"{STABLEDIFFUSION_POST_ENDPOINT}/cancel")
