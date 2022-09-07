import io
import json
import os
from itertools import islice
from random import choices
from time import time
from typing import Dict, Generator, Tuple, TypeVar, Union

import aiohttp
import discord
from discord.http import Route
from redbot.core import commands

T = TypeVar("T")
U = TypeVar("U")

STABLEDIFFUSION_POST_ENDPOINT = os.environ.get("STABLEDIFFUSION_POST_ENDPOINT")
DEFAULT_REQUEST_STEPS = 30
CONFIG_PROPERTIES = {
    'iterations', 'steps', 'cfgscale', 'sampler', 'width', 'height', 'seed', 'initimg',
    'strength', 'fit', 'gfpgan_strength', 'upscale_level', 'upscale_strength'
}
GENERATE_RESP_T = Union[Dict[str, Dict[str, Union[str, Union[Dict[str, str], io.BytesIO]]]], int, str]

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

    async def validate_cancel_reaction(self, reaction, user):
        """If someone clicks cancel reaction, return True."""
        if (
                user.id != self.bot_user_id
                and self.msg.id == reaction.message.id
                and reaction.emoji == "❌"
        ):
            return True
        return False


class GenerationFailure(Exception): pass


class StableDiffusion(commands.Cog):
    """Stable Diffusion image generation"""

    def __init__(self, bot):
        self.bot = bot
        self.status_msg: StatusMessage = None
        self.ctx = None

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
        await self.status_msg.create()
        request_config = self.request_config(prompt)

        try:
            async with ctx.typing():
                start = time()
                images = await self.generate_images(request_config)
                gen_time = time() - start

            await self.status_msg.update(content="Uploading to discord...")

            for files_images_chunk in chunks(images, chunk_size=4):
                embed = discord.Embed(
                    colour=await ctx.embed_color(),
                    title="Stable Diffusion results",
                    url="https://huggingface.co/spaces/stabilityai/stable-diffusion"
                )
                embeds = []
                seeds = " ".join(f"{idx}: {img.seed}" for idx, img in enumerate(files_images_chunk.values()))
                for name, image in files_images_chunk.items():
                    em = embed.copy()
                    em.set_image(url=f"attachment://{name}")
                    em.set_footer(
                        text=f"{image.config['prompt']} by {ctx.author} in {round(gen_time, 1)}s\n{seeds}"
                    )
                    embeds.append(em)

                form = []
                payload = {"embeds": [e.to_dict() for e in embeds]}
                form.append({"name": "payload_json", "value": discord.utils.to_json(payload)})

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
                    await ctx.guild._state.http.request(
                        Route("POST", "/channels/{channel_id}/messages", channel_id=ctx.channel.id),
                        form=form,
                        files=(f.image for f in files_images_chunk.values())
                    )
                except discord.errors.DiscordServerError as e:
                    await ctx.send(f"Discord is sucking... >:( {e}")

        except GenerationFailure as e:
            await self.ctx.send(f"Something went wrong... :( [{e}]")
        finally:
            await self.status_msg.msg.delete()

    def request_config(self, prompt) -> Dict:

        prompt, num_of_images = self.extract_count_from_prompt(prompt)
        prompt, prompt_config = self.parse_config_from_prompt(prompt)

        request_config = {
            "prompt": prompt,
            "iterations": str(num_of_images),
            "steps": str(DEFAULT_REQUEST_STEPS),
            "cfgscale": "7.5",
            "sampler": "k_lms",
            "width": "512",
            "height": "512",
            "seed": "-1",
            "initimg": None,
            "strength": "1",
            "fit": "on",
            "gfpgan_strength": "0.8",
            "upscale_level": "2",
            "upscale_strength": "0.75",
        }
        request_config.update(prompt_config)

        return request_config

    def extract_count_from_prompt(self, prompt) -> Tuple[str, int]:
        # HACK: Support returning arbitrary number of images
        num_of_images = prompt.split(" ")[-1:]
        try:
            num_of_images = min(int(num_of_images[0].strip()), 8)
            prompt = prompt.rstrip(str(num_of_images)).strip()
        except ValueError:
            num_of_images = 1
        return prompt, num_of_images

    def parse_config_from_prompt(self, prompt) -> Tuple[str, Dict]:
        new_prompt = []
        prompt_config = {}
        for piece in prompt.strip().split(" "):
            if ":" in piece and piece.split(":")[0] in CONFIG_PROPERTIES:
                prompt_config[piece.split(":")[0]] = piece.split(":")[1]
            else:
                new_prompt.append(piece)
        return " ".join(new_prompt), prompt_config

    async def generate_images(self, request_config: dict) -> Dict[str, Image]:
        """Request and retrieve generated images."""
        images = {}
        current_step = 0
        step_update_size = 8  # step frequency to update status message
        progress_bar = ProgressBar(total=int(request_config["iterations"]) * int(request_config["steps"]))
        await self.status_msg.update(content=progress_bar.update(current_step))
        await self.status_msg.add_cancel_reaction()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(STABLEDIFFUSION_POST_ENDPOINT, json=request_config) as response:
                    response.raise_for_status()

                    async for line in response.content:
                        resp = json.loads(line)
                        event = resp.get("event", "").lower()

                        if event.startswith("upscaling"):
                            await self.status_msg.update(content="Upscaling images...")

                        elif event == "result":
                            async with session.get(STABLEDIFFUSION_POST_ENDPOINT + resp["url"][1:]) as image:
                                name = resp["url"].split("/")[-1]
                                images[name] = Image(
                                    image=discord.File(io.BytesIO(await image.content.read()), name),
                                    seed=resp["seed"],
                                    config=resp["config"],
                                )

                        elif event == "step":
                            current_step += 1
                            if current_step == progress_bar.total or current_step % step_update_size == 0:
                                await self.status_msg.update(content=progress_bar.update(current_step))

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
        if self.status_msg.validate_cancel_reaction(reaction, user):
            async with aiohttp.ClientSession() as session:
                await session.get(f"{STABLEDIFFUSION_POST_ENDPOINT}/cancel")
