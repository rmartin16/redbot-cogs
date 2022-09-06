import base64
import io
import json
import os
from itertools import islice
from random import choices
from time import time
from typing import List, Union

import aiohttp
import discord
from discord.http import Route
from redbot.core import commands

STABLEDIFFUSION_POST_ENDPOINT = os.environ.get("STABLEDIFFUSION_POST_ENDPOINT")
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


def chunks(data, chunk_size):
    """Iteratively return chunks of a dictionary"""
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k: data[k] for k in islice(it, chunk_size)}


class StableDiffusion(commands.Cog):
    """Stable Diffusion image generation"""

    def __init__(self, bot):
        self.bot = bot

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    # @commands.command()
    # @commands.guild_only()
    # async def generate_random(
    #         self,
    #         ctx: commands.Context,
    #         num_of_words: str = "4",
    #         word_list: str = "FULL",
    #         num_of_images: str = "1",
    # ):
    #     if not WORDS_FULL and not WORDS_COMMON:
    #         return await ctx.send("Failed to load word list...")
    #     try:
    #         num_of_words = int(num_of_words)
    #     except ValueError:
    #         num_of_words = 4
    #     WORDS = WORDS_FULL if word_list.upper() == "FULL" else WORDS_COMMON
    #     prompt = " ".join(choices(WORDS, k=num_of_words))
    #     await ctx.send(f"Here's the best i can do with `{prompt}` from {word_list.lower()} word list...")
    #     await self.generate(ctx, prompt=f"{prompt} {num_of_images}")

    @commands.max_concurrency(3, commands.BucketType.default)
    @commands.command(name="stablediffusion")
    @commands.guild_only()
    async def generate(self, ctx: commands.Context, *, prompt: str):
        """Generate images through Stable Diffusion."""
        embed_links = ctx.channel.permissions_for(ctx.guild.me).embed_links
        if not embed_links:
            return await ctx.send(
                "I need the `Embed Links` permission here before you can use this command."
            )

        # HACK: Support returning arbitrary number of images
        num_of_images = prompt.split(" ")[-1:]
        try:
            num_of_images = min(int(num_of_images[0].strip()), 8)
            prompt = prompt.rstrip(str(num_of_images)).strip()
        except ValueError:
            num_of_images = 1

        async with ctx.typing():
            start = time()
            images = await self.generate_images(prompt, num_of_images, ctx)
            gen_time = time() - start

        if not isinstance(images, list):
            return await ctx.send(f"Something went wrong... :( [{images}]")

        if not images:
            return await ctx.send(f"I didn't find anything for `{prompt}`.")

        file_images = {index: discord.File(image, filename=f"{index}.png") for index, image in enumerate(images)}
        for files_images_chunk in chunks(file_images, chunk_size=4):
            embed = discord.Embed(
                colour=await ctx.embed_color(),
                title="Stable Diffusion results",
                url="https://huggingface.co/spaces/stabilityai/stable-diffusion"
            )
            embeds = []
            for index, image in files_images_chunk.items():
                em = embed.copy()
                em.set_image(url=f"attachment://{index}.png")
                em.set_footer(
                    text=(
                        f"Results for: {prompt}, requested by {ctx.author}\n"
                        f"View this output on a desktop client for best results. ({round(gen_time, 1)}s)"
                    )
                )
                embeds.append(em)

            form = []
            payload = {"embeds": [e.to_dict() for e in embeds]}
            form.append({"name": "payload_json", "value": discord.utils.to_json(payload)})
            for index, file in files_images_chunk.items():
                form.append(
                    {
                        "name": f"file{index}",
                        "value": file.fp,
                        "filename": file.filename,
                        "content_type": "application/octet-stream",
                    }
                )

            r = Route("POST", "/channels/{channel_id}/messages", channel_id=ctx.channel.id)
            try:
                await ctx.guild._state.http.request(r, form=form, files=files_images_chunk.values())
            except discord.errors.DiscordServerError as e:
                await ctx.send(f"Discord is sucking... >:( {e}")

    @staticmethod
    async def generate_images(prompt: str, num_of_images: int = 1, ctx=None) -> Union[List[io.BytesIO], int, str]:
        try:
            steps = 50
            payload = {
                "prompt": prompt,
                "iterations": str(num_of_images),
                "steps": str(steps),
                "cfgscale": "7.5",
                "sampler": "k_lms",
                "width": "512",
                "height": "512",
                "seed": "-1",
                "initimg": None,
                "strength": "1",
                "fit": "on",
                "gfpgan_strength": "0.8",
                "upscale_level": "",
                "upscale_strength": "0.75"
            }
            images = []
            total_steps = num_of_images * steps
            current_step = 0
            msg_template = "Running... {}/" + str(total_steps)
            interim_msg = await ctx.send(msg_template.format(0))
            async with aiohttp.ClientSession() as session:
                async with session.post(STABLEDIFFUSION_POST_ENDPOINT, json=payload) as response:
                    if not response.status == 200:
                        return response.status
                    async for line in response.content:
                        resp = json.loads(line)
                        if resp.get("url"):
                            async with session.get(STABLEDIFFUSION_POST_ENDPOINT + resp['url'][1:]) as image:
                                images.append(io.BytesIO(await image.content.read()))
                        else:
                            current_step += 1
                            await interim_msg.edit(content=msg_template.format(current_step))

                return images
        except aiohttp.ClientConnectionError as e:
            return f"Stable Diffusion backend is probably down [{e}]"
        except Exception as e:
            return repr(e)
