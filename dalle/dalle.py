import base64
import io
import os
from itertools import islice
from random import choices
from time import time
from typing import List, Union

import aiohttp
import discord
from discord.http import Route
from redbot.core import commands

DALLE_POST_ENDPOINT = os.environ.get("DALLE_POST_ENDPOINT")
FULL_WORD_LIST_FILE = os.environ.get("DALLE_FULL_WORD_LIST", "/data/words_full")
COMMON_WORD_LIST_FILE = os.environ.get("DALLE_COMMOM_WORD_LIST", "/data/words_common")
try:
    with open(FULL_WORD_LIST_FILE) as f:
        WORDS_FULL = list(set(f.read().splitlines()))
except:
    WORDS_FULL = []
try:
    with open(COMMON_WORD_LIST_FILE) as f:
        WORDS_COMMON = list(set(f.read().splitlines()))
except:
    WORDS_COMMON = []


def chunks(data, chunk_size):
    """Iteratively return chunks of a dictionary"""
    it = iter(data)
    for i in range(0, len(data), chunk_size):
       yield {k:data[k] for k in islice(it, chunk_size)}


class DallE(commands.Cog):
    """Dall-E mini image generation"""

    def __init__(self, bot):
        self.bot = bot

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    @commands.command()
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
        except:
            num_of_words = 4
        WORDS = WORDS_FULL if word_list.upper() == "FULL" else WORDS_COMMON
        prompt = " ".join(choices(WORDS, k=num_of_words))
        await ctx.send(f"Here's the best i can do with `{prompt}` from {word_list.lower()} word list...")
        await self.generate(ctx, prompt=f"{prompt} {num_of_images}")

    @commands.max_concurrency(3, commands.BucketType.default)
    @commands.command()
    @commands.guild_only()
    async def generate(self, ctx: commands.Context, *, prompt: str):
        """
        Generate images through Dall-E mini.

        https://huggingface.co/spaces/dalle-mini/dalle-mini
        """
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
        except:
            num_of_images = 1

        async with ctx.typing():
            start = time()
            images = await self.generate_images(prompt, num_of_images)
            gen_time = time() - start

        if not isinstance(images, list):
            return await ctx.send(f"Something went wrong... :( [{images}]")

        if not images:
            return await ctx.send(f"I didn't find anything for `{prompt}`.")

        file_images = {i: discord.File(image, filename=f"{i}.png") for i, image in enumerate(images)}
        for files_images_chunk in chunks(file_images, chunk_size=4):
            embed = discord.Embed(
                colour=await ctx.embed_color(),
                title="Dall-E Mini results",
                url="https://huggingface.co/spaces/dalle-mini/dalle-mini"
            )
            embeds = []
            for i, image in files_images_chunk.items():
                em = embed.copy()
                em.set_image(url=f"attachment://{i}.png")
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
            await ctx.guild._state.http.request(r, _create_and_send_embed=form, files=files_images_chunk)

    @staticmethod
    async def generate_images(prompt: str, num_of_images: int = 1) -> Union[List[io.BytesIO], int, str]:
        try:
            async with aiohttp.ClientSession() as session:
                dalle_request = {"text": prompt, "num_images": num_of_images}
                async with session.post(DALLE_POST_ENDPOINT, json=dalle_request) as response:
                    if response.status == 200:
                        return [
                            io.BytesIO(base64.decodebytes(bytes(image, "utf-8")))
                            for image in await response.json()
                        ]
                    return response.status
        except aiohttp.ClientConnectionError as e:
            return f"dalle backend is probably down [{e}]"
        except Exception as e:
            return repr(e)
