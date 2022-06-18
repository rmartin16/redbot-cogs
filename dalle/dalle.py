import base64
import io
import logging
import os
from random import choices
from typing import List, Union

import aiohttp
import discord
from discord.http import Route
from redbot.core import commands

DALLE_POST_ENDPOINT = os.environ.get("DALLE_POST_ENDPOINT")
WORD_LIST = os.environ.get("DALLE_WORD_LIST", "/data/words_full")
try:
    with open(WORD_LIST) as f:
        WORDS = list(set(f.read().splitlines()))
    logging.getLogger(__name__).info(f"Read in {len(WORDS)} words in to random word list.")
except:
    WORDS = []


class DallE(commands.Cog):
    """Dall-E mini image generation"""

    def __init__(self, bot):
        self.bot = bot

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    @commands.command()
    @commands.guild_only()
    async def generate_random(self, ctx: commands.Context, *, num_of_words: str = "4"):
        if not WORDS:
            return await ctx.send("Failed to load word list...")
        try:
            num_of_words = int(num_of_words)
        except:
            num_of_words = 4
        prompt = " ".join(choices(WORDS, k=num_of_words))
        await ctx.send(f"Here's the best i can do with `{prompt}`...")
        await self.generate(ctx, prompt=prompt)

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
            num_of_images = min(int(num_of_images[0].strip()), 4)
            prompt = prompt.rstrip(str(num_of_images)).strip()
        except:
            num_of_images = 1

        async with ctx.typing():
            images = await self.generate_images(prompt, num_of_images)

        if not isinstance(images, list):
            return await ctx.send(f"Something went wrong... :( [{images}]")

        if not images:
            return await ctx.send(f"I didn't find anything for `{prompt}`.")

        file_images = [discord.File(image, filename=f"{i}.png") for i, image in enumerate(images)]
        embed = discord.Embed(
            colour=await ctx.embed_color(),
            title="Dall-E Mini results",
            url="https://huggingface.co/spaces/dalle-mini/dalle-mini"
        )
        embeds = []
        for i, image in enumerate(file_images):
            em = embed.copy()
            em.set_image(url=f"attachment://{i}.png")
            em.set_footer(
                text=(
                    f"Results for: {prompt}, requested by {ctx.author}\n"
                    "View this output on a desktop client for best results."
                )
            )
            embeds.append(em)

        form = []
        payload = {"embeds": [e.to_dict() for e in embeds]}
        form.append({"name": "payload_json", "value": discord.utils.to_json(payload)})
        for index, file in enumerate(file_images):
            form.append(
                {
                    "name": f"file{index}",
                    "value": file.fp,
                    "filename": file.filename,
                    "content_type": "application/octet-stream",
                }
            )

        r = Route("POST", "/channels/{channel_id}/messages", channel_id=ctx.channel.id)
        await ctx.guild._state.http.request(r, form=form, files=file_images)

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
