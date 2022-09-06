import io
import json
import os
from itertools import islice
from random import choices
from time import time
from typing import Dict, List, Union

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

        if not isinstance(images, dict):
            return await ctx.send(f"Something went wrong... :( [{images}]")

        if not images:
            return
            # return await ctx.send(f"I didn't find anything for `{prompt}`.")

        for files_images_chunk in chunks(images, chunk_size=4):
            embed = discord.Embed(
                colour=await ctx.embed_color(),
                title="Stable Diffusion results",
                url="https://huggingface.co/spaces/stabilityai/stable-diffusion"
            )
            embeds = []
            for index, image in files_images_chunk.items():
                em = embed.copy()
                em.set_image(url=f"attachment://{index}")
                em.set_footer(
                    text=(
                        f"Results for: {prompt}, requested by {ctx.author} ({round(gen_time, 1)}s)\n"
                        + " ".join(f"{idx}: {img['config']['seed']}" for idx, img in files_images_chunk.items())
                    )
                )
                embeds.append(em)

            form = []
            payload = {"embeds": [e.to_dict() for e in embeds]}
            form.append({"name": "payload_json", "value": discord.utils.to_json(payload)})
            for index, image in files_images_chunk.items():
                form.append(
                    {
                        "name": index,
                        "value": image["image"].fp,
                        "filename": image["image"].filename,
                        "content_type": "application/octet-stream",
                    }
                )

            r = Route("POST", "/channels/{channel_id}/messages", channel_id=ctx.channel.id)
            try:
                await ctx.guild._state.http.request(
                    r,
                    form=form,
                    files=(f["image"] for f in files_images_chunk.values())
                )
            except discord.errors.DiscordServerError as e:
                await ctx.send(f"Discord is sucking... >:( {e}")

    async def generate_images(
            self, prompt: str, num_of_images: int = 1, ctx=None
    ) -> Union[Dict[str, Dict[str, Union[str, Union[Dict[str, str], io.BytesIO]]]], int, str]:
        steps = 50
        prompt, details = get_details_from_prompt(prompt)
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
            "upscale_level": "2",
            "upscale_strength": "0.75",
        }
        payload.update(details)
        results = []
        images = {}
        total_steps = num_of_images * int(payload["steps"])
        current_step = 0
        step_update = 8
        progress_bar = ProgressBar(total=total_steps)
        interim_msg = await ctx.send(progress_bar.update(current_step))
        await interim_msg.add_reaction("❌")
        self.channels[str(ctx.channel.id)] = {"msg_id": interim_msg.id}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(STABLEDIFFUSION_POST_ENDPOINT, json=payload) as response:
                    if not response.status == 200:
                        return response.status

                    async for line in response.content:
                        resp = json.loads(line)
                        event = resp.get("event", "").lower()

                        if event.startswith("upscaling"):
                            await interim_msg.edit(content="Upscaling images...")
                        elif event == "result":
                            results.append(resp)
                        elif event == "step":
                            current_step += 1
                            if current_step == total_steps or current_step % step_update == 0:
                                await interim_msg.edit(content=progress_bar.update(current_step))

                for result in results:
                    async with session.get(STABLEDIFFUSION_POST_ENDPOINT + result["url"][1:]) as image:
                        name = result["url"].split("/")[-1]
                        images[name] = {
                            "image": discord.File(io.BytesIO(await image.content.read()), filename=name),
                            "config": result["config"],
                        }

        except aiohttp.ClientConnectionError as e:
            return f"Stable Diffusion backend is probably down [{e}]"
        except Exception as e:
            return repr(e)
        else:
            return images
        finally:
            await interim_msg.delete()

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


def get_details_from_prompt(prompt):
    new_prompt = []
    prompt_details = {}
    detail_list = [
        'iterations', 'steps', 'cfgscale', 'sampler', 'width', 'height', 'seed', 'initimg',
        'strength', 'fit', 'gfpgan_strength', 'upscale_level', 'upscale_strength'
    ]
    for piece in prompt.split(" "):
        for detail_name in detail_list:
            if piece.startswith(f"{detail_name}:"):
                detail_value = piece.split(":")[1]
                prompt_details[detail_name] = detail_value
                break
        else:
            new_prompt.append(piece)
    return " ".join(new_prompt), prompt_details


class ProgressBar:
    def __init__(self, total: int):
        """
        Context manager to display a progress bar in the console.
        Continuously call update() on the yielded object to redraw the progress bar.
        The progress bar will reach 100% when completed == total.

        :param total: integer representing 100% of progress
        """
        self.bar_width = 30
        self.completed_char = "#"
        self.remaining_char = "."

        self.total = total

    def update(self, completed: int):
        """
        Build the progress bar and return it.

        :param completed: amount of the total to show as completed.
        """
        completed_count = int(self.bar_width * completed / self.total)
        bar_completed = self.completed_char * completed_count
        bar_remaining = self.remaining_char * (self.bar_width - completed_count)
        percent_done = int(completed_count * (100 / self.bar_width))
        return f"`{bar_completed}{bar_remaining} {percent_done}%`"
