import asyncio
import json
from pathlib import Path

import aiohttp
from redbot.core import commands


CHATGPT_POST_ENDPOINT = "http://10.16.16.16:5000/query"


class StatusMessage:
    def __init__(self, ctx):
        self.ctx = ctx
        self.msg = None

    async def create(self, content):
        self.msg = await self.ctx.send(content)

    async def update(self, content):
        await self.msg.edit(content=content)


class ChatGPT(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    @commands.max_concurrency(1, commands.BucketType.default)
    @commands.command()
    @commands.guild_only()
    async def chatgpt(self, ctx: commands.Context, *, prompt: str):
        """Generate text through ChatGPT."""

        async with aiohttp.ClientSession() as session:
            async with session.post(CHATGPT_POST_ENDPOINT, json={"prompt": prompt}) as response:
                response.raise_for_status()
                chat_response = (await response.json()).get("answer", "no response...")

        await ctx.send(chat_response)
