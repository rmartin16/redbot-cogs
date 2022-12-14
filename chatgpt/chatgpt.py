import asyncio
import json
from pathlib import Path

from redbot.core import commands

from revChatGPT.revChatGPT import Chatbot

CHATGPT_CONFIG_PATH = Path("/data/chatgpt.config")


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

        loop = asyncio.get_event_loop()

        self.chatbot_config = self.get_config()
        self.chatbot = None

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    async def send_query(self, prompt) -> str:
        loop = asyncio.get_event_loop()
        if self.chatbot is None:
            self.chatbot = await loop.run_in_executor(None, Chatbot, self.chatbot_config)
        await loop.run_in_executor(None, self.chatbot.refresh_session, )
        response = await loop.run_in_executor(None, self.chatbot.get_chat_response, (prompt, "text"))
        return response['message']

    def get_config(self) -> dict:
        with open(CHATGPT_CONFIG_PATH, 'r') as f:
            return json.load(f)


    @commands.max_concurrency(1, commands.BucketType.default)
    @commands.command()
    @commands.guild_only()
    async def chatgpt(self, ctx: commands.Context, *, prompt: str):
        """Generate text through ChatGPT."""

        # chat_msg = StatusMessage(ctx=ctx)

        response = await self.send_query(prompt)

        await ctx.send(response)
