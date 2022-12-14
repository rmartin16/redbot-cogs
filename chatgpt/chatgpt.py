import json
from pathlib import Path

import discord
from redbot.core import commands

from revChatGPT.revChatGPT import AsyncChatbot as Chatbot

CHATGPT_CONFIG_PATH = Path("/config/data/chatgpt.config")


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

        self.chatbot_config = self.get_config()
        self.chatbot = Chatbot(self.chatbot_config, conversation_id=None)

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    async def handle_response(self, prompt) -> str:
        self.chatbot.refresh_session()
        response = await self.chatbot.get_chat_response(prompt, output="text")
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

        response = await self.handle_response(prompt)

        await ctx.send(response)
