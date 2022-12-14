import json

import aiohttp
from redbot.core import commands


CHATGPT_POST_ENDPOINT = "http://10.16.16.16:5000"


class GenerationFailure(Exception):
    pass

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
        max_message_len = 1900
        try:
            async with ctx.typing():
                response = await self.query_chatgpt(prompt)
                if response > max_message_len:
                    chunks = (response[i:i+max_message_len] for i in range(0, len(response), max_message_len))
                    for part in chunks:
                        await ctx.send(part)
                else:
                    await ctx.send(response)
        except Exception as e:
            await ctx.send(f"ERROR: {e}")

    async def query_chatgpt(self, prompt: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{CHATGPT_POST_ENDPOINT}/query", json={"prompt": prompt}) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    if "answer" in json_response:
                        return json_response["answer"]
                    return f"This is what i got back...: {json_response}"
        except json.decoder.JSONDecodeError as e:
            raise GenerationFailure(f"This isn't JSON... [{e}]")
        except aiohttp.ClientResponseError as e:
            raise GenerationFailure(f"Bad HTTP response... [{e}]")
        except aiohttp.ClientConnectionError as e:
            raise GenerationFailure(f"ClientConnectionError [{e}]")
        except Exception as e:
            raise GenerationFailure(f"Unknown error: {repr(e)}")

    @commands.command()
    @commands.guild_only()
    async def chatgptreset(self, ctx: commands.Context):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{CHATGPT_POST_ENDPOINT}/query") as response:
                if response.status != 200:
                    await ctx.send("Chat reset failed")
                else:
                    await ctx.send("Chat reset successful")
