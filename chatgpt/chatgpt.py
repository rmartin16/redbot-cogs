import json
import asyncio
import threading
from pathlib import Path
from json import loads
from requests import RequestException
from traceback import format_exc
from time import time

from redbot.core import commands
from revChatGPT.V1 import Chatbot

CHATGPT_POST_ENDPOINT = "http://10.16.16.16:5000"
CHATGPT_CONFIG_PATH = Path("/data/chatgpt.config")
DISCORD_UPDATE_FREQ = 0.5  # seconds


class GenerationFailure(Exception):
    pass


def async_wrap_iter(it):
    """Wrap blocking iterator into an asynchronous one"""
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()
    exception = None
    _END = object()

    async def yield_queue_items():
        while True:
            next_item = await q.get()
            if next_item is _END:
                break
            yield next_item
        if exception is not None:
            # the iterator has raised, propagate the exception
            raise exception

    def iter_to_queue():
        nonlocal exception
        try:
            for item in it:
                # This runs outside the event loop thread, so we
                # must use thread-safe API to talk to the queue.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
        except Exception as e:
            exception = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(_END), loop).result()

    threading.Thread(target=iter_to_queue).start()
    return yield_queue_items()


class StatusMessage:
    def __init__(self, ctx):
        self.ctx = ctx
        self.msg = None

    async def create(self, content=""):
        self.msg = await self.ctx.send(content)

    async def update(self, content):
        if content:
            await self.msg.edit(content=content)


class ChatGPT(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ctx = None
        self.status_msg: StatusMessage = None

        self.chatbot = Chatbot(config=loads(open(CHATGPT_CONFIG_PATH).read()))

    async def red_delete_data_for_user(self, **kwargs):
        """Nothing to delete."""
        return

    @commands.max_concurrency(1, commands.BucketType.default)
    @commands.command()
    @commands.guild_only()
    async def chatgpt(self, ctx: commands.Context, *, prompt: str):
        """Generate text through ChatGPT."""
        self.ctx = ctx
        self.status_msg = StatusMessage(ctx=ctx)
        await self.status_msg.create("Querying...")

        max_message_len = 1900
        try:
            async with ctx.typing():
                current_response: str = ""
                response_start = 0
                next_update = time() + DISCORD_UPDATE_FREQ
                async for response in self.query_chatgpt(prompt):
                    current_response = response[response_start:]

                    # Limit length of any one message
                    if len(current_response) > max_message_len:
                        self.status_msg = StatusMessage(ctx=ctx)
                        await self.status_msg.create("Starting new message...")
                        response_start = len(response)

                    # send latest message
                    if current_response and time() > next_update:
                        await self.status_msg.update(current_response)
                        next_update = time() + DISCORD_UPDATE_FREQ

                # send the last response just in case the freq limiter prevented it
                if current_response:
                    await self.status_msg.update(current_response)

        except Exception as e:
            await self.ctx.send(f"Something went wrong... :( [{e}]")

    async def query_chatgpt(self, prompt: str):
        try:
            async for data in async_wrap_iter(self.chatbot.ask(prompt=prompt)):
                yield data.get("message", "")
        except json.decoder.JSONDecodeError as e:
            raise GenerationFailure(f"This isn't JSON... [{e}]")
        except RequestException as e:
            raise GenerationFailure(f"RequestException [{e}]")
        except Exception as e:
            raise GenerationFailure(f"ChatGPT ERROR {repr(e)}\n{format_exc()}")
            # raise GenerationFailure(f"Unknown error: {repr(e)}")

    @commands.command()
    @commands.guild_only()
    async def chatgptreset(self, ctx: commands.Context):
        try:
            self.chatbot.reset_chat()
        except Exception as e:
            await self.ctx.send(f"Something went wrong... :( [{e}]")
        else:
            await ctx.send("Chat reset successful")
