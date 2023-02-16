import json
import asyncio
import threading
import aiohttp
from pathlib import Path
from json import loads
from traceback import print_exc
from time import time

from redbot.core import commands
from revChatGPT.V1 import Chatbot

CHATGPT_POST_ENDPOINT = "http://10.16.16.16:5000"
CHATGPT_CONFIG_PATH = Path("/data/chatgpt.config")
DISCORD_UPDATE_FREQ = 2  # seconds


class GenerationFailure(Exception):
    pass


def async_wrap_iter(it):
    """Wrap blocking iterator into an asynchronous one"""
    loop = asyncio.get_event_loop()
    q = asyncio.Queue(1)
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
                next_update = time() + DISCORD_UPDATE_FREQ
                async for response in self.query_chatgpt(prompt):
                    if response or time() > next_update:
                        await self.status_msg.update(response)
                        next_update = time() + DISCORD_UPDATE_FREQ

                # response = await self.query_chatgpt(prompt)
                # if len(response) > max_message_len:
                #     chunks = (response[i:i+max_message_len] for i in range(0, len(response), max_message_len))
                #     for part in chunks:
                #         await ctx.send(part)
                # else:
                #     await ctx.send(response)
        except Exception as e:
            await self.status_msg.msg.delete()
            await self.ctx.send(f"Something went wrong... :( [{e}]")

    async def query_chatgpt(self, prompt: str):
        try:
            response = self.chatbot.ask(prompt=prompt)
            ait = async_wrap_iter(response)
            async for data in ait:
                yield data["message"]

            # async with aiohttp.ClientSession() as session:
            #     async with session.post(f"{CHATGPT_POST_ENDPOINT}/query", json={"prompt": prompt}) as response:
            #         response.raise_for_status()
            #         json_response = await response.json()
            #         if "answer" in json_response:
            #             return json_response["answer"]
            #         elif "error" in json_response:
            #             return json_response["error"]
            #         return f"This is what i got back...: {json_response}"
        except json.decoder.JSONDecodeError as e:
            raise GenerationFailure(f"This isn't JSON... [{e}]")
        except aiohttp.ClientResponseError as e:
            raise GenerationFailure(f"Bad HTTP response... [{e}]")
        except aiohttp.ClientConnectionError as e:
            raise GenerationFailure(f"ClientConnectionError [{e}]")
        except Exception as e:
            print_exc()
            raise GenerationFailure(f"ChatGPT ERROR {e.code}: {e.message} from {e.source} ({repr(e)})")
            # raise GenerationFailure(f"Unknown error: {repr(e)}")

    @commands.command()
    @commands.guild_only()
    async def chatgptreset(self, ctx: commands.Context):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{CHATGPT_POST_ENDPOINT}/reset") as response:
                if response.status != 200:
                    await ctx.send("Chat reset failed")
                else:
                    await ctx.send("Chat reset successful")
