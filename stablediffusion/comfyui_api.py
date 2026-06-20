import json
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

import aiohttp


class GenerationFailure(Exception):
    """Image generation failed."""


class GenerationCancelled(GenerationFailure):
    """Image generation was cancelled via /interrupt."""


class ComfyUIClient:
    """Minimal async client for ComfyUI's HTTP + websocket API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    async def queue_prompt(self, workflow: dict, client_id: str, prompt_id: str) -> None:
        payload = {"prompt": workflow, "client_id": client_id, "prompt_id": prompt_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/prompt", json=payload) as resp:
                resp.raise_for_status()

    async def get_history(self, prompt_id: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                resp.raise_for_status()
                return await resp.json()

    async def get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/view?{params}") as resp:
                resp.raise_for_status()
                return await resp.read()

    async def interrupt(self, prompt_id: Optional[str] = None) -> None:
        body = {"prompt_id": prompt_id} if prompt_id else {}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/interrupt", json=body) as resp:
                resp.raise_for_status()

    async def run_workflow(
        self,
        workflow: dict,
        prompt_id: str,
        client_id: str,
        on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> dict:
        """Queue the workflow, stream progress over the websocket, and return the
        /history entry once execution finishes. Raises GenerationCancelled if
        interrupted, or GenerationFailure on an execution_error."""
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}") as ws:
                await self.queue_prompt(workflow, client_id, prompt_id)

                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    message = json.loads(msg.data)
                    data = message.get("data", {})
                    msg_type = message.get("type")

                    if msg_type == "progress" and on_progress is not None:
                        await on_progress(data.get("value", 0), data.get("max", 0))
                    elif msg_type == "executing":
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            break
                    elif msg_type == "execution_interrupted" and data.get("prompt_id") == prompt_id:
                        raise GenerationCancelled("Generation was cancelled.")
                    elif msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        raise GenerationFailure(data)

        history = await self.get_history(prompt_id)
        return history[prompt_id]

    async def fetch_output_images(self, history_entry: dict) -> list:
        images = []
        for node_output in history_entry.get("outputs", {}).values():
            for image in node_output.get("images", []):
                images.append(
                    await self.get_image_bytes(image["filename"], image["subfolder"], image["type"])
                )
        return images
