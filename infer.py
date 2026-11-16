"""Async OpenAI-compatible inference across two independent llama-server replicas."""
import asyncio
import itertools
import os

import httpx

SERVERS = [
    os.getenv("SERVER0", "http://127.0.0.1:8081"),
    os.getenv("SERVER1", "http://127.0.0.1:8082"),
]


class DualGPU:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            trust_env=False,
        )
        self.servers = itertools.cycle(SERVERS)
        self.models: dict[str, str] = {}

    async def __aenter__(self):
        try:
            for server in SERVERS:
                resp = await self.client.get(f"{server}/v1/models")
                resp.raise_for_status()
                self.models[server] = resp.json()["data"][0]["id"]
        except Exception:
            await self.client.aclose()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def generate(self, prompt: str, max_tokens: int = 128) -> dict:
        server = next(self.servers)
        resp = await self.client.post(
            f"{server}/v1/chat/completions",
            json={
                "model": self.models[server],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def main() -> None:
    prompts = [
        "Что такое Transformer? Ответь кратко.",
        "Что такое Attention? Ответь кратко.",
        "Объясни KV-cache в двух предложениях.",
        "Что такое Flash Attention? Ответь кратко.",
        "Как работает LayerNorm? Ответь кратко.",
        "Объясни continuous batching в двух предложениях.",
    ]
    async with DualGPU() as llm:
        results = await asyncio.gather(*(llm.generate(p) for p in prompts))
    for prompt, result in zip(prompts, results):
        print(f"\nВопрос: {prompt}")
        print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
