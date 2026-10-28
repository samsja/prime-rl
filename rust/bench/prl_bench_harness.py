"""Benchmark harness plugin (`env.agent.harness.id = "prl-bench-harness"`): one chat
completion through the interception endpoint, run in-process. The built-in `null` harness
launches a Python program per rollout, which caps an env-server worker at a few rollouts
per second; this keeps the env servers cheap so the orchestrator is what gets measured.
Importable when rust/bench is on PYTHONPATH."""

from typing import Literal

import httpx

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import message_to_wire
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

__all__ = ["BenchHarness"]

_client: httpx.AsyncClient | None = None


class BenchHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"


class BenchHarness(Harness[BenchHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        pass

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        global _client
        if _client is None:
            _client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=1024))
        system_prompt, prompt = self.resolve_prompt(data)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(message) for message in prompt)
        response = await _client.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
            json={"model": ctx.model, "messages": messages},
        )
        response.raise_for_status()
        return ProgramResult(exit_code=0, stdout=response.text, stderr="")
