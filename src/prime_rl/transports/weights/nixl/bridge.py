"""Orchestrator side of the NIXL weight transfer for the Rust orchestrator.

The Rust ``WeightReceiver`` drives the handshake (broadcast-dir markers, ``/init_broadcaster``,
``/update_weights``) and calls into this class (``prl_io::weights::NixlBridge``) for the parts
with no Rust bindings: the orchestrator's NIXL agent (UCX) and its ModelExpress session. Each
method is the agent/ModelExpress half of the matching step of ``NIXLWeightReceiver``.
"""

from __future__ import annotations

from modelexpress.client import MxClient

from prime_rl.configs.shared import NIXLWeightBroadcastConfig
from prime_rl.transports.weights.nixl.agent import NixlAgent, NixlPeer, make_agent_name, policy_notification, set_ucx_env_defaults
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerTensorTable


class NixlReceiverBridge:
    def __init__(self, config: NIXLWeightBroadcastConfig) -> None:
        set_ucx_env_defaults(0)
        self.nixl_agent = NixlAgent(make_agent_name("orchestrator", 0))
        self.trainer_peer: NixlPeer | None = None
        self.model_express = ModelExpressSession(
            client=MxClient(server_url=f"{config.host}:{config.port}"),
            role="orchestrator",
            rank=0,
            session_id=config.session_id,
            worker_id="orchestrator",
        )

    def publish(self) -> None:
        """Publish this agent's NIXL metadata to ModelExpress (after ``/init_broadcaster``)."""
        self.model_express.publish(nixl_metadata=self.nixl_agent.get_metadata())

    def wait_ready(self, step: int, timeout: float) -> None:
        """Connect the trainer peer on first use, then wait for its ``ready`` notification of v{step}."""
        if self.trainer_peer is None:
            trainer_refs = self.model_express.wait_for("trainer", count=1, timeout=timeout)
            trainer_worker = self.model_express.fetch(trainer_refs[0])
            trainer_table = TrainerTensorTable.decode(trainer_worker.nixl_metadata)
            self.trainer_peer = self.nixl_agent.add_remote_agent(trainer_table.agents[0].metadata)
            self.nixl_agent.make_connection(self.trainer_peer)
        self.nixl_agent.wait_for_notification([self.trainer_peer], policy_notification(step, "ready"), timeout=timeout)

    def send_complete(self, step: int) -> None:
        """Tell the trainer the engines applied v{step}."""
        assert self.trainer_peer is not None, "send_complete before wait_ready"
        self.nixl_agent.send_notification(self.trainer_peer, policy_notification(step, "complete"))
