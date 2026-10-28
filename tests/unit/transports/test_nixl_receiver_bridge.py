from types import SimpleNamespace

from prime_rl.configs.shared import NIXLWeightBroadcastConfig
from prime_rl.transports.weights.nixl import bridge as bridge_module
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerAgent, TrainerTensorTable


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def get_metadata(self):
        return b"orchestrator-metadata"

    def add_remote_agent(self, metadata):
        self.calls.append(("add_remote_agent", metadata))
        return SimpleNamespace(remote_agent_name="trainer")

    def make_connection(self, peer):
        self.calls.append(("make_connection", peer.remote_agent_name))

    def wait_for_notification(self, peers, notification, *, timeout):
        self.calls.append(("wait_for_notification", [p.remote_agent_name for p in peers], notification, timeout))

    def send_notification(self, peer, notification):
        self.calls.append(("send_notification", peer.remote_agent_name, notification))


class FakeSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def publish(self, *, nixl_metadata=b""):
        self.calls.append(("publish", nixl_metadata))

    def wait_for(self, role, *, count, timeout):
        self.calls.append(("wait_for", role, count, timeout))
        return ["trainer-ref"]

    def fetch(self, ref):
        table = TrainerTensorTable(
            agents=[TrainerAgent(name="trainer", metadata=b"trainer-metadata", device_id=0)],
            staging_buffer_count=1,
            groups=[],
        )
        return SimpleNamespace(nixl_metadata=table.encode())


def test_nixl_receiver_bridge_connects_trainer_once(monkeypatch):
    monkeypatch.setattr(bridge_module, "set_ucx_env_defaults", lambda device_index: None)
    monkeypatch.setattr(bridge_module, "NixlAgent", FakeAgent)
    monkeypatch.setattr(bridge_module, "ModelExpressSession", FakeSession)
    monkeypatch.setattr(bridge_module, "MxClient", lambda server_url: server_url)

    bridge = bridge_module.NixlReceiverBridge(NIXLWeightBroadcastConfig(host="mx", port=8001, session_id="s1"))
    bridge.publish()
    bridge.wait_ready(1, 60.0)
    bridge.wait_ready(2, 60.0)
    bridge.send_complete(2)

    assert bridge.model_express.kwargs["client"] == "mx:8001"
    assert bridge.model_express.kwargs["session_id"] == "s1"
    assert bridge.model_express.calls == [("publish", b"orchestrator-metadata"), ("wait_for", "trainer", 1, 60.0)]
    assert bridge.nixl_agent.calls == [
        ("add_remote_agent", b"trainer-metadata"),
        ("make_connection", "trainer"),
        ("wait_for_notification", ["trainer"], "policy:0000000000000001:ready", 60.0),
        ("wait_for_notification", ["trainer"], "policy:0000000000000002:ready", 60.0),
        ("send_notification", "trainer", "policy:0000000000000002:complete"),
    ]
