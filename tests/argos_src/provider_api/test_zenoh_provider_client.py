import base64
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from argos_src.provider_api.factory import (
    DEFAULT_PROVIDER_TRANSPORT,
    create_provider_client,
)
from argos_src.provider_api.fake import FakeProviderClient
from argos_src.provider_api.namespaces import provider_event_key
from argos_src.provider_api.transports.zenoh import ZenohProviderClient
from argos_src.provider_api.wire import (
    OP_BATTERY_EVENT,
    OP_CHARGING_DOCK,
    OP_CANCEL_CHARGING_DOCK,
    OP_GO2_ACTION,
    OP_MOVE_VELOCITY,
    OP_NAVIGATE_TO_POSE,
    OP_PUBLISH_VELOCITY,
    OP_STOP_MOTION,
    build_event,
    decode_message,
    encode_message,
)


class _Subscriber:
    def __init__(self, session, key, callback):
        self.session = session
        self.key = key
        self.callback = callback
        self.undeclared = False

    def undeclare(self):
        self.undeclared = True
        self.session.subscribers.pop(self.key, None)


class _FakeZenohSession:
    def __init__(self):
        self.puts = []
        self.subscribers = {}
        self.declared_subscribers = []
        self.responses = {}

    def declare_subscriber(self, key, callback):
        subscriber = _Subscriber(self, key, callback)
        self.subscribers[key] = subscriber
        self.declared_subscribers.append(key)
        return subscriber

    def put(self, key, payload):
        self.puts.append((key, payload))
        message = decode_message(payload)
        if message.get("type") != "request":
            return
        request_id = message["id"]
        result = self.responses.get(message["op"], {})
        response = {
            "id": request_id,
            "type": "response",
            "ok": True,
            "result": result,
            "error": None,
            "ts": 1.0,
        }
        response_subscriber_key = key.rsplit("/request/", 1)[0] + f"/response/{request_id}"
        subscriber = self.subscribers.get(response_subscriber_key)
        if subscriber is not None:
            subscriber.callback(SimpleNamespace(payload=encode_message(response)))

    def emit_event(self, key, message):
        subscriber = self.subscribers.get(key)
        if subscriber is not None:
            subscriber.callback(SimpleNamespace(payload=encode_message(message)))


def _last_request(session):
    _key, payload = session.puts[-1]
    return decode_message(payload)


def _client(session):
    return ZenohProviderClient(
        key_prefix="argos/providers/puffle-go2",
        resource_id="base",
        session=session,
    )


def test_factory_creates_zenoh_provider_client(monkeypatch):
    created = {}

    class _FakeZenohModule:
        class Config:
            def insert_json5(self, *_args):
                return None

        @staticmethod
        def open(_config):
            created["opened"] = True
            return _FakeZenohSession()

    monkeypatch.delenv("ARGOS_PROVIDER_TRANSPORT", raising=False)
    monkeypatch.setitem(sys.modules, "zenoh", _FakeZenohModule)

    client = create_provider_client(
        key_prefix="argos/providers/puffle-go2",
        resource_id="base",
    )
    client.start()

    assert DEFAULT_PROVIDER_TRANSPORT == "zenoh"
    assert isinstance(client, ZenohProviderClient)
    assert created["opened"] is True


def test_factory_requires_zenoh_provider_route():
    with pytest.raises(ValueError, match="requires manifest-derived"):
        create_provider_client()


def test_factory_keeps_fake_provider_for_tests():
    client = create_provider_client(transport="fake")

    assert isinstance(client, FakeProviderClient)


def test_factory_passes_zenoh_provider_routing_options():
    client = create_provider_client(
        transport="zenoh",
        key_prefix="argos/providers/puffle-go2",
        resource_id="base",
        connect_endpoints=["tcp/127.0.0.1:7447"],
    )

    assert isinstance(client, ZenohProviderClient)
    assert client.key_prefix == "argos/providers/puffle-go2"
    assert client._resource_id == "base"
    assert client._connect_endpoints == ("tcp/127.0.0.1:7447",)


def test_factory_rejects_ros2_transport():
    with pytest.raises(ValueError, match="not supported"):
        create_provider_client(transport="ros2")


def test_zenoh_transport_reports_missing_python_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "zenoh", None)
    client = ZenohProviderClient(
        key_prefix="argos/providers/puffle-go2",
        resource_id="base",
    )

    with pytest.raises(RuntimeError, match="pip install eclipse-zenoh"):
        client.start()


def test_go2_action_uses_provider_resource_request_keys():
    session = _FakeZenohSession()
    client = _client(session)

    client.perform_go2_action(
        api_id=1003,
        parameter={"data": True},
        priority=2,
    )

    request = _last_request(session)
    assert request["type"] == "request"
    assert request["op"] == OP_GO2_ACTION
    assert request["args"] == {
        "api_id": 1003,
        "parameter": {"data": True},
        "topic": "rt/api/sport/request",
        "priority": 2,
    }
    assert session.puts[-1][0] == (
        f"argos/providers/puffle-go2/resources/base/request/{request['id']}"
    )
    assert (
        f"argos/providers/puffle-go2/resources/base/response/{request['id']}"
        in session.declared_subscribers
    )


def test_motion_calls_use_capability_contract():
    session = _FakeZenohSession()
    session.responses[OP_MOVE_VELOCITY] = {"duration": 0.4}
    client = _client(session)

    duration = client.move_velocity(linear_x=0.2, angular_z=0.1, duration=0.4)
    client.publish_velocity(angular_z=-0.25)

    move_request = decode_message(session.puts[-2][1])
    publish_request = decode_message(session.puts[-1][1])
    assert duration == 0.4
    assert move_request["op"] == OP_MOVE_VELOCITY
    assert move_request["args"]["linear_x"] == 0.2
    assert move_request["args"]["angular_z"] == 0.1
    assert publish_request["op"] == OP_PUBLISH_VELOCITY
    assert publish_request["args"]["angular_z"] == -0.25


def test_blocking_navigation_uses_completion_budget_for_provider_request():
    session = _FakeZenohSession()
    session.responses[OP_NAVIGATE_TO_POSE] = {
        "accepted": True,
        "outcome": "succeeded",
    }
    client = _client(session)

    client.navigate_to_pose(
        goal_id="goal-1",
        x=1.0,
        y=2.0,
        theta=0.5,
        blocking=True,
        timeout_sec=80.0,
    )

    request = _last_request(session)
    assert request["args"]["timeout_sec"] == 80.0
    assert request["timeout_ms"] == 85_000


def test_nonblocking_navigation_keeps_short_provider_request_timeout():
    session = _FakeZenohSession()
    session.responses[OP_NAVIGATE_TO_POSE] = {"accepted": True}
    client = _client(session)

    client.navigate_to_pose(
        goal_id="goal-1",
        x=1.0,
        y=2.0,
        theta=0.5,
        blocking=False,
        timeout_sec=80.0,
    )

    request = _last_request(session)
    assert request["timeout_ms"] == 3_000


def test_charging_alignment_uses_fixed_budget_plus_provider_grace():
    session = _FakeZenohSession()
    session.responses[OP_CHARGING_DOCK] = {"success": True}
    client = _client(session)

    client.dock_for_charging(
        approach_pose={"x": 1.0, "y": 2.0, "theta": 0.5, "frame_id": "map"},
        dock_timeout_sec=60.0,
        alignment_only=True,
    )

    request = _last_request(session)
    assert request["op"] == OP_CHARGING_DOCK
    assert request["args"]["alignment_only"] is True
    assert request["args"]["dock_timeout_sec"] == 60.0
    assert request["timeout_ms"] == 65_000


def test_charging_alignment_cancel_uses_explicit_provider_operation():
    session = _FakeZenohSession()
    session.responses[OP_CANCEL_CHARGING_DOCK] = {"canceled": True}
    client = _client(session)

    result = client.cancel_charging_dock()

    request = _last_request(session)
    assert result == {"canceled": True}
    assert request["op"] == OP_CANCEL_CHARGING_DOCK
    assert request["args"] == {}


def test_zero_velocity_uses_motion_stop_capability():
    session = _FakeZenohSession()
    client = _client(session)

    client.publish_velocity()

    request = _last_request(session)
    assert request["op"] == OP_STOP_MOTION
    assert request["args"] == {}


def test_transform_and_intrinsics_decode_plain_results():
    session = _FakeZenohSession()
    session.responses["tf.transform"] = {
        "translation": [1.0, 2.0, 3.0],
        "rotation": [0.0, 0.0, 0.5, 0.5],
        "stamp_s": 12.5,
    }
    session.responses["camera.intrinsics"] = {
        "fx": 1.0,
        "fy": 2.0,
        "cx": 3.0,
        "cy": 4.0,
        "width": 640,
        "height": 480,
    }
    client = _client(session)

    transform = client.get_transform("odom", "base_link")
    intrinsics = client.get_latest_intrinsics(resource_id="head_realsense")

    assert transform.translation == (1.0, 2.0, 3.0)
    assert transform.rotation == (0.0, 0.0, 0.5, 0.5)
    assert transform.stamp_s == 12.5
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    transform_request = decode_message(session.puts[-2][1])
    assert transform_request["args"]["timeout"] == 0.05


def test_image_snapshot_decodes_raw_array_payload():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    session = _FakeZenohSession()
    session.responses["camera.latest_image"] = {
        "resource_id": "head_realsense",
        "image": {
            "encoding": "raw",
            "dtype": "uint8",
            "shape": [2, 2, 3],
            "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
        },
    }
    client = _client(session)

    frame = client.get_latest_image(resource_id="head_realsense")

    assert frame is not None
    assert frame.resource_id == "head_realsense"
    np.testing.assert_array_equal(frame.image, image)


def test_image_snapshot_converts_rgb8_raw_payload_to_internal_bgr():
    image_rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    expected_bgr = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    session = _FakeZenohSession()
    session.responses["camera.latest_image"] = {
        "resource_id": "head_realsense",
        "image": {
            "encoding": "rgb8",
            "dtype": "uint8",
            "shape": [1, 2, 3],
            "data_b64": base64.b64encode(image_rgb.tobytes()).decode("ascii"),
        },
    }
    client = _client(session)

    frame = client.get_latest_image(resource_id="head_realsense")

    assert frame is not None
    np.testing.assert_array_equal(frame.image, expected_bgr)


def test_image_snapshot_converts_raw_payload_format_rgb8_to_internal_bgr():
    image_rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    expected_bgr = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    session = _FakeZenohSession()
    session.responses["camera.latest_image"] = {
        "resource_id": "arducam_001",
        "image": {
            "encoding": "raw",
            "format": "rgb8",
            "dtype": "uint8",
            "shape": [1, 2, 3],
            "data_b64": base64.b64encode(image_rgb.tobytes()).decode("ascii"),
        },
    }
    client = _client(session)

    frame = client.get_latest_image(resource_id="arducam_001")

    assert frame is not None
    np.testing.assert_array_equal(frame.image, expected_bgr)


def test_image_snapshot_preserves_bgr8_raw_payload():
    image_bgr = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    session = _FakeZenohSession()
    session.responses["camera.latest_image"] = {
        "resource_id": "head_realsense",
        "image": {
            "encoding": "bgr8",
            "dtype": "uint8",
            "shape": [1, 2, 3],
            "data_b64": base64.b64encode(image_bgr.tobytes()).decode("ascii"),
        },
    }
    client = _client(session)

    frame = client.get_latest_image(resource_id="head_realsense")

    assert frame is not None
    np.testing.assert_array_equal(frame.image, image_bgr)


def test_provider_resource_battery_events_update_subscribers():
    session = _FakeZenohSession()
    client = _client(session)
    snapshots = []

    unsubscribe = client.subscribe_battery(snapshots.append)
    session.emit_event(
        provider_event_key("argos/providers/puffle-go2", "base"),
        build_event(
            op=OP_BATTERY_EVENT,
            data={
                "percentage": 0.72,
                "current": -1.1,
                "power_supply_status": 2,
            },
        ),
    )
    unsubscribe()

    assert len(snapshots) == 1
    assert snapshots[0].percentage == 0.72
    assert snapshots[0].current == -1.1
    assert snapshots[0].power_supply_status == 2


def test_protocol_rejects_non_object_payload():
    with pytest.raises(ValueError):
        decode_message(json.dumps(["not", "an", "object"]))
