from __future__ import annotations

import json
from pathlib import Path

from scripts.labs.perception_lab_common import LabRunWriter, read_jsonl


def test_lab_run_writer_creates_standard_files(tmp_path: Path) -> None:
    writer = LabRunWriter(
        component="face",
        mode="enrollment",
        root=tmp_path,
        run_id="test-run",
    )

    writer.write_manifest({"profile": "static_interaction"})
    writer.append_sample(
        {"sample_id": "frame_0001", "components": {"face_detection": {"measured": True}}},
        {"sample_id": "frame_0001", "labels": {"actual_face_count": None}},
    )
    writer.write_quick_summary(["# Summary", "- ok"])

    assert writer.run_dir == tmp_path / "face" / "enrollment" / "test-run"
    assert writer.manifest_path.exists()
    assert writer.samples_path.exists()
    assert writer.labels_path.exists()
    assert (writer.reports_dir / "quick_summary.md").exists()

    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "static_interaction"
    assert manifest["component"] == "face"
    assert manifest["mode"] == "enrollment"

    assert read_jsonl(writer.samples_path)[0]["sample_id"] == "frame_0001"
    assert read_jsonl(writer.labels_path)[0]["labels"]["actual_face_count"] is None
