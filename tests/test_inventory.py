"""Unit tests for scripts/inventory.py (Phase 0)."""

import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import inventory  # noqa: E402


def test_parse_dataset_card(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "Thattadavu - 24\nNattadavu - 32\nKattiASdavu - 1\n",
        encoding="utf-8",
    )
    parsed = inventory.parse_dataset_card(readme)
    assert parsed["found"] is True
    assert parsed["categories"] == {"Thattadavu": 24, "Nattadavu": 32, "KattiASdavu": 1}
    assert parsed["total_expected"] == 57


def test_read_mapping_xlsx_merges_alternates_and_catches_typos(tmp_path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["video_name", "adavu_name1", "adavu_name2", "adavu_name3"])
    ws.append([None, None, None, None])
    ws.append(["videos/ParavalAdavu1.mp4", "ParavalAdavu", "TaTaiTaiTaAdavu", "ThaThaiThaiThaAdavu"])
    ws.append(["videos/TaTeiTeiTa1.mp4", "ParavalAdavu", "TaTaiTaiTaAdavu", "ThaThaiThaiThaAdavu"])
    ws.append(["vidoes/ThaHathaJamThariThaAdavu _ Kalakshetra style.mp4", "ThahathaJamTharithaAdavu", None, None])
    wb.save(tmp_path / "adavus_mapping.xlsx")

    mapping = inventory.read_mapping_xlsx(tmp_path)
    assert mapping["found"] is True
    assert mapping["row_count"] == 3
    assert mapping["canonical_alternates"]["ParavalAdavu"] == ["TaTaiTaiTaAdavu", "ThaThaiThaiThaAdavu"]
    assert "ThaHathaJamThariThaAdavu _ Kalakshetra style.mp4" in mapping["by_filename"]
    assert any("blank row" in d for d in mapping["discrepancies"])
    assert any("vidoes/" in d for d in mapping["discrepancies"])


def test_check_song_sequences_empty(tmp_path: Path):
    res = inventory.check_song_sequences(tmp_path, [])
    assert res["has_song_to_dance_sequences"] is False
    assert res["standalone_audio_files"] == []

