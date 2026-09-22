"""
tests/test_track_metadata.py

TDD tests for pipeline/openai_schema.py, pipeline/track_metadata.py and the
batch/create_batch_tracks.py + batch/retrieve_batch_tracks.py CLIs.
Run with: pytest tests/test_track_metadata.py -v

No real API calls: kitai.batch functions are monkeypatched on the CLI modules.
File I/O uses tmp_path (+ monkeypatch.chdir so relative constants resolve there).
"""

import csv
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "batch"))

from pipeline.openai_schema import make_openai_strict
from pipeline import track_metadata as tm
from pipeline.constants import (
    TRACKS_INPUT_FILE,
    TRACK_METADATA_FILE,
    TRACK_METADATA_CSV,
    PENDING_TRACKS_BATCH_FILE,
    TRACKS_BATCH_META_FILE,
)
import create_batch_tracks as cbt
import retrieve_batch_tracks as rbt


# ── Fixtures ─────────────────────────────────────────────────────────────────

CSV_HEADER = "artist,title,count\n"


def _write_csv(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV_HEADER + body, encoding="utf-8")
    return path


def _valid_metadata(**overrides) -> dict:
    """A TrackMetadata payload that validates; override single fields per test."""
    payload = {
        "identified": True,
        "confidence": "high",
        "artists": ["Marco Carola"],
        "title": "Step By Step",
        "mix_name": "Original Mix",
        "remixers": [],
        "featured_artists": [],
        "record_label": "Zenit",
        "catalog_number": None,
        "release_title": None,
        "release_type": "EP",
        "release_year": 2003,
        "formats": ["vinyl"],
        "primary_genre": "minimal techno",
        "subgenres": ["loop techno"],
        "bpm_estimate": 130,
        "musical_key": None,
        "energy": "high",
        "mood_tags": ["hypnotic"],
        "dj_set_role": "peak-time",
        "groove": "chunky, rolling loop groove",
        "percussion": ["tribal drums", "clattering hi-hats"],
        "bassline": "dry, stabbing sub bass",
        "vocals": None,
        "melodic_elements": ["hypnotic stab"],
        "texture": ["raw", "gritty"],
        "emotional_character": "relentless, hypnotic",
        "dancefloor_effect": "locks the room into a trance-like shuffle",
        "review_blurb": "A relentless loop workout that never lets go.",
        "description_basis": "known track",
        "similar_artists": ["Christian Varela"],
        "description": "Loop-driven Neapolitan minimal techno.",
        "artist_details": [{
            "name": "Marco Carola",
            "real_name": None,
            "aliases": [],
            "country": "Italy",
            "city": "Naples",
            "active_since": 1990,
            "associated_labels": ["Zenit", "Question"],
            "associated_acts": [],
            "short_bio": "Neapolitan techno DJ and producer.",
        }],
        "parsing_notes": None,
    }
    payload.update(overrides)
    return payload


def _result_item(custom_id: str, content: str | None, status: int = 200,
                 finish_reason: str = "stop", refusal: str | None = None,
                 error=None) -> dict:
    return {
        "id": "req_1",
        "custom_id": custom_id,
        "response": {
            "status_code": status,
            "body": {"choices": [{
                "finish_reason": finish_reason,
                "message": {"content": content, "refusal": refusal},
            }]},
        },
        "error": error,
    }


def _track(artist="Marco Carola", title="Step By Step", count=1) -> tm.TrackInput:
    return tm.TrackInput(
        track_id=tm.make_track_id(artist, title),
        artist=artist, title=title, play_count=count,
    )


# ── make_openai_strict ───────────────────────────────────────────────────────

class _Inner(BaseModel):
    a: int


class _Outer(BaseModel):
    name: str | None = None
    inner: _Inner
    items: list[_Inner]


class TestMakeOpenaiStrict:
    def test_adds_required_and_no_additional_properties_on_all_objects(self):
        schema = make_openai_strict(_Outer.model_json_schema())
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == ["inner", "items", "name"]
        inner = schema["$defs"]["_Inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["a"]

    def test_strips_default_keys(self):
        schema = make_openai_strict(_Outer.model_json_schema())
        assert "default" not in json.dumps(schema)

    def test_ref_siblings_are_dropped(self):
        raw = {"type": "object", "properties": {
            "x": {"$ref": "#/$defs/X", "description": "d"}}}
        schema = make_openai_strict(raw)
        assert schema["properties"]["x"] == {"$ref": "#/$defs/X"}

    def test_recurses_into_anyof(self):
        raw = {"anyOf": [{"type": "object", "properties": {"k": {"type": "string"}}},
                         {"type": "null"}]}
        schema = make_openai_strict(raw)
        assert schema["anyOf"][0]["additionalProperties"] is False
        assert schema["anyOf"][0]["required"] == ["k"]

    def test_does_not_mutate_input(self):
        raw = _Outer.model_json_schema()
        before = json.dumps(raw, sort_keys=True)
        make_openai_strict(raw)
        assert json.dumps(raw, sort_keys=True) == before


# ── make_track_id ────────────────────────────────────────────────────────────

class TestMakeTrackId:
    def test_is_deterministic(self):
        assert tm.make_track_id("Adam Beyer", "China Girl") == tm.make_track_id("Adam Beyer", "China Girl")

    def test_ignores_case_and_whitespace(self):
        assert tm.make_track_id(" adam  BEYER ", "China girl") == tm.make_track_id("Adam Beyer", "China Girl")

    def test_differs_for_different_tracks(self):
        assert tm.make_track_id("Adam Beyer", "Awc (Part 1)") != tm.make_track_id("Adam Beyer", "Awc (Part 2)")

    def test_is_12_hex_chars(self):
        tid = tm.make_track_id("a", "b")
        assert len(tid) == 12 and all(c in "0123456789abcdef" for c in tid)


# ── load_tracks ──────────────────────────────────────────────────────────────

class TestLoadTracks:
    def test_happy_path_parses_rows(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", 'Adam Beyer,China Girl,3\n"Vanjee, Julian Collazos",Rumba,1\n')
        tracks = tm.load_tracks(path)
        assert [(t.artist, t.title, t.play_count) for t in tracks] == [
            ("Adam Beyer", "China Girl", 3),
            ("Vanjee, Julian Collazos", "Rumba", 1),
        ]

    def test_skips_rows_where_artist_and_title_are_both_unknown(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", "(unknown),(unknown),29\nUnknown,,1\n(unknown),Marco Carola - Weekend,1\n")
        tracks = tm.load_tracks(path)
        assert [t.title for t in tracks] == ["Marco Carola - Weekend"]

    def test_keeps_row_when_only_title_is_unknown(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", "2000 and One,(unknown),1\n")
        assert len(tm.load_tracks(path)) == 1

    def test_duplicate_normalised_rows_merge_and_sum_counts(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", "2000 And One,Spanish Fly,1\n2000 and One,spanish fly,2\n")
        tracks = tm.load_tracks(path)
        assert len(tracks) == 1
        assert tracks[0].play_count == 3
        assert tracks[0].artist == "2000 And One"  # first spelling wins

    def test_missing_count_defaults_to_one(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", "Winx,Don't Laugh,\n")
        assert tm.load_tracks(path)[0].play_count == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = _write_csv(tmp_path / "t.csv", "")
        assert tm.load_tracks(path) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tm.load_tracks(tmp_path / "nope.csv")

    def test_missing_columns_raise_value_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("name,song\nx,y\n", encoding="utf-8")
        with pytest.raises(ValueError, match="artist"):
            tm.load_tracks(path)


# ── select_tracks_to_submit ──────────────────────────────────────────────────

class TestSelectTracksToSubmit:
    def test_excludes_already_done(self):
        a, b = _track("A", "1"), _track("B", "2")
        assert tm.select_tracks_to_submit([a, b], done_ids={a.track_id}) == [b]

    def test_limit_caps_result(self):
        tracks = [_track("A", str(i)) for i in range(5)]
        assert len(tm.select_tracks_to_submit(tracks, set(), limit=2)) == 2

    def test_limit_none_or_zero_means_no_cap(self):
        tracks = [_track("A", str(i)) for i in range(3)]
        assert len(tm.select_tracks_to_submit(tracks, set(), limit=None)) == 3
        assert len(tm.select_tracks_to_submit(tracks, set(), limit=0)) == 3

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError):
            tm.select_tracks_to_submit([], set(), limit=-1)


# ── build_task ───────────────────────────────────────────────────────────────

class TestBuildTask:
    def test_task_shape(self):
        t = _track()
        task = tm.build_task(t)
        assert task["custom_id"] == f"track-{t.track_id}"
        assert task["method"] == "POST"
        assert task["url"] == "/v1/chat/completions"
        body = task["body"]
        assert body["model"] == tm.TRACK_METADATA_MODEL
        assert body["temperature"] == 0
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == tm.STRICT_SCHEMA

    def test_user_message_carries_raw_artist_and_title(self):
        task = tm.build_task(_track("(unknown)", "Ricardo Villalobos - Mdma"))
        system, user = task["body"]["messages"]
        assert system["role"] == "system" and user["role"] == "user"
        payload = json.loads(user["content"])
        assert payload == {"artist": "(unknown)", "title": "Ricardo Villalobos - Mdma"}

    def test_model_override(self):
        assert tm.build_task(_track(), model="gpt-x")["body"]["model"] == "gpt-x"

    def test_strict_schema_is_strict_everywhere(self):
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert node["additionalProperties"] is False
                    assert set(node["required"]) == set(node["properties"])
                assert "default" not in node
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(tm.STRICT_SCHEMA)


# ── parse_result_item ────────────────────────────────────────────────────────

class TestParseResultItem:
    def setup_method(self):
        self.track = _track()
        self.cid = tm.custom_id_for(self.track)
        self.inputs = {self.cid: self.track}

    def test_happy_path_builds_record(self):
        item = _result_item(self.cid, json.dumps(_valid_metadata()))
        record = tm.parse_result_item(item, self.inputs, batch_id="batch_1", model="m")
        assert record["track_id"] == self.track.track_id
        assert record["input_artist"] == "Marco Carola"
        assert record["input_title"] == "Step By Step"
        assert record["play_count"] == 1
        assert record["batch_id"] == "batch_1"
        assert record["model"] == "m"
        assert record["metadata"]["primary_genre"] == "minimal techno"

    def test_unknown_custom_id_raises(self):
        item = _result_item("track-deadbeef0000", json.dumps(_valid_metadata()))
        with pytest.raises(tm.TrackResultError, match="custom_id"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_item_error_raises(self):
        item = _result_item(self.cid, None, error={"code": "x"})
        with pytest.raises(tm.TrackResultError, match="item error"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_non_200_raises(self):
        item = _result_item(self.cid, "{}", status=500)
        with pytest.raises(tm.TrackResultError, match="HTTP 500"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_refusal_raises(self):
        item = _result_item(self.cid, None, refusal="no")
        with pytest.raises(tm.TrackResultError, match="refus"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_truncated_output_raises(self):
        item = _result_item(self.cid, '{"identified": tr', finish_reason="length")
        with pytest.raises(tm.TrackResultError, match="length"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_invalid_json_raises(self):
        item = _result_item(self.cid, "not json")
        with pytest.raises(tm.TrackResultError, match="schema"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_schema_violation_raises(self):
        item = _result_item(self.cid, json.dumps(_valid_metadata(primary_genre="polka")))
        with pytest.raises(tm.TrackResultError, match="schema"):
            tm.parse_result_item(item, self.inputs, batch_id="b", model="m")

    def test_unidentified_track_is_still_a_valid_record(self):
        meta = _valid_metadata(identified=False, confidence="low", artists=[],
                               title=None, artist_details=[], primary_genre="other")
        item = _result_item(self.cid, json.dumps(meta))
        record = tm.parse_result_item(item, self.inputs, batch_id="b", model="m")
        assert record["metadata"]["identified"] is False


# ── load_records / merge_records / write_records ─────────────────────────────

class TestRecordsIO:
    def _record(self, track_id: str, title: str = "T") -> dict:
        return {"track_id": track_id, "input_artist": "A", "input_title": title,
                "play_count": 1, "batch_id": "b", "model": "m",
                "metadata": _valid_metadata(title=title)}

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert tm.load_records(tmp_path / "none.jsonl") == {}

    def test_merge_keeps_existing_on_duplicate(self):
        existing = {"x": self._record("x", "old")}
        merged, added, dupes = tm.merge_records(existing, [self._record("x", "new"), self._record("y")])
        assert merged["x"]["input_title"] == "old"
        assert set(merged) == {"x", "y"}
        assert (added, dupes) == (1, 1)

    def test_merge_overwrite_replaces_existing(self):
        existing = {"x": self._record("x", "old")}
        merged, added, dupes = tm.merge_records(existing, [self._record("x", "new")], overwrite=True)
        assert merged["x"]["input_title"] == "new"
        assert (added, dupes) == (1, 0)

    def test_write_then_load_roundtrip(self, tmp_path):
        jsonl, csv_path = tmp_path / "o.jsonl", tmp_path / "o.csv"
        records = {"x": self._record("x"), "y": self._record("y")}
        tm.write_records(records, jsonl, csv_path)
        assert tm.load_records(jsonl) == records
        assert list(tmp_path.glob("*.tmp")) == []

    def test_csv_flattens_lists_and_nested_objects(self, tmp_path):
        jsonl, csv_path = tmp_path / "o.jsonl", tmp_path / "o.csv"
        rec = self._record("x")
        rec["metadata"]["mood_tags"] = ["hypnotic", "dark"]
        tm.write_records({"x": rec}, jsonl, csv_path)
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["track_id"] == "x"
        assert row["mood_tags"] == "hypnotic; dark"
        assert json.loads(row["artist_details"])[0]["city"] == "Naples"
        assert row["catalog_number"] == ""  # None -> empty cell

    def test_jsonl_is_sorted_by_track_id(self, tmp_path):
        jsonl, csv_path = tmp_path / "o.jsonl", tmp_path / "o.csv"
        tm.write_records({"b": self._record("b"), "a": self._record("a")}, jsonl, csv_path)
        ids = [json.loads(l)["track_id"] for l in jsonl.read_text(encoding="utf-8").splitlines()]
        assert ids == ["a", "b"]


# ── create_batch_tracks CLI ──────────────────────────────────────────────────

class _FakeClient:
    """Stand-in for OpenAI(); kitai functions are monkeypatched so it is never called."""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_csv(TRACKS_INPUT_FILE, "Adam Beyer,China Girl,1\n(unknown),Marco Carola - Weekend,2\n(unknown),(unknown),9\n")
    return tmp_path


class TestCreateBatchTracksMain:
    def test_submits_and_writes_sentinels(self, workdir, monkeypatch):
        calls = {}

        def fake_submit(client, tasks, endpoint, metadata=None):
            calls["tasks"], calls["endpoint"] = tasks, endpoint
            return "batch_123"

        monkeypatch.setattr(cbt, "submit_batch_job", fake_submit)
        rc = cbt.main([], client=_FakeClient())
        assert rc == 0
        assert calls["endpoint"] == "/v1/chat/completions"
        assert len(calls["tasks"]) == 2
        assert PENDING_TRACKS_BATCH_FILE.read_text(encoding="utf-8") == "batch_123"
        meta = json.loads(TRACKS_BATCH_META_FILE.read_text(encoding="utf-8"))
        assert meta["batch_id"] == "batch_123"
        assert set(meta["tracks"]) == {t["custom_id"] for t in calls["tasks"]}

    def test_refuses_when_batch_already_pending(self, workdir, monkeypatch):
        PENDING_TRACKS_BATCH_FILE.write_text("batch_old", encoding="utf-8")
        monkeypatch.setattr(cbt, "submit_batch_job", lambda *a, **k: pytest.fail("must not submit"))
        assert cbt.main([], client=_FakeClient()) == 1

    def test_skips_tracks_already_enriched(self, workdir, monkeypatch):
        done_id = tm.make_track_id("Adam Beyer", "China Girl")
        TRACK_METADATA_FILE.write_text(json.dumps({"track_id": done_id}) + "\n", encoding="utf-8")
        captured = {}
        monkeypatch.setattr(cbt, "submit_batch_job",
                            lambda c, tasks, endpoint, metadata=None: captured.setdefault("t", tasks) and "b")
        cbt.main([], client=_FakeClient())
        assert [t["custom_id"] for t in captured["t"]] == [
            f"track-{tm.make_track_id('(unknown)', 'Marco Carola - Weekend')}"]

    def test_nothing_to_submit_is_clean_noop(self, workdir, monkeypatch):
        ids = [tm.make_track_id("Adam Beyer", "China Girl"),
               tm.make_track_id("(unknown)", "Marco Carola - Weekend")]
        TRACK_METADATA_FILE.write_text("".join(json.dumps({"track_id": i}) + "\n" for i in ids), encoding="utf-8")
        monkeypatch.setattr(cbt, "submit_batch_job", lambda *a, **k: pytest.fail("must not submit"))
        assert cbt.main([], client=_FakeClient()) == 0
        assert not PENDING_TRACKS_BATCH_FILE.exists()

    def test_dry_run_writes_debug_jsonl_but_does_not_submit(self, workdir, monkeypatch):
        monkeypatch.setattr(cbt, "submit_batch_job", lambda *a, **k: pytest.fail("must not submit"))
        assert cbt.main(["--dry-run", "--limit", "1"], client=_FakeClient()) == 0
        assert not PENDING_TRACKS_BATCH_FILE.exists()
        from pipeline.constants import BATCH_FILE_TRACKS
        assert len(BATCH_FILE_TRACKS.read_text(encoding="utf-8").splitlines()) == 1


# ── retrieve_batch_tracks CLI ────────────────────────────────────────────────

def _status(status: str) -> dict:
    return {"batch_id": "batch_123", "status": status,
            "is_terminal": status in {"completed", "failed", "expired", "cancelled"},
            "is_complete": status == "completed",
            "counts": {"total": 1, "completed": 1, "failed": 0},
            "output_file_id": "f", "error_file_id": None}


@pytest.fixture
def pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    track = _track()
    cid = tm.custom_id_for(track)
    PENDING_TRACKS_BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    PENDING_TRACKS_BATCH_FILE.write_text("batch_123", encoding="utf-8")
    TRACKS_BATCH_META_FILE.write_text(json.dumps({
        "batch_id": "batch_123", "model": "m",
        "tracks": {cid: {"track_id": track.track_id, "artist": track.artist,
                         "title": track.title, "play_count": track.play_count}},
    }), encoding="utf-8")
    return cid


class TestRetrieveBatchTracksMain:
    def test_no_pending_file_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert rbt.main(client=_FakeClient()) == 1

    def test_in_progress_exits_2(self, pending, monkeypatch):
        monkeypatch.setattr(rbt, "check_batch_job", lambda c, b: _status("in_progress"))
        assert rbt.main(client=_FakeClient()) == 2
        assert PENDING_TRACKS_BATCH_FILE.exists()

    def test_terminal_failure_exits_1_and_clears_sentinels(self, pending, monkeypatch):
        monkeypatch.setattr(rbt, "check_batch_job", lambda c, b: _status("expired"))
        assert rbt.main(client=_FakeClient()) == 1
        assert not PENDING_TRACKS_BATCH_FILE.exists()
        assert not TRACKS_BATCH_META_FILE.exists()

    def test_completed_writes_outputs_and_clears_sentinels(self, pending, monkeypatch):
        monkeypatch.setattr(rbt, "check_batch_job", lambda c, b: _status("completed"))
        monkeypatch.setattr(rbt, "download_batch_results",
                            lambda c, b: [_result_item(pending, json.dumps(_valid_metadata()))])
        assert rbt.main(client=_FakeClient()) == 0
        records = tm.load_records(TRACK_METADATA_FILE)
        assert len(records) == 1
        assert TRACK_METADATA_CSV.exists()
        assert not PENDING_TRACKS_BATCH_FILE.exists()
        assert not TRACKS_BATCH_META_FILE.exists()

    def test_partial_failures_still_write_successes_and_exit_0(self, pending, monkeypatch):
        other = _track("Adam Beyer", "China Girl")
        other_cid = tm.custom_id_for(other)
        meta = json.loads(TRACKS_BATCH_META_FILE.read_text(encoding="utf-8"))
        meta["tracks"][other_cid] = {"track_id": other.track_id, "artist": other.artist,
                                     "title": other.title, "play_count": 1}
        TRACKS_BATCH_META_FILE.write_text(json.dumps(meta), encoding="utf-8")
        monkeypatch.setattr(rbt, "check_batch_job", lambda c, b: _status("completed"))
        monkeypatch.setattr(rbt, "download_batch_results", lambda c, b: [
            _result_item(pending, json.dumps(_valid_metadata())),
            _result_item(other_cid, "garbage"),
        ])
        assert rbt.main(client=_FakeClient()) == 0
        assert len(tm.load_records(TRACK_METADATA_FILE)) == 1
        # sentinels cleared: failed tracks are simply resubmitted next run
        assert not PENDING_TRACKS_BATCH_FILE.exists()


# ── enrich_tracks_direct CLI (synchronous Chat Completions, no Batch API) ────

sys.path.insert(0, str(PROJECT_ROOT / "enrich"))
import enrich_tracks_direct as etd


class _FakeCompletion:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self._d = {"choices": [{"finish_reason": finish_reason,
                                "message": {"content": content, "refusal": None}}]}

    def model_dump(self) -> dict:
        return self._d


class _FakeDirectClient:
    """Mimics OpenAI().chat.completions.create; `responses` maps user content -> content | Exception."""

    def __init__(self, responses: dict):
        self.responses, self.calls = responses, []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **body):
        self.calls.append(body)
        user = json.loads(body["messages"][1]["content"])
        out = self.responses.get(user["title"], json.dumps(_valid_metadata()))
        if isinstance(out, Exception):
            raise out
        return _FakeCompletion(out)


class TestCompletionToResultItem:
    def test_wraps_completion_in_batch_item_shape(self):
        item = etd.completion_to_result_item("track-x", _FakeCompletion('{"a":1}'))
        assert item["custom_id"] == "track-x"
        assert item["error"] is None
        assert item["response"]["status_code"] == 200
        assert item["response"]["body"]["choices"][0]["message"]["content"] == '{"a":1}'

    def test_exception_becomes_item_error(self):
        item = etd.error_to_result_item("track-x", RuntimeError("boom"))
        assert item["error"] == {"type": "RuntimeError", "message": "boom"}
        assert item["response"] is None


class TestEnrichTracksDirectMain:
    def test_happy_path_writes_records_with_direct_batch_id(self, workdir):
        client = _FakeDirectClient({})
        assert etd.main([], client=client) == 0
        assert len(client.calls) == 2
        records = tm.load_records(TRACK_METADATA_FILE)
        assert len(records) == 2
        assert {r["batch_id"] for r in records.values()} == {"direct"}
        assert TRACK_METADATA_CSV.exists()

    def test_sends_the_same_body_as_the_batch_task(self, workdir):
        client = _FakeDirectClient({})
        etd.main(["--limit", "1"], client=client)
        track = tm.load_tracks(TRACKS_INPUT_FILE)[0]
        assert client.calls[0] == tm.build_task(track)["body"]

    def test_skips_tracks_in_pending_batch_by_default(self, workdir):
        pending_id = tm.make_track_id("Adam Beyer", "China Girl")
        TRACKS_BATCH_META_FILE.write_text(json.dumps({"batch_id": "b", "model": "m", "tracks": {
            f"track-{pending_id}": {"track_id": pending_id, "artist": "Adam Beyer",
                                    "title": "China Girl", "play_count": 1}}}), encoding="utf-8")
        client = _FakeDirectClient({})
        etd.main([], client=client)
        titles = [json.loads(c["messages"][1]["content"])["title"] for c in client.calls]
        assert titles == ["Marco Carola - Weekend"]

    def test_include_pending_flag_overrides_skip(self, workdir):
        pending_id = tm.make_track_id("Adam Beyer", "China Girl")
        TRACKS_BATCH_META_FILE.write_text(json.dumps({"batch_id": "b", "model": "m", "tracks": {
            f"track-{pending_id}": {"track_id": pending_id, "artist": "Adam Beyer",
                                    "title": "China Girl", "play_count": 1}}}), encoding="utf-8")
        client = _FakeDirectClient({})
        etd.main(["--include-pending"], client=client)
        assert len(client.calls) == 2

    def test_partial_failure_keeps_successes_and_exits_0(self, workdir):
        client = _FakeDirectClient({"China Girl": RuntimeError("rate limited")})
        assert etd.main([], client=client) == 0
        assert len(tm.load_records(TRACK_METADATA_FILE)) == 1

    def test_all_failed_exits_1(self, workdir):
        client = _FakeDirectClient({"China Girl": RuntimeError("x"),
                                    "Marco Carola - Weekend": "not json"})
        assert etd.main([], client=client) == 1
        assert tm.load_records(TRACK_METADATA_FILE) == {}

    def test_nothing_to_do_makes_no_calls(self, workdir):
        ids = [tm.make_track_id("Adam Beyer", "China Girl"),
               tm.make_track_id("(unknown)", "Marco Carola - Weekend")]
        TRACK_METADATA_FILE.write_text("".join(json.dumps({"track_id": i}) + "\n" for i in ids), encoding="utf-8")
        client = _FakeDirectClient({})
        assert etd.main([], client=client) == 0
        assert client.calls == []


class TestSoundDescriptionSchema:
    def test_sound_fields_present_in_strict_schema(self):
        props = tm.STRICT_SCHEMA["properties"]
        for field in ("groove", "percussion", "bassline", "vocals", "melodic_elements", "texture",
                      "emotional_character", "dancefloor_effect", "review_blurb", "description_basis"):
            assert field in props, field
        assert "instrumentation" not in props  # superseded by percussion/bassline/melodic_elements

    def test_description_basis_is_a_closed_set(self):
        with pytest.raises(Exception):
            tm.TrackMetadata.model_validate(_valid_metadata(description_basis="vibes"))

    def test_prompt_carries_style_reference(self):
        assert "grabs you by the hips" in tm.SYSTEM_PROMPT
        assert "do not copy" in tm.SYSTEM_PROMPT.lower()


class TestSelectTracksRefresh:
    def test_refresh_ignores_done_ids(self):
        a = _track("A", "1")
        assert tm.select_tracks_to_submit([a], done_ids={a.track_id}, refresh=True) == [a]


class TestEnrichTracksDirectRefresh:
    def _seed(self):
        tid = tm.make_track_id("Adam Beyer", "China Girl")
        old = {"track_id": tid, "input_artist": "Adam Beyer", "input_title": "China Girl",
               "play_count": 1, "batch_id": "batch_old", "model": "m",
               "metadata": _valid_metadata(review_blurb=None)}
        TRACK_METADATA_FILE.write_text(json.dumps(old) + "\n", encoding="utf-8")
        return tid

    def test_without_refresh_existing_tracks_are_skipped(self, workdir):
        self._seed()
        client = _FakeDirectClient({})
        etd.main([], client=client)
        assert len(client.calls) == 1

    def test_refresh_reasks_and_overwrites(self, workdir):
        tid = self._seed()
        client = _FakeDirectClient({})
        assert etd.main(["--refresh"], client=client) == 0
        assert len(client.calls) == 2
        rec = tm.load_records(TRACK_METADATA_FILE)[tid]
        assert rec["batch_id"] == "direct"
        assert rec["metadata"]["review_blurb"] == "A relentless loop workout that never lets go."

    def test_refresh_failure_keeps_old_record(self, workdir):
        tid = self._seed()
        client = _FakeDirectClient({"China Girl": RuntimeError("down")})
        etd.main(["--refresh"], client=client)
        assert tm.load_records(TRACK_METADATA_FILE)[tid]["batch_id"] == "batch_old"
