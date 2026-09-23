"""
tests/test_artist_profiles.py

TDD tests for pipeline/openai_results.py, pipeline/artist_profiles.py and
enrich/enrich_artists.py. No real API calls: fake clients only.
Run with: pytest tests/test_artist_profiles.py -v
"""

import csv
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "enrich"))

from pipeline import artist_profiles as ap
from pipeline import openai_results as orr
from pipeline import track_metadata as tm
import enrich_artists as ea


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _profile(**overrides) -> dict:
    p = {
        "known": True, "confidence": "high", "name": "Heartthrob",
        "real_name": "Jesse Siminski", "aliases": [], "country": "United States", "city": None,
        "active_since": 2005, "associated_labels": ["Minus"], "associated_acts": ["Magda"],
        "styles": ["minimal techno"], "notable_releases": ["Baby Kate"],
        "short_bio": "Minus-era minimal techno producer.",
    }
    p.update(overrides)
    return p


def _track_rec(tid, artists, remixers=(), featured=(), title="T"):
    return {"track_id": tid, "input_artist": "x", "input_title": title, "play_count": 1,
            "batch_id": "b", "model": "m",
            "metadata": {"artists": list(artists), "remixers": list(remixers),
                         "featured_artists": list(featured), "title": title}}


class _Completion:
    def __init__(self, content, finish_reason="stop"):
        self._d = {"choices": [{"finish_reason": finish_reason,
                                "message": {"content": content, "refusal": None}}]}

    def model_dump(self):
        return self._d


class _FakeClient:
    """chat.completions.create -> content keyed by the requested artist name."""

    def __init__(self, responses=None):
        self.responses, self.calls = responses or {}, []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **body):
        self.calls.append(body)
        name = json.loads(body["messages"][1]["content"])["artist"]
        out = self.responses.get(name, json.dumps(_profile(name=name)))
        if isinstance(out, Exception):
            raise out
        return _Completion(out)


# ── openai_results ───────────────────────────────────────────────────────────

class _M(BaseModel):
    a: int


class TestValidateResultItem:
    def _item(self, content, status=200, finish="stop", refusal=None, error=None):
        return {"custom_id": "c", "error": error, "response": None if error else {
            "status_code": status, "body": {"choices": [{"finish_reason": finish,
                                                          "message": {"content": content, "refusal": refusal}}]}}}

    def test_happy_path(self):
        assert orr.validate_result_item(self._item('{"a": 1}'), _M).a == 1

    @pytest.mark.parametrize("kwargs,match", [
        ({"content": None, "error": {"x": 1}}, "item error"),
        ({"content": "{}", "status": 500}, "HTTP 500"),
        ({"content": None, "refusal": "no"}, "refus"),
        ({"content": '{"a"', "finish": "length"}, "length"),
        ({"content": '{"a": "x"}'}, "schema"),
    ])
    def test_failures_raise_result_error(self, kwargs, match):
        with pytest.raises(orr.ResultError, match=match):
            orr.validate_result_item(self._item(**kwargs), _M)

    def test_track_result_error_is_the_shared_error(self):
        assert tm.TrackResultError is orr.ResultError


class TestCallChatCompletions:
    def test_returns_one_item_per_body_with_errors_captured(self):
        client = _FakeClient({"B": RuntimeError("boom")})
        bodies = {"c1": {"messages": [{}, {"content": json.dumps({"artist": "A"})}]},
                  "c2": {"messages": [{}, {"content": json.dumps({"artist": "B"})}]}}
        items = {i["custom_id"]: i for i in orr.call_chat_completions(client, bodies, workers=2)}
        assert items["c1"]["error"] is None and items["c1"]["response"]["status_code"] == 200
        assert items["c2"]["error"] == {"type": "RuntimeError", "message": "boom"}


# ── artist ids / collection ──────────────────────────────────────────────────

class TestMakeArtistId:
    def test_case_and_whitespace_insensitive(self):
        assert ap.make_artist_id(" loco  DICE") == ap.make_artist_id("Loco Dice")

    def test_distinct(self):
        assert ap.make_artist_id("Gaiser") != ap.make_artist_id("Magda")


class TestCollectArtists:
    def test_dedupes_and_counts_across_roles(self):
        recs = {"1": _track_rec("1", ["Heartthrob"], remixers=["Magda"], title="Baby Kate"),
                "2": _track_rec("2", ["heartthrob"], remixers=["Konrad Black"], title="Baby Kate"),
                "3": _track_rec("3", ["Magda"], title="Black Leather Wonder")}
        artists = {a.name: a for a in ap.collect_artists(recs)}
        assert set(artists) == {"Heartthrob", "Magda", "Konrad Black"}
        assert artists["Heartthrob"].track_count == 2
        assert artists["Magda"].track_count == 2
        assert artists["Magda"].example_titles == ["Baby Kate", "Black Leather Wonder"]

    def test_example_titles_capped_and_unique(self):
        recs = {str(i): _track_rec(str(i), ["A"], title=f"T{i % 7}") for i in range(20)}
        (a,) = ap.collect_artists(recs)
        assert len(a.example_titles) == ap.MAX_EXAMPLE_TITLES
        assert len(set(a.example_titles)) == ap.MAX_EXAMPLE_TITLES

    @pytest.mark.parametrize("junk", ["", "VA", "Various Artists", "(unknown)", "Unknown"])
    def test_skips_placeholder_names(self, junk):
        assert ap.collect_artists({"1": _track_rec("1", [junk])}) == []

    def test_records_without_metadata_are_ignored(self):
        assert ap.collect_artists({"1": {"track_id": "1"}}) == []

    def test_sorted_by_track_count_then_name(self):
        recs = {"1": _track_rec("1", ["B"]), "2": _track_rec("2", ["A"]), "3": _track_rec("3", ["B"])}
        assert [a.name for a in ap.collect_artists(recs)] == ["B", "A"]


# ── task / parse ─────────────────────────────────────────────────────────────

class TestBuildArtistTask:
    def _artist(self):
        return ap.ArtistInput(ap.make_artist_id("Luciano"), "Luciano", 2, ["Bomberos", "Sferic"])

    def test_payload_has_name_and_library_context(self):
        body = ap.build_artist_task(self._artist(), model="gpt-4.1")["body"]
        payload = json.loads(body["messages"][1]["content"])
        assert payload == {"artist": "Luciano", "example_tracks_in_library": ["Bomberos", "Sferic"]}
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["temperature"] == 0

    def test_reasoning_model_omits_temperature(self):
        assert "temperature" not in ap.build_artist_task(self._artist(), model="gpt-5")["body"]

    def test_custom_id(self):
        a = self._artist()
        assert ap.build_artist_task(a)["custom_id"] == f"artist-{a.artist_id}"


class TestEnforceArtistNoInference:
    def test_known_unchanged(self):
        p = ap.ArtistProfile.model_validate(_profile())
        assert ap.enforce_no_inference(p) == p

    def test_unknown_clears_everything_but_name(self):
        p = ap.enforce_no_inference(ap.ArtistProfile.model_validate(_profile(known=False)))
        assert p.name == "Heartthrob"
        assert p.confidence == "low"
        assert p.short_bio == ap.UNKNOWN_ARTIST_NOTE
        for f in ("real_name", "country", "city", "active_since"):
            assert getattr(p, f) is None, f
        for f in ("aliases", "associated_labels", "associated_acts", "styles", "notable_releases"):
            assert getattr(p, f) == [], f


class TestParseArtistResult:
    def test_record_shape(self):
        a = ap.ArtistInput(ap.make_artist_id("Heartthrob"), "Heartthrob", 3, ["Baby Kate"])
        item = {"custom_id": ap.custom_id_for(a), "error": None, "response": {"status_code": 200, "body": {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(_profile()), "refusal": None}}]}}}
        rec = ap.parse_artist_result(item, {ap.custom_id_for(a): a}, source="direct", model="gpt-5")
        assert rec["artist_id"] == a.artist_id
        assert rec["name"] == "Heartthrob" and rec["track_count"] == 3
        assert rec["profile"]["real_name"] == "Jesse Siminski"

    def test_unknown_custom_id_raises(self):
        with pytest.raises(orr.ResultError, match="custom_id"):
            ap.parse_artist_result({"custom_id": "artist-zzz"}, {}, source="d", model="m")


# ── store ────────────────────────────────────────────────────────────────────

class TestArtistStore:
    def test_paths(self):
        assert ap.store_paths(None) == (ap.ARTIST_PROFILES_FILE, ap.ARTIST_PROFILES_CSV)
        j, c = ap.store_paths("gpt-5")
        assert j.name == "artist_profiles_gpt-5.jsonl" and c.name == "artist_profiles_gpt-5.csv"
        with pytest.raises(ValueError):
            ap.store_paths("../x")

    def test_write_load_roundtrip_and_csv(self, tmp_path):
        rec = {"artist_id": "x", "name": "Magda", "track_count": 2, "source": "direct", "model": "m",
               "profile": _profile(name="Magda", associated_labels=["Minus", "Items & Things"])}
        j, c = tmp_path / "a.jsonl", tmp_path / "a.csv"
        ap.write_records({"x": rec}, j, c)
        assert ap.load_records(j) == {"x": rec}
        with c.open(encoding="utf-8", newline="") as f:
            (row,) = list(csv.DictReader(f))
        assert row["name"] == "Magda"
        assert row["associated_labels"] == "Minus; Items & Things"
        assert row["real_name"] == "Jesse Siminski"


# ── enrich_artists CLI ───────────────────────────────────────────────────────

@pytest.fixture
def track_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jsonl, csv_path = tm.store_paths("gpt-5")
    recs = {"1": _track_rec("1", ["Heartthrob"], remixers=["Magda"], title="Baby Kate"),
            "2": _track_rec("2", ["Gaiser"], title="Seepage")}
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in recs.values()), encoding="utf-8")
    return tmp_path


class TestEnrichArtistsMain:
    def test_writes_profiles_for_every_artist(self, track_store):
        client = _FakeClient()
        assert ea.main(["--tracks-tag", "gpt-5", "--output-tag", "gpt-5", "--model", "gpt-5"], client=client) == 0
        assert len(client.calls) == 3
        recs = ap.load_records(ap.store_paths("gpt-5")[0])
        assert {r["name"] for r in recs.values()} == {"Heartthrob", "Magda", "Gaiser"}

    def test_skips_done_and_refresh_reasks(self, track_store):
        ea.main(["--tracks-tag", "gpt-5", "--output-tag", "gpt-5"], client=_FakeClient())
        again = _FakeClient()
        ea.main(["--tracks-tag", "gpt-5", "--output-tag", "gpt-5"], client=again)
        assert again.calls == []
        refresh = _FakeClient()
        ea.main(["--tracks-tag", "gpt-5", "--output-tag", "gpt-5", "--refresh"], client=refresh)
        assert len(refresh.calls) == 3

    def test_limit(self, track_store):
        client = _FakeClient()
        ea.main(["--tracks-tag", "gpt-5", "--limit", "1"], client=client)
        assert len(client.calls) == 1

    def test_all_failed_exits_1(self, track_store):
        client = _FakeClient({n: RuntimeError("x") for n in ("Heartthrob", "Magda", "Gaiser")})
        assert ea.main(["--tracks-tag", "gpt-5"], client=client) == 1

    def test_missing_track_store_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert ea.main(["--tracks-tag", "nope"], client=_FakeClient()) == 1
