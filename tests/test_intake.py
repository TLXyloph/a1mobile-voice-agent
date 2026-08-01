"""Intake has to be safe to hand to someone who has never seen a config file.

Three things can go wrong here and each one is worse than a crash.

An owner says something the parser half-understands and the profile silently
holds a number nobody meant - that is the $311 call again, moved upstream. A
price sheet is read and a missing cost becomes zero, at which point every price
clears every margin and the floor is decorative. Or the .env writer, reaching in
to update COST_MATERIALS, rewrites the line above it and takes the OpenAI key
with it.

So: every rejection is asserted to be an instruction rather than an exception,
every missing field is asserted to stay missing, and the credential test uses a
fixture .env stuffed with realistic secrets and asserts them byte-identical
afterwards.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business.campaign import Envelope  # noqa: E402
from src.business.pricing import CostModel  # noqa: E402
from src.mcp import intake_docs, intake_server  # noqa: E402
from src.mcp.intake_profile import (  # noqa: E402
    QUESTIONS,
    Refusal,
    missing_fields,
    next_question,
    to_config,
)

#: A complete bakery, in the words an owner would actually use.
SCRIPT: tuple[tuple[str, str], ...] = (
    ("unit", "muffin"),
    ("items_per_person", "2"),
    ("capacity_period", "weekly"),
    ("capacity_total", "400"),
    ("materials_per_unit", "$0.80"),
    ("labor_per_unit", "0.40"),
    ("transport_basis", "per delivery"),
    ("transport_cost", "18"),
    ("units_per_delivery", "120"),
    ("min_margin_pct", "30"),
    ("target_margin_pct", "45"),
    ("max_discount_pct", "10"),
    ("earliest_date", "2026-08-10"),
    ("latest_date", "2026-11-20"),
    ("blackout_days", "Sunday and Monday"),
    ("approval_mode", "only the ones outside the limits"),
)

#: The typed equivalent, for tests that go straight at `to_config`.
ANSWERS: dict[str, object] = {
    "unit": "muffin",
    "items_per_person": 2,
    "capacity_period": "week",
    "capacity_total": 400,
    "materials_per_unit": Decimal("0.80"),
    "labor_per_unit": Decimal("0.40"),
    "transport_basis": "per_delivery",
    "transport_cost": Decimal("18"),
    "units_per_delivery": 120,
    "min_margin_pct": Decimal("30"),
    "target_margin_pct": Decimal("45"),
    "max_discount_pct": Decimal("10"),
    "earliest_date": date(2026, 8, 10),
    "latest_date": date(2026, 11, 20),
    "blackout_days": ("sunday", "monday"),
    "approval_mode": "out_of_envelope",
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test may touch the real config/.env or the real saved profile."""
    intake_server.reset_session()
    monkeypatch.setattr(intake_server, "PROFILE_PATH", tmp_path / "business_profile.json")
    monkeypatch.setattr(intake_server, "ENV_PATH", tmp_path / ".env")
    yield
    intake_server.reset_session()


def run_script(stop_before: str | None = None) -> list[dict]:
    """Answer the interview, optionally stopping short of one field."""
    intake_server.start_intake("Rosewater Bakehouse", "bakery")
    out = []
    for field, value in SCRIPT:
        if field == stop_before:
            break
        out.append(intake_server.answer(field, value))
    return out


# -- the sequence -----------------------------------------------------------


def test_start_intake_returns_only_the_first_question():
    first = intake_server.start_intake("Rosewater Bakehouse", "bakery")

    assert first["ok"] and not first["done"]
    assert first["field"] == QUESTIONS[0].field == "unit"
    assert "question" in first
    # One question at a time: nothing else is leaked to be asked in a batch.
    assert "questions" not in first and "missing" not in first


def test_start_intake_refuses_a_blank_business_name():
    result = intake_server.start_intake("   ", "bakery")

    assert result["ok"] is False
    assert "instruction" in result


def test_sequence_advances_one_field_at_a_time_and_terminates():
    steps = run_script()

    asked = [s["field"] for s in steps[:-1]]
    # Each answer hands back the *next* field, never repeats the one just given.
    assert asked == [field for field, _ in SCRIPT[1:]]
    assert steps[-1]["done"] is True
    assert next_question(intake_server._session.answers) is None
    assert missing_fields(intake_server._session.answers) == ()


def test_conditional_question_only_appears_when_it_is_relevant():
    intake_server.start_intake("Studio", "agency")
    for field, value in SCRIPT:
        if field == "transport_basis":
            result = intake_server.answer(field, "per unit")
            break
        intake_server.answer(field, value)

    # Per-unit delivery needs no divisor, so the interview skips straight past it.
    assert result["field"] == "transport_cost"
    assert "units_per_delivery" not in missing_fields(intake_server._session.answers)


def test_completion_summary_states_the_items_versus_people_conversion():
    steps = run_script()

    summary = steps[-1]["summary"]
    # The $311 bug, said out loud before anything is saved.
    assert "60 muffin" in summary and "not 30" in summary


# -- refusals are instructions, never exceptions ----------------------------


@pytest.mark.parametrize(
    ("field", "value", "expect"),
    [
        ("unit", "30", "not a number"),
        ("items_per_person", "0", "at least 1"),
        ("capacity_total", "-5", "at least 1"),
        ("capacity_total", "not a number", "whole number"),
        ("materials_per_unit", "-1.00", "cannot be negative"),
        ("labor_per_unit", "banana", "could not read"),
        ("min_margin_pct", "130", "between 0 and 100"),
        ("min_margin_pct", "100", "under 100"),
        ("max_discount_pct", "-4", "between 0 and 100"),
        ("blackout_days", "Blursday", "did not recognise"),
        ("approval_mode", "maybe sometimes", "one of"),
    ],
)
def test_nonsense_values_are_rejected_with_an_instruction(field, value, expect):
    intake_server.start_intake("Rosewater Bakehouse", "bakery")

    result = intake_server.answer(field, value)  # must not raise

    assert result["ok"] is False
    assert expect in result["instruction"].lower()
    # The instruction is the next thing to say, so the question comes back with it.
    assert result["question"]
    assert field not in intake_server._session.answers


def test_an_ambiguous_fraction_of_a_percent_is_asked_about_not_guessed():
    intake_server.start_intake("Rosewater Bakehouse", "bakery")

    rejected = intake_server.answer("min_margin_pct", "0.3")
    assert rejected["ok"] is False
    assert "30 percent" in rejected["instruction"]

    # Saying it explicitly is accepted, because it is no longer ambiguous.
    assert intake_server.answer("min_margin_pct", "0.3%")["ok"] is True
    assert intake_server._session.answers["min_margin_pct"] == Decimal("0.3")


def test_a_target_margin_under_the_floor_is_refused_with_both_numbers():
    intake_server.start_intake("Rosewater Bakehouse", "bakery")
    intake_server.answer("min_margin_pct", "30")

    result = intake_server.answer("target_margin_pct", "20")

    assert result["ok"] is False
    assert "30" in result["instruction"] and "20" in result["instruction"]


def test_an_unknown_field_is_redirected_to_the_open_question():
    intake_server.start_intake("Rosewater Bakehouse", "bakery")

    result = intake_server.answer("profit", "lots")

    assert result["ok"] is False
    assert "unit" in result["instruction"]


def test_answering_before_starting_says_to_start():
    assert intake_server.answer("unit", "muffin")["ok"] is False
    assert intake_server.intake_status()["ok"] is False


# -- status -----------------------------------------------------------------


def test_status_separates_what_is_known_from_what_is_missing():
    run_script(stop_before="min_margin_pct")

    status = intake_server.intake_status()

    assert status["known"]["unit"] == "muffin"
    assert "min_margin_pct" in status["missing"]
    assert status["complete"] is False and status["ready_to_save"] is False
    # Every gap comes with the sentence that fills it.
    assert all(q["question"] for q in status["missing_questions"])


# -- saving -----------------------------------------------------------------


def test_a_partial_profile_cannot_be_saved():
    run_script(stop_before="max_discount_pct")

    result = intake_server.save_profile()

    assert result["ok"] is False and result["saved"] is False
    assert "max_discount_pct" in result["missing"]
    assert not intake_server.PROFILE_PATH.exists()
    assert not intake_server.ENV_PATH.exists()


def test_save_profile_round_trips_through_load_profile():
    run_script()

    saved = intake_server.save_profile()
    assert saved["ok"] and saved["saved"]
    assert intake_server.PROFILE_PATH.exists()

    # A fresh session must not be what makes load work.
    intake_server.reset_session()
    loaded = intake_server.load_profile()

    assert loaded["ok"] is True
    assert loaded["business_name"] == "Rosewater Bakehouse"
    assert loaded["answers"] == saved_answers(saved)
    assert loaded["derived"] == saved["derived"]
    assert loaded["answers"]["blackout_days"] == "sunday,monday"


def saved_answers(saved: dict) -> dict:
    return json.loads(intake_server.PROFILE_PATH.read_text())["answers"]


def test_load_profile_refuses_a_hand_edited_profile_that_no_longer_builds():
    run_script()
    intake_server.save_profile()

    profile = json.loads(intake_server.PROFILE_PATH.read_text())
    profile["answers"]["min_margin_pct"] = "-10"
    intake_server.PROFILE_PATH.write_text(json.dumps(profile))

    result = intake_server.load_profile()

    assert result["ok"] is False
    assert "instruction" in result


def test_load_profile_with_nothing_saved_says_so():
    assert intake_server.load_profile()["ok"] is False


# -- the .env writer --------------------------------------------------------

#: Realistic, and every one of them a line that must come out untouched.
FIXTURE_ENV = """\
# --- Models -------------------------------------------------------------
OPENAI_API_KEY=sk-proj-abc123DEADBEEF
ANTHROPIC_API_KEY=sk-ant-api03-zzz

# --- Telephony ----------------------------------------------------------
TELEPHONY_PROVIDER=livekit
LIVEKIT_API_KEY=APIabc123
LIVEKIT_API_SECRET=super-secret-value
A1MOBILE_TEAM_KEY=team-abc-999
TWILIO_AUTH_TOKEN=auth-token-value
DB_PASSWORD=hunter2

# --- Business -----------------------------------------------------------
CAPACITY_TOTAL=400
COST_MATERIALS=0.80
export MIN_MARGIN_PCT=30
UNRELATED_SETTING=keep-me
"""

SECRET_LINES = (
    "OPENAI_API_KEY=sk-proj-abc123DEADBEEF",
    "ANTHROPIC_API_KEY=sk-ant-api03-zzz",
    "LIVEKIT_API_KEY=APIabc123",
    "LIVEKIT_API_SECRET=super-secret-value",
    "A1MOBILE_TEAM_KEY=team-abc-999",
    "TWILIO_AUTH_TOKEN=auth-token-value",
    "DB_PASSWORD=hunter2",
)


@pytest.fixture()
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(FIXTURE_ENV)
    return path


def test_env_write_updates_keys_in_place_without_duplicating(env_file):
    intake_server.write_env({"CAPACITY_TOTAL": "420", "COST_MATERIALS": "0.82"}, env_file)

    lines = env_file.read_text().splitlines()
    assert lines.count("CAPACITY_TOTAL=420") == 1
    assert "CAPACITY_TOTAL=400" not in lines
    assert sum(line.startswith("COST_MATERIALS=") for line in lines) == 1
    # In place: the key stays where the reader expects it, under its comment.
    assert lines.index("CAPACITY_TOTAL=420") == FIXTURE_ENV.splitlines().index(
        "CAPACITY_TOTAL=400"
    )


def test_env_write_never_touches_a_credential_line(env_file):
    intake_server.write_env(
        {"CAPACITY_TOTAL": "420", "MIN_MARGIN_PCT": "35", "BUSINESS_NAME": "Rosewater"},
        env_file,
    )

    after = env_file.read_text().splitlines()
    for secret in SECRET_LINES:
        assert secret in after, f"credential line was altered or dropped: {secret}"
    assert "super-secret-value" in env_file.read_text()


def test_env_write_refuses_a_credential_shaped_key_outright():
    for key in ("OPENAI_API_KEY", "LIVEKIT_API_SECRET", "DB_PASSWORD", "A1MOBILE_TEAM_KEY"):
        with pytest.raises(RuntimeError, match="credential"):
            intake_server.write_env({key: "x"}, Path("/does/not/matter"))


def test_env_write_keeps_unrelated_keys_comments_and_export_prefixes(env_file):
    intake_server.write_env({"MIN_MARGIN_PCT": "35"}, env_file)

    after = env_file.read_text()
    assert "UNRELATED_SETTING=keep-me" in after
    assert "# --- Telephony" in after
    assert "TELEPHONY_PROVIDER=livekit" in after
    # The line was exported before; rewriting the value must not change that.
    assert "export MIN_MARGIN_PCT=35" in after


def test_env_write_appends_only_keys_that_are_not_already_there(env_file):
    result = intake_server.write_env(
        {"CAPACITY_TOTAL": "420", "ITEMS_PER_PERSON": "2"}, env_file
    )

    assert result["updated_in_place"] == ["CAPACITY_TOTAL"]
    assert result["appended"] == ["ITEMS_PER_PERSON"]
    assert env_file.read_text().count("ITEMS_PER_PERSON=") == 1


def test_env_write_creates_the_file_when_there_is_none(tmp_path):
    path = tmp_path / "fresh" / ".env"

    intake_server.write_env({"CAPACITY_TOTAL": "400"}, path)

    assert "CAPACITY_TOTAL=400" in path.read_text()


def test_save_profile_writes_the_keys_run_call_actually_reads(env_file, monkeypatch):
    monkeypatch.setattr(intake_server, "ENV_PATH", env_file)
    run_script()

    intake_server.save_profile()

    text = env_file.read_text()
    for key in (
        "CAPACITY_TOTAL", "CAPACITY_UNIT", "COST_MATERIALS", "COST_LABOR",
        "COST_TRANSPORT", "MIN_MARGIN_PCT", "TARGET_MARGIN_PCT",
    ):
        assert f"{key}=" in text, f"{key} is read by run_call.py but was not written"
    assert "COST_TRANSPORT=0.15" in text  # 18.00 spread over 120, rounded up
    assert "ITEMS_PER_PERSON=2" in text
    for secret in SECRET_LINES:
        assert secret in text.splitlines()


# -- document parsing -------------------------------------------------------


def write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "menu.csv"
    path.write_text(
        "Item,Price\n"
        "Blueberry muffin,3.50\n"
        "Banana bread,4.00\n"
        "Materials cost per muffin,0.82\n"
        "Labour cost,0.45\n"
        "Weekly capacity,420\n"
    )
    return path


def test_parse_document_reports_what_it_could_not_find(tmp_path):
    result = intake_server.parse_document(str(write_csv(tmp_path)))

    assert result["ok"] is True
    assert result["found"]["materials_per_unit"]["value"] == "0.82"
    # The three it cannot know are named, not defaulted.
    for absent in ("transport_cost", "min_margin_pct", "target_margin_pct"):
        assert absent in result["missing"]
        assert absent not in result["found"]


def test_parse_document_never_defaults_a_missing_cost_to_zero(tmp_path):
    path = tmp_path / "sparse.md"
    path.write_text("# Sheet\nWeekly capacity: 300\n")

    intake_server.start_intake("Sparse Co", "bakery")
    result = intake_server.parse_document(str(path))

    assert "materials_per_unit" not in result["applied"]
    assert "materials_per_unit" in result["still_missing"]
    assert intake_server._session.answers.get("materials_per_unit") is None


def test_parse_document_treats_menu_prices_as_prices_not_costs(tmp_path):
    result = intake_server.parse_document(str(write_csv(tmp_path)))

    prices = {item["item"] for item in result["menu_items"]}
    assert "Blueberry muffin" in prices
    # A $3.50 price read as a $3.50 cost would invert the whole margin sum.
    assert result["found"]["materials_per_unit"]["value"] != "3.50"
    assert any("not costs" in note for note in result["notes"])


def test_parse_document_will_not_choose_between_two_meanings_of_margin(tmp_path):
    path = tmp_path / "vague.txt"
    path.write_text("Margin: 35\n")

    result = intake_server.parse_document(str(path))

    assert "min_margin_pct" not in result["found"]
    assert "target_margin_pct" not in result["found"]
    assert any("margin" in note for note in result["notes"])


def test_parse_document_applies_findings_to_an_open_interview(tmp_path):
    intake_server.start_intake("Golden Crumb", "bakery")

    result = intake_server.parse_document(str(write_csv(tmp_path)))

    assert result["applied"]["capacity_total"]["value"] == 420
    assert result["applied"]["materials_per_unit"]["evidence"]
    assert intake_server.intake_status()["sources"]["capacity_total"].startswith("document:")


def test_parse_document_rejects_a_document_value_it_would_reject_from_a_person(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("Materials cost: -4.00\nMinimum margin: 250\n")
    intake_server.start_intake("Odd Co", "bakery")

    result = intake_server.parse_document(str(path))

    assert "materials_per_unit" in result["rejected"]
    assert "min_margin_pct" in result["rejected"]
    assert result["rejected"]["min_margin_pct"]["instruction"]
    assert intake_server._session.answers == {}


@pytest.mark.parametrize(
    "path", ["", "/tmp/definitely-not-here-9182.csv", "/etc/hosts"]
)
def test_parse_document_refuses_bad_paths_with_an_instruction(path):
    result = intake_server.parse_document(path)

    assert result["ok"] is False and result["instruction"]


def test_parse_document_rejects_a_null_byte_in_the_path():
    with pytest.raises(intake_docs.DocumentError):
        intake_docs.safe_path("menu\x00.csv")


def test_parse_document_reads_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "costs.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    for row in (["Materials cost", 0.9], ["Labour cost", 0.5], ["Max discount", 12]):
        sheet.append(row)
    book.save(path)

    result = intake_server.parse_document(str(path))

    assert result["found"]["materials_per_unit"]["value"] == "0.9"
    assert result["found"]["max_discount_pct"]["value"] == "12"


# -- to_config: the profile becomes the objects the agent runs on -----------


def test_to_config_builds_a_real_cost_model_and_envelope():
    config = to_config(ANSWERS)

    assert isinstance(config.costs, CostModel)
    assert isinstance(config.envelope, Envelope)
    assert config.envelope.problems() == []
    assert config.costs.unit == "muffin"
    assert config.ledger.total == 400
    assert config.ledger.unit == "muffins"
    assert config.items_per_person == 2


def test_to_config_amortises_a_per_delivery_transport_cost_upward():
    config = to_config(ANSWERS)

    # 18.00 over 120 is 0.15 exactly; the rounding is ceiling, never down.
    assert config.costs.transport_per_unit == Decimal("0.15")
    assert config.costs.unit_cost == Decimal("1.35")


def test_to_config_rounds_an_awkward_amortisation_up_not_down():
    answers = dict(ANSWERS, transport_cost=Decimal("10"), units_per_delivery=3)

    # 3.333... must become 3.34, because a cost rounded down is a floor that
    # does not cover it.
    assert to_config(answers).costs.transport_per_unit == Decimal("3.34")


def test_to_config_floor_price_becomes_the_envelope_floor():
    config = to_config(ANSWERS)

    assert config.envelope.min_price == float(config.costs.floor_price(1))
    permitted, _ = config.envelope.permits(price=config.envelope.min_price - 0.01)
    assert permitted is False


def test_to_config_refuses_a_profile_whose_costs_are_all_zero():
    answers = dict(
        ANSWERS,
        materials_per_unit=Decimal("0"),
        labor_per_unit=Decimal("0"),
        transport_cost=Decimal("0"),
    )

    with pytest.raises(ValueError, match="every price look profitable"):
        to_config(answers)


def test_to_config_refuses_an_incomplete_profile():
    answers = {k: v for k, v in ANSWERS.items() if k != "capacity_total"}

    with pytest.raises(ValueError, match="incomplete"):
        to_config(answers)


def test_to_config_lets_the_owning_class_reject_an_inverted_date_window():
    answers = dict(ANSWERS, latest_date=date(2026, 1, 1))

    with pytest.raises(ValueError, match="envelope is not usable"):
        to_config(answers)


def test_to_config_output_is_json_serialisable_for_the_saved_profile():
    assert json.dumps(to_config(ANSWERS).to_dict())


def test_coercers_raise_refusal_not_a_bare_valueerror():
    """The tool layer only catches Refusal, so nothing else may escape a coercer."""
    from src.mcp.intake_profile import BY_FIELD

    for field, bad in (("unit", ""), ("capacity_total", "x"), ("min_margin_pct", "nope")):
        with pytest.raises(Refusal):
            BY_FIELD[field].coerce(bad)


# -- the VoiceOS launch contract --------------------------------------------
#
# VoiceOS starts a server as `python3 /path/to/server.py`, which is not how any
# of the tests above import it. That difference is not cosmetic: run that way,
# sys.path[0] is the file's own directory and every `from src.` import fails, so
# the process dies before the handshake and the host reports nothing useful.
# These two pin the shape their docs specify.

SERVER_FILE = Path(__file__).resolve().parents[1] / "src" / "mcp" / "intake_server.py"


def test_the_server_is_also_named_mcp_as_voiceos_templates_expect():
    assert intake_server.mcp is intake_server.server


async def test_the_file_survives_being_run_directly_from_an_unrelated_cwd(tmp_path):
    """The real thing: spawn `python <abs path>` and complete an MCP handshake."""
    from mcp import ClientSession, StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_FILE)], cwd=str(tmp_path)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or init.serverInfo
            assert info.name == "a1mobile-intake"

            tools = {t.name for t in (await session.list_tools()).tools}
            assert tools == {
                "start_intake", "answer", "intake_status",
                "parse_document", "save_profile", "load_profile",
            }

            result = await session.call_tool(
                "start_intake", {"business_name": "Rosewater", "vertical": "bakery"}
            )
            assert json.loads(result.content[0].text)["field"] == "unit"
