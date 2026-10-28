"""Sim-design tests (Oct 2026): make agents debate instead of echoing seeds.
Each lever is tested without an LLM call. Tests are added alongside the
change they cover, one commit at a time.
"""

import importlib.util
import os


def _load_twitter_runner_module():
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    )
    path = os.path.join(scripts_dir, "run_twitter_simulation.py")
    spec = importlib.util.spec_from_file_location("run_twitter_simulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twitter_available_actions_include_create_comment():
    # Without CREATE_COMMENT, Twitter runs structurally cannot produce
    # comments (every run: 0). The Reddit script already includes it.
    module = _load_twitter_runner_module()
    actions = module.TwitterSimulationRunner.AVAILABLE_ACTIONS
    # A real action enum, not merely the string "CREATE_COMMENT" in file.
    assert module.ActionType.CREATE_COMMENT in actions
    # The previously working actions must still be there (no silent swap).
    for action in (
        module.ActionType.CREATE_POST,
        module.ActionType.QUOTE_POST,
        module.ActionType.REPOST,
        module.ActionType.LIKE_POST,
        module.ActionType.FOLLOW,
        module.ActionType.DO_NOTHING,
    ):
        assert action in actions


import csv

from app.services.oasis_profile_generator import (
    TWITTER_DISCUSSION_GUIDANCE,
    OasisAgentProfile,
    OasisProfileGenerator,
)


def test_twitter_csv_user_char_carries_discussion_guidance(tmp_path):
    # The stock OASIS Twitter system prompt only says "choose some
    # actions", so agents default to do_nothing/like/bare repost. The
    # guidance block in user_char (injected into the system prompt) is
    # the in-repo lever that makes opinionated discussion the default.
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    profile = OasisAgentProfile(
        user_id=0,
        user_name="maya",
        name="Maya Chen",
        bio="29-year-old beginner investor, checks phone at lunch.",
        persona="Cautious, afraid of dumb mistakes, wants plain language.",
    )
    out = tmp_path / "twitter_profiles.csv"
    generator._save_twitter_csv([profile], str(out))
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert "29-year-old beginner investor" in row["user_char"]
    assert "afraid of dumb mistakes" in row["user_char"]
    assert TWITTER_DISCUSSION_GUIDANCE in row["user_char"]
    assert "create_comment" in row["user_char"]
    assert "quote_post" in row["user_char"]
    # Guidance is internal: the public description stays the plain bio.
    assert row["description"] == "29-year-old beginner investor, checks phone at lunch."
    assert "Discussion behavior" not in row["description"]


from app.services.simulation_config_generator import SimulationConfigGenerator


def test_event_config_prompt_requires_debate_seeds():
    # Previous prompt asked for "initial posts" with no stance/question
    # requirement, so seeds were announcements the crowd could only echo.
    # The prompt is a pure function, tested without an LLM call.
    prompt = SimulationConfigGenerator._build_event_config_prompt(
        simulation_requirement="Debate a Telegram stock-alert bot launch.",
        context_truncated="MoveLensAlertBot offers DM alerts and briefings.",
        type_info="- person: Maya Chen, Tom Alvarez",
    )
    assert "4-6" in prompt
    assert "question" in prompt.lower() or "问题" in prompt
    assert "trade-off" in prompt.lower() or "权衡" in prompt
    assert "English" in prompt
    assert "poster_type" in prompt
    # Stance diversity is required, not just "write seeds".
    assert "skeptical" in prompt.lower() or "怀疑" in prompt
    assert "price" in prompt.lower() or "价格" in prompt
    # JSON shape unchanged: the parser downstream still gets its fields.
    assert "hot_topics" in prompt
    assert "narrative_direction" in prompt
    assert "initial_posts" in prompt
    assert "reasoning" in prompt
