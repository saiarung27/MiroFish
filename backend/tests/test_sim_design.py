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


from app.services.report_agent import ReportAgent


def _parser():
    # _parse_tool_calls only uses the VALID_TOOL_NAMES class attribute
    # (via _is_valid_tool_call), so no LLM/graph init is needed.
    return ReportAgent.__new__(ReportAgent)


def test_tool_call_parser_validates_xml_format_calls():
    agent = _parser()
    # "tool"/"params" keys normalize to "name"/"parameters".
    calls = agent._parse_tool_calls(
        'Action: I will call a tool.\n'
        '<tool_call>{"tool": "sim_posts", "params": {"limit": 5}}</tool_call>'
    )
    assert calls == [{"name": "sim_posts", "parameters": {"limit": 5}}]
    # Standard "name" calls still pass through unchanged.
    calls = agent._parse_tool_calls(
        '<tool_call>{"name": "sim_posts", "parameters": {"limit": 5}}</tool_call>'
    )
    assert calls == [{"name": "sim_posts", "parameters": {"limit": 5}}]


def test_tool_call_parser_drops_nameless_or_unknown_xml_calls():
    agent = _parser()
    # A nameless object must never reach the section loop's call['name']
    # indexing (that KeyError killed report_b29df4d49c29 outright).
    assert agent._parse_tool_calls(
        '<tool_call>{"parameters": {"limit": 5}}</tool_call>'
    ) == []
    assert agent._parse_tool_calls(
        '<tool_call>{"name": "not_a_tool", "parameters": {}}</tool_call>'
    ) == []


def test_group_initial_actions_keeps_every_seed_for_one_agent():
    # Regression for the Oct 2026 rerun: all 6 seeds were assigned to
    # agent 1 and the old dict assignment kept only the last one, so
    # the 5 debate seeds never reached the feed.
    module = _load_twitter_runner_module()
    agents = {}

    def get_agent(agent_id):
        if agent_id not in agents:
            agents[agent_id] = object()
        return agents[agent_id]

    posts = [
        {"poster_agent_id": 1, "content": "seed one"},
        {"poster_agent_id": 1, "content": "seed two"},
        {"poster_agent_id": 1, "content": "seed three"},
        {"poster_agent_id": 2, "content": "other agent seed"},
    ]
    actions, skipped = module.group_initial_actions(posts, get_agent)
    assert skipped == []
    assert len(actions) == 2
    shared = actions[agents[1]]
    assert isinstance(shared, list) and len(shared) == 3
    assert [a.action_args["content"] for a in shared] == [
        "seed one",
        "seed two",
        "seed three",
    ]
    solo = actions[agents[2]]
    assert not isinstance(solo, list)
    assert solo.action_args["content"] == "other agent seed"


def test_group_initial_actions_skips_unresolvable_agents():
    module = _load_twitter_runner_module()

    def get_agent(agent_id):
        raise ValueError("no such agent")

    actions, skipped = module.group_initial_actions(
        [{"poster_agent_id": 9, "content": "x"}], get_agent
    )
    assert actions == {}
    assert len(skipped) == 1 and skipped[0][0] == 9


def test_event_config_prompt_requires_type_token_poster_type():
    # The Oct 2026 rerun's LLM copied example NAMES (Aisha Khan, Market
    # Pulse Alerts) into poster_type instead of type tokens, so every
    # seed fell back to the same highest-influence agent.
    prompt = SimulationConfigGenerator._build_event_config_prompt(
        simulation_requirement="Debate a Telegram stock-alert bot launch.",
        context_truncated="MoveLensAlertBot offers DM alerts and briefings.",
        type_info="- PriceSensitiveSaver: Aisha Khan",
    )
    assert "never a person's name" in prompt
    assert "never a product name" in prompt
