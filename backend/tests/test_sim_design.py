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
