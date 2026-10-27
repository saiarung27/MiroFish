"""Grounding gate tests: a report may only quote/price what the sim
actually produced. Regression for the Telegram E2E report, which
invented quotes and a $10-20/mo price that appear in zero DB rows.
"""

import os
import sqlite3

import pytest

from app.services.report_grounding import (
    normalize_text,
    strip_unverified,
    verify_report,
)
from app.services.zep_tools import ZepToolsService


BEHAVIOR = [
    "I like the free tier a lot, and the alerts come fast.",
    "Maya Chen says she checks her phone at lunch.",
]
BRIEF = [
    "Free tier covers 5 tickers. Paid tier adds unlimited tickers "
    "and instant push alerts.",
]


def test_behavior_quote_verified():
    md = '> "I like the free tier a lot, and the alerts come fast."\n'
    v = verify_report(md, BEHAVIOR, BRIEF)
    assert v.quotes_checked == 1 and v.quotes_verified == 1
    assert v.claims[0].source == "behavior"


def test_quote_with_sentence_final_period_still_matches():
    # Report quotes end in "." inside the quotes; the post does not.
    md = '> "I like the free tier a lot."\n'
    v = verify_report(md, ["I like the free tier a lot"], BRIEF)
    assert v.quotes_verified == 1


def test_brief_only_quote_with_user_attribution_is_unverified():
    # A user-attributed claim may NOT ground in the brief alone: the
    # brief is the input, not what agents did.
    md = 'Users said "paid tier adds unlimited tickers."\n'
    v = verify_report(md, BEHAVIOR, BRIEF)
    assert v.claims[0].status == "unverified"


def test_brief_only_quote_without_attribution_is_verified():
    md = '> "Free tier covers 5 tickers."\n'
    v = verify_report(md, BEHAVIOR, BRIEF)
    assert v.claims[0].status == "verified"
    assert v.claims[0].source == "brief"


def test_invented_quote_stripped():
    md = '> "Users would pay $10-20 per month for this."\n\nReal paragraph.\n'
    v = verify_report(md, BEHAVIOR, BRIEF)
    clean = strip_unverified(md, v)
    assert "$10-20" not in clean
    assert "Users would pay" not in clean
    assert "Real paragraph." in clean
    assert v.verdict == "ungrounded"


def test_invented_user_price_sentence_dropped():
    md = "Users would pay $99 per month for this.\n\nFree tier is nice.\n"
    v = verify_report(md, BEHAVIOR, BRIEF)
    clean = strip_unverified(md, v)
    assert "$99" not in clean
    assert "Free tier is nice." in clean


def test_behavior_number_kept_and_brief_descriptive_number_kept():
    md = (
        "One user wrote: \"I would pay $5 a month.\"\n"
        "Free tier covers 5 tickers.\n"
    )
    v = verify_report(
        md, ["I would pay $5 a month"], BRIEF
    )
    clean = strip_unverified(md, v)
    assert "$5" in clean
    assert v.verdict == "grounded"


def test_verdict_grounded_after_stripping():
    md = (
        '> "I like the free tier a lot."\n\n'
        '> "Users would pay $10-20 per month."\n'
    )
    v = verify_report(md, BEHAVIOR, BRIEF)
    clean = strip_unverified(md, v)
    assert "I like the free tier a lot." in clean
    assert "$10-20" not in clean
    assert v.verdict == "grounded_after_stripping"


def test_normalize_unifies_quotes_and_case():
    assert normalize_text("“Hello, World!”") == normalize_text('"hello world"')


def _make_sim_db(tmp_path, sim_id):
    sim_dir = tmp_path / "simulations" / sim_id
    sim_dir.mkdir(parents=True)
    db = sim_dir / "twitter_simulation.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE user (user_id INTEGER, agent_id TEXT, user_name TEXT,"
        " name TEXT, bio TEXT, created_at TEXT, num_followings INTEGER,"
        " num_followers INTEGER)"
    )
    conn.execute(
        "CREATE TABLE post (post_id INTEGER, user_id INTEGER,"
        " original_post_id INTEGER, content TEXT, quote_content TEXT,"
        " created_at TEXT, num_likes INTEGER, num_dislikes INTEGER,"
        " num_shares INTEGER, num_reports INTEGER)"
    )
    conn.execute(
        "CREATE TABLE comment (comment_id INTEGER, post_id INTEGER,"
        " user_id INTEGER, content TEXT, created_at TEXT,"
        " num_likes INTEGER, num_dislikes INTEGER)"
    )
    conn.execute(
        "INSERT INTO user VALUES (1, 'a1', 'maya', 'Maya Chen', '', '', 0, 0)"
    )
    conn.execute(
        "INSERT INTO post VALUES (1, 1, NULL, 'I like the free tier a lot.',"
        " '', '', 0, 0, 0, 0)"
    )
    conn.execute(
        "INSERT INTO comment VALUES (1, 1, 1, 'Same here, alerts are fast.',"
        " '', 0, 0)"
    )
    conn.commit()
    conn.close()
    return tmp_path / "simulations"


@pytest.fixture()
def svc_reify(tmp_path, monkeypatch):
    # Real constructor path under the Reify backend (no Zep client,
    # no network): get_sim_posts only reads the local platform DB.
    monkeypatch.setenv("MIROFISH_MEMORY_BACKEND", "reify")
    monkeypatch.setenv("MIROFISH_REIFY_ROOT", str(tmp_path / "graphs"))
    return ZepToolsService()


def test_sim_posts_reads_tmp_db(tmp_path, svc_reify):
    base = _make_sim_db(tmp_path, "sim_test123")
    svc = svc_reify
    result = svc.get_sim_posts(
        simulation_id="sim_test123", base_dir=str(base)
    )
    assert len(result.posts) == 1
    assert result.posts[0]["author"] == "Maya Chen"
    assert result.posts[0]["content"] == "I like the free tier a lot."
    assert len(result.comments) == 1
    assert "Same here" in result.texts()[1]


def test_sim_posts_missing_db_is_empty_not_error(tmp_path, svc_reify):
    svc = svc_reify
    result = svc.get_sim_posts(
        simulation_id="sim_nope", base_dir=str(tmp_path)
    )
    assert result.posts == [] and result.comments == []


def test_sim_posts_rejects_path_traversal(tmp_path, svc_reify):
    svc = svc_reify
    result = svc.get_sim_posts(
        simulation_id="../../etc", base_dir=str(tmp_path)
    )
    assert result.posts == [] and result.comments == []


def test_chinese_money_prose_is_gated():
    # The old report's exact failure mode: "每月10至20美元" prose with
    # no $ sign and no quote. Must be extracted, attributed, and --
    # since it is in neither behavior nor brief -- dropped.
    md = "用户普遍认为每月10至20美元的价格是可以接受的。\n"
    v = verify_report(md, BEHAVIOR, BRIEF)
    assert v.numbers_checked == 1
    assert v.claims[0].status == "unverified"
    clean = strip_unverified(md, v)
    assert "10至20美元" not in clean


def test_chinese_money_in_behavior_verifies():
    md = "用户说10美元一个月是合理的。\n"
    v = verify_report(md, ["10美元一个月是合理的"], BRIEF)
    assert v.numbers_verified == 1
