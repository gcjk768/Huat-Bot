from huatbot import listener
from huatbot.vault import Vault

CHAT = "-1001/77"


def _vault(tmp_path):
    v = Vault(tmp_path)
    v.data_dir.mkdir(parents=True)
    return v


def _run(update, vault):
    sent, answered = [], []
    cmd = listener.handle(update, vault=vault, token="t", chat_id=CHAT, bot_username="huat_bot",
                          send=lambda tok, chat, text, **kw: sent.append((text, kw.get("reply_markup"))),
                          answer=lambda tok, cid, text="": answered.append(text))
    return cmd, sent, answered


def _msg(text, chat=-1001, thread=77):
    return {"message": {"chat": {"id": chat}, "message_thread_id": thread, "text": text}}


def test_parse_command():
    assert listener.parse_command("/huat", "huat_bot") == "huat"
    assert listener.parse_command("/HuatNext@huat_bot now", "huat_bot") == "huatnext"
    assert listener.parse_command("/huat@other_bot", "huat_bot") is None
    assert listener.parse_command("/ask", "huat_bot") is None
    assert listener.parse_command("huat", "huat_bot") is None


def test_commands_resend_saved_messages_in_own_topic_only(tmp_path):
    vault = _vault(tmp_path)
    assert _run(_msg("/huat"), vault)[1][0][0] == listener.NOTHING_YET
    listener.save_last_messages(vault, ["one", "two"])
    cmd, sent, _ = _run(_msg("/huat"), vault)
    assert cmd == "huat" and [t for t, _ in sent] == ["one", "two"]
    assert sent[0][1] is None and sent[1][1] == listener.BUTTONS  # buttons on the last message only
    assert [t for t, _ in _run(_msg("/huatnext"), vault)[1]] == ["two"]
    assert _run(_msg("/huat", thread=99), vault)[0] is None  # another bot's topic
    assert _run(_msg("/huat", chat=5), vault)[0] is None  # another chat


def test_buttons_are_whitelisted_and_always_answered(tmp_path):
    vault = _vault(tmp_path)
    listener.save_last_messages(vault, ["one", "two"])
    q = {"id": "9", "data": "huatnext", "message": {"chat": {"id": -1001}, "message_thread_id": 77}}
    cmd, sent, answered = _run({"callback_query": q}, vault)
    assert cmd == "huatnext" and len(sent) == 1 and answered == [""]
    cmd, sent, answered = _run({"callback_query": {**q, "data": "run_paid_thing"}}, vault)
    assert cmd is None and not sent and answered == ["Not available here"]
