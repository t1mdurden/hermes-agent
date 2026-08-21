"""The hardline floor must not depend on how the program word is spelled.

`_CMDPOS` (tools/approval.py:463-471) anchors the shutdown/reboot family and
the three `rm` floor rules to a start-of-command position, and its class
requires the BARE program word right after that position. A program written
with a path — `/sbin/shutdown`, `./shutdown`, `/bin/rm` — never satisfies it,
so every `_CMDPOS`-anchored rule stops firing on the path spelling of the exact
command it exists to catch.

Before the basename fold, `detect_hardline_command("/sbin/shutdown -h now")`
returned False and `DANGEROUS_PATTERNS` had no entry for it either, so the
absolute-path spelling of a host shutdown had no guard at all. `/bin/rm -rf /`
still tripped the softer dangerous list, but the floor documented as applying
"BEFORE yolo / mode=off / cron approve-mode so no session-level setting can
bypass it" (approval.py:436-437, :4362-4369) did not hold.

`kill -1` is the control: that rule is written `\\bkill\\s`, not `_CMDPOS`, and
both spellings were always caught. It is the reason the anchor — not the
pattern list — is the cause.

Per SECURITY.md §2.4 / §3.2 the approval gate is an in-process heuristic and
not a security boundary; a denylist over shell strings is structurally
incomplete and this test does not claim otherwise. It asserts that the floor
covers both spellings of the commands it already lists.
"""

import pytest

from tools.approval import (
    _basename_command_words,
    _command_word_basename,
    check_all_command_guards,
    detect_hardline_command,
    disable_session_yolo,
    enable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)


# Each pair is (bare spelling, path spelling). The bare one already worked;
# the path one is the regression.
_SPELLING_PAIRS = [
    ("shutdown -h now", "/sbin/shutdown -h now"),
    ("shutdown -h now", "./shutdown -h now"),
    ("reboot", "/sbin/reboot"),
    ("poweroff", "/sbin/poweroff"),
    ("halt", "/sbin/halt"),
    ("systemctl poweroff", "/usr/bin/systemctl poweroff"),
    ("systemctl reboot", "/bin/systemctl reboot"),
    ("init 0", "/sbin/init 0"),
    ("telinit 6", "/sbin/telinit 6"),
    ("rm -rf /", "/bin/rm -rf /"),
    ("rm -rf /etc", "/bin/rm -rf /etc"),
    ("rm -rf ~", "/usr/bin/rm -rf ~"),
    ("sudo rm -rf /", "sudo /bin/rm -rf /"),
    ("cd /tmp && reboot", "cd /tmp && /sbin/reboot"),
]

# Text that merely CONTAINS a path or a dangerous word but never runs it as a
# command. `_CMDPOS` exists because these used to trip the floor and could not
# run at all; the basename fold must not bring that back. Folding is applied
# only to words the quote-aware tokenizer reports as command-position words,
# so an argument is never rewritten.
_MUST_STAY_ALLOWED = [
    "echo /sbin/reboot",
    "echo 'poweroff'",
    "cat /etc/shutdown.conf",
    "ls /bin/rm",
    "ls -la",
    "cat README.md",
    "git commit -m 'fix rm -rf / doc'",
    'gh pr create --title "block rm -rf / spellings"',
    "grep -rn shutdown /var/log",
    "python3 scripts/reboot_helper.py --dry-run",
    "./scripts/halt_check.sh",
    "/usr/bin/git status",
    "/usr/bin/python3 -m pytest -q",
    "/usr/local/bin/node server.js",
    "docker run --rm -it img /bin/bash",
    "find / -name '*.log'",
    "rsync -a /src /dst",
    "tar -czf /tmp/a.tgz /var/log",
    "systemctl status hermes",
    "rm -rf ./build",
]


@pytest.fixture
def clean_session(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("hardline_abspath_test")
    try:
        disable_session_yolo("hardline_abspath_test")
        yield
    finally:
        disable_session_yolo("hardline_abspath_test")
        reset_current_session_key(token)


# ── the helper, on its own ────────────────────────────────────────────


@pytest.mark.parametrize(
    "word,expected",
    [
        ("/sbin/shutdown", "shutdown"),
        ("./shutdown", "shutdown"),
        ("/usr/bin/systemctl", "systemctl"),
        ('"/bin/rm"', "rm"),
        ("'/bin/rm'", "rm"),
        ("rm", None),            # nothing to fold
        ("shutdown", None),
        ("/", None),             # empty tail
        ("/bin/", None),
        ("/a b/c d", None),      # whitespace in the tail is not a program name
    ],
)
def test_command_word_basename(word, expected):
    assert _command_word_basename(word) == expected


def test_basename_fold_touches_command_words_only():
    assert _basename_command_words("sudo /bin/rm -rf /") == "sudo rm -rf /"
    assert _basename_command_words("cd /tmp && /sbin/reboot") == "cd /tmp && reboot"
    # The path here is an ARGUMENT, so it survives untouched.
    assert _basename_command_words("echo /sbin/reboot") == "echo /sbin/reboot"
    assert _basename_command_words("cat /etc/shutdown.conf") == "cat /etc/shutdown.conf"
    # Nothing to do is a no-op, not a rebuild.
    assert _basename_command_words("ls -la") == "ls -la"


# ── the regression ────────────────────────────────────────────────────


@pytest.mark.parametrize("bare,with_path", _SPELLING_PAIRS)
def test_path_spelling_is_hardline_when_the_bare_spelling_is(bare, with_path):
    assert detect_hardline_command(bare)[0] is True, f"control regressed: {bare!r}"
    assert detect_hardline_command(with_path)[0] is True, (
        f"{with_path!r} evades the floor that catches {bare!r}"
    )


@pytest.mark.parametrize("command", _MUST_STAY_ALLOWED)
def test_data_and_ordinary_commands_are_not_hardline(command):
    is_hardline, description = detect_hardline_command(command)
    assert is_hardline is False, f"false positive on {command!r}: {description}"


def test_kill_all_control_was_never_affected():
    """`kill -1` uses `\\bkill\\s`, not `_CMDPOS` — both spellings always matched.

    It is the control that isolates the anchor as the cause rather than the
    pattern list, and it must keep matching after the fold.
    """
    assert detect_hardline_command("kill -1")[0] is True
    assert detect_hardline_command("/bin/kill -1")[0] is True


# ── the floor's actual contract, end to end ───────────────────────────


def test_path_spelling_blocked_through_the_guard_chain(clean_session):
    for command in ("/sbin/shutdown -h now", "/sbin/reboot", "/bin/rm -rf /"):
        result = check_all_command_guards(command, "local")
        assert result["approved"] is False, f"{command!r} was approved"
        assert result.get("hardline") is True, f"{command!r} did not hit the floor"


def test_yolo_cannot_bypass_the_path_spelling_either(clean_session):
    """The floor is documented as applying before yolo. That has to be true of
    both spellings, or the documented contract holds only for one of them."""
    enable_session_yolo("hardline_abspath_test")
    for command in ("/sbin/shutdown -h now", "/sbin/reboot", "/bin/rm -rf /"):
        result = check_all_command_guards(command, "local")
        assert result["approved"] is False, f"yolo let {command!r} through the floor"
        assert result.get("hardline") is True


def test_yolo_still_approves_ordinary_path_commands(clean_session):
    """The fold must not turn yolo into ask for everyday absolute-path programs."""
    enable_session_yolo("hardline_abspath_test")
    for command in ("/usr/bin/git status", "/usr/local/bin/node server.js", "ls -la"):
        result = check_all_command_guards(command, "local")
        assert result["approved"] is True, f"{command!r} was blocked: {result.get('message')}"
