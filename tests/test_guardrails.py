"""Tests for graft.guardrails — Bash command safety hooks."""

from __future__ import annotations

import pytest

from graft.guardrails import _check_command, _extract_command, bash_guardrail

# ---------------------------------------------------------------------------
# _extract_command
# ---------------------------------------------------------------------------


class TestExtractCommand:
    def test_normal_input(self):
        assert _extract_command({"command": "ls -la"}) == "ls -la"

    def test_missing_command(self):
        assert _extract_command({}) is None

    def test_non_string_command(self):
        assert _extract_command({"command": 123}) is None

    def test_empty_string(self):
        assert _extract_command({"command": ""}) == ""


# ---------------------------------------------------------------------------
# _check_command — find patterns
# ---------------------------------------------------------------------------


class TestCheckCommandFind:
    """Test detection of dangerous find commands."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "find / -name base.py",
            "find / -type f -name '*.py'",
            "find /  -name foo",
            "find / ",
            "find /",
        ],
    )
    def test_find_root_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None
        assert "project directory" in result.lower() or "filesystem" in result.lower()

    @pytest.mark.parametrize(
        "cmd",
        [
            "find /usr/local -name python",
            "find /opt -type f",
            "find /var/log -name '*.log'",
            "find /etc -name 'hosts'",
            "find /Users -name '.env'",
            "find /home -type d",
            "find /Library -name 'Preferences'",
            "find /Applications -name '*.app'",
            "find /System -name 'kernel'",
        ],
    )
    def test_find_system_dirs_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None
        assert (
            "system directories" in result.lower()
            or "project directory" in result.lower()
        )

    @pytest.mark.parametrize(
        "cmd",
        [
            "find . -name base.py",
            "find ./src -name '*.py'",
            "find src -type f -name '*.ts'",
            "find backend/app -name models.py",
            "find . -type d -name __pycache__",
            # absolute but specific project path — should be allowed
            "find /Users/kevin/projects/my-app -name 'config.py'",
            "find -name base.py",  # no path = cwd, safe
            "find -type f -name '*.py'",  # no path = cwd, safe
            "find -maxdepth 3 -name x",  # no path = cwd, safe
        ],
    )
    def test_find_project_dir_allowed(self, cmd: str):
        assert _check_command(cmd) is None

    def test_find_chained_after_semicolon(self):
        result = _check_command("echo hello; find / -name foo")
        assert result is not None

    def test_find_chained_after_pipe(self):
        result = _check_command("echo hello | find / -name foo")
        assert result is not None

    def test_find_chained_after_and(self):
        result = _check_command("true && find / -name foo")
        assert result is not None

    @pytest.mark.parametrize(
        "cmd",
        [
            "sudo find / -name base.py",
            "sudo find /usr -name python",
        ],
    )
    def test_sudo_find_still_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None


# ---------------------------------------------------------------------------
# _check_command — destructive patterns
# ---------------------------------------------------------------------------


class TestCheckCommandDestructive:
    """Test detection of catastrophic commands."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf / ",
            "rm -r -f /",
            "sudo rm -rf /",
        ],
    )
    def test_rm_root_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None
        assert "not allowed" in result.lower()

    def test_rm_project_dir_allowed(self):
        assert _check_command("rm -rf ./dist") is None
        assert _check_command("rm -rf node_modules") is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "dd if=/dev/zero of=/dev/sda",
            "dd of=/dev/disk0 if=image.iso",
        ],
    )
    def test_dd_device_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None
        assert "not allowed" in result.lower()

    def test_dd_file_allowed(self):
        assert _check_command("dd if=/dev/zero of=./test.img bs=1M count=1") is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "mkfs.ext4 /dev/sda1",
            "mkfs -t ext4 /dev/sda1",
        ],
    )
    def test_mkfs_denied(self, cmd: str):
        result = _check_command(cmd)
        assert result is not None


# ---------------------------------------------------------------------------
# _check_command — safe commands
# ---------------------------------------------------------------------------


class TestCheckCommandSafe:
    """Verify normal commands pass through."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls -la",
            "npm install",
            "pip install -r requirements.txt",
            "python -m pytest",
            "cat README.md",
            "grep -r 'import' src/",
            "git status",
            "docker-compose up -d",
            "curl https://api.example.com",
            "wc -l src/**/*.py",
            "npm run build",
            "uvicorn app.main:app --reload",
            "grep -r find src/",  # word "find" in non-find command
            "cat findme.txt",  # word "find" in filename
            "echo 'find / -name x'",  # find inside a quoted string (echo)
        ],
    )
    def test_normal_commands_allowed(self, cmd: str):
        assert _check_command(cmd) is None


# ---------------------------------------------------------------------------
# bash_guardrail async hook
# ---------------------------------------------------------------------------


class TestBashGuardrailHook:
    """Test the full async hook interface."""

    @pytest.mark.asyncio
    async def test_allows_safe_command(self):
        hook_input = {
            "hook_event_name": "PreToolUse",
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
            "tool_use_id": "test-123",
        }
        result = await bash_guardrail(hook_input, "test-123", {"signal": None})
        specific = result["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreToolUse"
        assert "permissionDecision" not in specific

    @pytest.mark.asyncio
    async def test_denies_find_root(self):
        hook_input = {
            "hook_event_name": "PreToolUse",
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {"command": "find / -name base.py"},
            "tool_use_id": "test-123",
        }
        result = await bash_guardrail(hook_input, "test-123", {"signal": None})
        specific = result["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreToolUse"
        assert specific["permissionDecision"] == "deny"
        assert "project directory" in specific["permissionDecisionReason"].lower()

    @pytest.mark.asyncio
    async def test_handles_missing_command(self):
        hook_input = {
            "hook_event_name": "PreToolUse",
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp",
            "tool_name": "Bash",
            "tool_input": {},
            "tool_use_id": "test-123",
        }
        result = await bash_guardrail(hook_input, "test-123", {"signal": None})
        specific = result["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreToolUse"
        assert "permissionDecision" not in specific
