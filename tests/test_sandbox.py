"""Tests for unified sandbox — READ-ONLY, verify behavior, report bugs."""
from __future__ import annotations

import re
from pathlib import Path

import pytest


# ==================================================================
# Sandbox core
# ==================================================================

class TestSandboxResolve:
    def test_relative_under_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir(); (r / "f.txt").write_text("hi")
        sb = Sandbox([r], [])
        assert sb.resolve("f.txt") == r / "f.txt"

    def test_relative_not_exists_falls_back(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        sb = Sandbox([r], [])
        assert sb.resolve("nope.txt") == r / "nope.txt"

    def test_absolute_preserved(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        f = tmp_path / "other.txt"; f.write_text("x")
        sb = Sandbox([r], [])
        assert sb.resolve(str(f)) == f.resolve()

    def test_no_roots_cwd_fallback(self):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([], [])
        assert sb.resolve("t.txt") == Path("t.txt").resolve()

    def test_multi_root_first_existing_wins(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r1 = tmp_path / "a"; r1.mkdir()
        r2 = tmp_path / "b"; r2.mkdir(); (r2 / "u.txt").write_text("y")
        sb = Sandbox([r1, r2], [])
        assert sb.resolve("u.txt") == r2 / "u.txt"


class TestSandboxCheckPath:
    def test_under_root_ok(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir(); f = r / "ok.txt"; f.write_text("x")
        sb = Sandbox([r], [])
        assert sb.check_path(f) is None

    def test_outside_root_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        o = tmp_path / "bad.txt"; o.write_text("x")
        sb = Sandbox([r], [])
        err = sb.check_path(o)
        assert err and "outside" in err.lower()

    def test_under_root_but_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        env = r / ".env"; env.write_text("S=1")
        sb = Sandbox([r], ["**/.env"])
        err = sb.check_path(env)
        assert err and "blocked" in err.lower()

    def test_deep_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        d = r / "x" / ".git" / "config"; d.parent.mkdir(parents=True); d.write_text("x")
        sb = Sandbox([r], ["**/.git/**"])
        assert sb.check_path(d) is not None

    def test_no_roots_allow_any_except_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        f = tmp_path / "a.txt"; f.write_text("x")
        sb = Sandbox([], [])
        assert sb.check_path(f) is None

    def test_no_roots_still_blocks_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        env = tmp_path / ".env"; env.write_text("x")
        sb = Sandbox([], ["**/.env"])
        assert sb.check_path(env) is not None

    def test_root_exact_allowed(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        sb = Sandbox([r], [])
        assert sb.check_path(r) is None


class TestCheckBashPath:
    def test_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([tmp_path], ["**/.env"])
        assert sb.check_bash_path("/p/.env") is not None

    def test_normal_ok(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([tmp_path], ["**/.env"])
        assert sb.check_bash_path("/tmp/data.txt") is None


class TestIsUnderRoot:
    def test_under(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        sb = Sandbox([r], [])
        assert sb.is_under_root(str(r / "f.txt")) is True

    def test_outside(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        sb = Sandbox([r], [])
        assert sb.is_under_root(str(tmp_path / "out.txt")) is False

    def test_exact_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        r = tmp_path / "ws"; r.mkdir()
        sb = Sandbox([r], [])
        assert sb.is_under_root(str(r)) is True

    def test_no_roots(self):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([], [])
        assert sb.is_under_root("/x") is False

    def test_sibling_not_confused(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        d = tmp_path / "Desktop"; d.mkdir()
        dp = tmp_path / "DesktopProjects"; dp.mkdir()
        sb = Sandbox([d], [])
        assert sb.is_under_root(str(dp / "f.txt")) is False


# ==================================================================
# _glob_match (fnmatch — check_path uses full paths always)
# ==================================================================

class TestGlobMatch:
    def test_full_path_env(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match("C:/Users/MR/Desktop/.env", "**/.env") is True

    def test_bare_no_match_starstar(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match(".env", "**/.env") is False

    def test_exact(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match(".env", ".env") is True

    def test_deep_git(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match("a/b/c/.git/config", "**/.git/**") is True

    def test_id_rsa_star(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match("C:/Users/MR/Desktop/id_rsa", "**/id_rsa*") is True
        assert _glob_match("C:/Users/MR/Desktop/id_rsa.pub", "**/id_rsa*") is True

    def test_non_match(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match("C:/Users/MR/Desktop/notes.txt", "**/.env") is False

    def test_windows_backslash(self):
        from personal_agent.tools.sandbox import _glob_match
        assert _glob_match(r"C:\Users\MR\Desktop\.env", "**/.env") is True


# ==================================================================
# Bash path sandbox
# ==================================================================

class TestBashPathSandbox:
    def test_blocked_env(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], ["**/.env", "**/config.yaml"])
        set_restrict_paths(True)
        assert _check_path_sandbox("cat .env") is not None

    def test_blocked_config(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], ["**/config.yaml"])
        set_restrict_paths(True)
        assert _check_path_sandbox("cat config.yaml") is not None

    def test_blocked_even_restrict_off(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], ["**/.env"])
        set_restrict_paths(False)
        assert _check_path_sandbox("cat .env") is not None

    def test_restrict_off_allows_etc(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], ["**/.env"])
        set_restrict_paths(False)
        assert _check_path_sandbox("cat /etc/passwd") is None

    def test_unix_system_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        for p in ["/etc/passwd", "/var/log", "/proc/cpuinfo", "/usr/bin/env"]:
            assert _check_path_sandbox(f"cat {p}") is not None, f"Should block: {p}"

    def test_windows_system_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        assert _check_path_sandbox(r"type C:\Windows\System32\hosts") is not None

    def test_tilde_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        assert _check_path_sandbox("cat ~/.ssh/id_rsa") is not None

    def test_parent_traversal_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        assert _check_path_sandbox("cat ../../secret.txt") is not None
        assert _check_path_sandbox("ls ../") is not None

    def test_relative_paths_ok(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        for cmd in ["cat notes.txt", "ls ./", "python s.py", "echo hi"]:
            assert _check_path_sandbox(cmd) is None, f"Should allow: {cmd}"

    def test_root_in_command_ok(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        assert _check_path_sandbox(f"cat {tmp_path}/f.txt") is None

    def test_simple_commands_ok(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        init_sandbox([tmp_path], [])
        set_restrict_paths(True)
        for cmd in ["ls", "pwd", "date", "whoami"]:
            assert _check_path_sandbox(cmd) is None, f"Should allow: {cmd}"

    def test_sibling_dir_substring_exposure(self, tmp_path: Path):
        """BUG: root substring check can falsely allow sibling dirs.
        Root 'Desktop' substring-matches 'DesktopProjects', so
        cat DesktopProjects/secret.txt may bypass if not caught by escape patterns."""
        from personal_agent.tools.sandbox import init_sandbox
        from personal_agent.tools.builtin.bash import _check_path_sandbox, set_restrict_paths
        d = tmp_path / "Desktop"; d.mkdir()
        dp = tmp_path / "DesktopProjects"; dp.mkdir()
        init_sandbox([d], [])
        set_restrict_paths(True)
        result = _check_path_sandbox(f"cat {dp}/secret.txt")
        # Current behavior: Desktop is substring of DesktopProjects → passes step 3
        # Whether this is blocked depends on escape patterns catching it
        # Document actual behavior:
        has_substr = str(d) in str(dp)
        assert has_substr, f"Test precondition: {d} should be substring of {dp}"
        # Known gap: if substring matches and escape patterns don't catch it, path slides through


# ==================================================================
# _glob_pattern_to_regex (bash command scanning)
# ==================================================================

class TestBashGlobToRegex:
    """Tests for _glob_pattern_to_regex — converts blocked glob to regex.

    BUG #1: replace('**/', '') strips leading **/ but trailing '/**'
    is not caught because pattern is '/**' not '**/'. Result: '.git/**'
    becomes regex that matches literal '.git/**' not '.git'.

    BUG #2: re.escape escapes * so glob * becomes literal * in regex.
    Should convert * to .* after escaping.
    """

    def test_dotenv(self):
        from personal_agent.tools.builtin.bash import _glob_pattern_to_regex
        r = _glob_pattern_to_regex("**/.env")
        # **/.env → .env → regex \.env (works for simple case)
        assert re.search(r, ".env") is not None
        assert re.search(r, "path/to/.env") is not None

    def test_git_trailing_starstar(self):
        from personal_agent.tools.builtin.bash import _glob_pattern_to_regex
        r = _glob_pattern_to_regex("**/.git/**")
        assert re.search(r, "path/.git/file") is not None
        assert re.search(r, ".git/") is not None   # fixed: trailing → /, matches directory refs

    def test_ssh_trailing_starstar(self):
        from personal_agent.tools.builtin.bash import _glob_pattern_to_regex
        r = _glob_pattern_to_regex("**/.ssh/**")
        assert re.search(r, "path/.ssh/config") is not None
        assert re.search(r, ".ssh/") is not None   # fixed: trailing → /, matches directory refs

    def test_id_rsa_wildcard(self):
        from personal_agent.tools.builtin.bash import _glob_pattern_to_regex
        r = _glob_pattern_to_regex("**/id_rsa*")
        assert re.search(r, "id_rsa") is not None       # fixed: *→wildcard, matches bare name
        assert re.search(r, "id_rsa.pub") is not None   # wildcard matches extension

    def test_config_yaml(self):
        from personal_agent.tools.builtin.bash import _glob_pattern_to_regex
        r = _glob_pattern_to_regex("**/config.yaml")
        assert re.search(r, "config.yaml") is not None


# ==================================================================
# Singleton lifecycle
# ==================================================================

class TestSingleton:
    def test_init_then_get(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox, get_sandbox
        sb = init_sandbox([tmp_path], ["**/.env"])
        assert get_sandbox() is sb

    def test_init_overwrites(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox, get_sandbox
        r1 = tmp_path / "a"; r1.mkdir()
        r2 = tmp_path / "b"; r2.mkdir()
        a = init_sandbox([r1], [])
        b = init_sandbox([r2], ["**/.env"])
        assert a is not b
        assert get_sandbox() is b


# ==================================================================
# File tool integration (async)
# ==================================================================

class TestFileReadIntegration:
    @pytest.mark.asyncio
    async def test_read_normal(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], [])
        f = tmp_path / "hello.txt"; f.write_text("world")
        from personal_agent.tools.builtin.file_read import _file_read
        r = await _file_read(str(f))
        assert "world" in r

    @pytest.mark.asyncio
    async def test_read_blocked_env(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], ["**/.env"])
        env = tmp_path / ".env"; env.write_text("X")
        from personal_agent.tools.builtin.file_read import _file_read
        r = await _file_read(str(env))
        assert "blocked" in r.lower()

    @pytest.mark.asyncio
    async def test_read_outside_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        ws = tmp_path / "ws"; ws.mkdir()
        o = tmp_path / "s.txt"; o.write_text("x")
        init_sandbox([ws], [])
        from personal_agent.tools.builtin.file_read import _file_read
        r = await _file_read(str(o))
        assert "outside" in r.lower()


class TestFileWriteIntegration:
    @pytest.mark.asyncio
    async def test_write_normal(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], [])
        from personal_agent.tools.builtin.file_write import _file_write
        r = await _file_write("out.txt", "data")
        assert "Written" in r
        assert (tmp_path / "out.txt").exists()

    @pytest.mark.asyncio
    async def test_write_blocked_env(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], ["**/.env"])
        from personal_agent.tools.builtin.file_write import _file_write
        r = await _file_write(".env", "X")
        assert "blocked" in r.lower()


class TestGlobIntegration:
    @pytest.mark.asyncio
    async def test_glob_in_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], [])
        (tmp_path / "a.py").write_text("x"); (tmp_path / "b.py").write_text("y")
        from personal_agent.tools.builtin.glob_tool import _glob
        r = await _glob("*.py", str(tmp_path))
        assert "a.py" in r and "b.py" in r

    @pytest.mark.asyncio
    async def test_glob_outside_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        ws = tmp_path / "ws"; ws.mkdir()
        o = tmp_path / "o.py"; o.write_text("x")
        init_sandbox([ws], [])
        from personal_agent.tools.builtin.glob_tool import _glob
        r = await _glob("*.py", str(o.parent))
        assert "outside" in r.lower() or "Error" in r


class TestGrepIntegration:
    @pytest.mark.asyncio
    async def test_grep_in_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], [])
        (tmp_path / "c.py").write_text("def fn(): pass")
        from personal_agent.tools.builtin.grep_tool import _grep
        r = await _grep("def", str(tmp_path))
        assert "fn" in r

    @pytest.mark.asyncio
    async def test_grep_outside_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        ws = tmp_path / "ws"; ws.mkdir()
        o = tmp_path / "d.txt"; o.write_text("x")
        init_sandbox([ws], [])
        from personal_agent.tools.builtin.grep_tool import _grep
        r = await _grep("x", str(o.parent))
        assert "outside" in r.lower() or "Error" in r


class TestFileEditIntegration:
    @pytest.mark.asyncio
    async def test_edit_normal(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        init_sandbox([tmp_path], [])
        f = tmp_path / "notes.md"; f.write_text("# T")
        from personal_agent.tools.builtin.file_edit import _file_edit
        r = await _file_edit("append", str(f), content="\nmore")
        assert "Appended" in r

    @pytest.mark.asyncio
    async def test_edit_outside_root(self, tmp_path: Path):
        from personal_agent.tools.sandbox import init_sandbox
        ws = tmp_path / "ws"; ws.mkdir()
        o = tmp_path / "f.md"; o.write_text("x")
        init_sandbox([ws], [])
        from personal_agent.tools.builtin.file_edit import _file_edit
        r = await _file_edit("append", str(o), content="more")
        assert "outside" in r.lower()


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_all_blocked_patterns(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        blocked = [
            "**/.env", "**/.env.*", "**/.git/**", "**/.ssh/**",
            "**/id_rsa*", "**/.netrc", "**/config.yaml",
            "**/pyproject.toml", "**/audit.log",
        ]
        sb = Sandbox([tmp_path], blocked)
        bad = [".env", ".env.prod", ".git/config", ".ssh/id_rsa",
               ".netrc", "config.yaml", "pyproject.toml", "audit.log"]
        for name in bad:
            p = tmp_path / name
            if "/" in name: p.parent.mkdir(parents=True, exist_ok=True)
            p.parent.mkdir(parents=True, exist_ok=True); p.touch()
            assert sb.check_path(p) is not None, f"Should block: {name}"

    def test_normal_files_not_blocked(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([tmp_path], ["**/.env", "**/.git/**", "**/.ssh/**"])
        for name in ["notes.txt", "script.py", "data.json", "README.md", "src/main.py"]:
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True); p.touch()
            assert sb.check_path(p) is None, f"Should not block: {name}"

    def test_env_star_matches_dotfiles(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        sb = Sandbox([tmp_path], ["**/.env.*"])
        for name in [".env.prod", ".env.local"]:
            p = tmp_path / name; p.touch()
            assert sb.check_path(p) is not None, f"Should block: {name}"

    def test_sandbox_root_sibling(self, tmp_path: Path):
        from personal_agent.tools.sandbox import Sandbox
        d = tmp_path / "Desktop"; d.mkdir()
        dp = tmp_path / "DesktopProjects"; dp.mkdir()
        sb = Sandbox([d], [])
        assert sb.is_under_root(str(d / "f.txt")) is True
        assert sb.is_under_root(str(dp / "f.txt")) is False


# ==================================================================
# Config parsing
# ==================================================================

class TestConfigParsing:
    def test_roots_list(self):
        import yaml
        cfg = yaml.safe_load("sandbox:\n  roots:\n    - /a\n    - /b")
        sb = cfg["sandbox"]
        assert isinstance(sb["roots"], list) and len(sb["roots"]) == 2

    def test_roots_default(self):
        import yaml
        cfg = yaml.safe_load("storage:\n  data_dir: ./d")
        sb = cfg.get("sandbox", {})
        assert sb.get("roots", ["./data"]) == ["./data"]
        assert sb.get("blocked", []) == []
        assert sb.get("bash_restrict_paths", True) is True

    def test_blocked_populated(self):
        import yaml
        cfg = yaml.safe_load("sandbox:\n  blocked:\n    - '**/.env'\n    - '**/.git/**'")
        assert len(cfg["sandbox"]["blocked"]) == 2
