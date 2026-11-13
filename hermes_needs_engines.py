#!/usr/bin/env python3
"""Decide whether a Hermes CLI invocation should start the GPU router.

Exit 0 when it should, exit 1 when it should not. ``hermes`` with no
subcommand is a chat. Management commands such as ``config``, ``status``, and
``doctor`` must not start the router. The router loads a GPU only for a
request, and unloads it again after the idle timeout.
"""

from __future__ import annotations

import sys

# Flags that always consume the following argument.
_WITH_VALUE = {
    "-z",
    "--oneshot",
    "--usage-file",
    "--cpu-profile",
    "-m",
    "--model",
    "--provider",
    "--reasoning",
    "-t",
    "--toolsets",
    "--resume",
    "--in",
    "--skills",
}
# Flags that consume the next argument only when it does not look like a flag.
_WITH_OPTIONAL = {"--continue"}
_FLAGS = {
    "--version",
    "-V",
    "--no-restore-cwd",
    "--worktree",
    "--accept-hooks",
    "--yolo",
    "--pass-session-id",
    "--checkpoints",
    "--ignore-user-config",
    "--ignore-rules",
    "--safe-mode",
    "--tui",
    "--native",
    "--cli",
    "--dev",
}
_AGENT_COMMANDS = {"chat", "serve", "acp", "dashboard", "desktop", "gui", "webapp"}
_GATEWAY_RUNS = {"run", "start", "restart", "ensure", "attach"}


def needs_engines(argv: list[str]) -> bool:
    """Return true when this argv starts an agent that will call a model."""
    if "--help" in argv or "-h" in argv or "--version" in argv or "-V" in argv:
        return False
    positionals: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            positionals.extend(argv[index + 1 :])
            break
        if arg in _WITH_VALUE:
            index += 2
            continue
        if arg in _WITH_OPTIONAL:
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if arg in _FLAGS or arg.startswith("-"):
            index += 1
            continue
        positionals.append(arg)
        index += 1
    if not positionals:
        return True
    command = positionals[0]
    if command in _AGENT_COMMANDS:
        return True
    if command == "gateway" and len(positionals) > 1 and positionals[1] in _GATEWAY_RUNS:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return 0 if needs_engines(args) else 1


if __name__ == "__main__":
    sys.exit(main())
