"""Command parser for the TUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class CommandType(Enum):
    """Types of commands."""

    INGEST = auto()
    QUERY = auto()
    STATUS = auto()
    LIST = auto()
    CONFIG = auto()
    DELETE = auto()
    INIT = auto()
    HELP = auto()
    QUIT = auto()
    CLEAR = auto()
    UNKNOWN = auto()


@dataclass
class ParsedCommand:
    """A parsed command with its arguments."""

    type: CommandType
    args: list[str]
    raw: str
    flags: dict[str, str | bool]


class CommandParser:
    """Parse user input into commands."""

    # Command aliases
    ALIASES: dict[str, CommandType] = {
        # Explicit commands
        "ingest": CommandType.INGEST,
        "add": CommandType.INGEST,
        "index": CommandType.INGEST,
        "query": CommandType.QUERY,
        "ask": CommandType.QUERY,
        "search": CommandType.QUERY,
        "status": CommandType.STATUS,
        "stats": CommandType.STATUS,
        "info": CommandType.STATUS,
        "list": CommandType.LIST,
        "ls": CommandType.LIST,
        "sources": CommandType.LIST,
        "config": CommandType.CONFIG,
        "settings": CommandType.CONFIG,
        "delete": CommandType.DELETE,
        "remove": CommandType.DELETE,
        "rm": CommandType.DELETE,
        "init": CommandType.INIT,
        "setup": CommandType.INIT,
        "configure": CommandType.INIT,
        "help": CommandType.HELP,
        "?": CommandType.HELP,
        "quit": CommandType.QUIT,
        "exit": CommandType.QUIT,
        "q": CommandType.QUIT,
        "clear": CommandType.CLEAR,
        "cls": CommandType.CLEAR,
    }

    # Natural language patterns that suggest queries
    QUERY_PATTERNS = [
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "who ",
        "which ",
        "can ",
        "does ",
        "is ",
        "are ",
        "do ",
        "should ",
        "could ",
        "would ",
        "will ",
        "explain ",
        "describe ",
        "tell me ",
    ]

    def parse(self, input_text: str) -> ParsedCommand:
        """Parse user input into a command."""
        text = input_text.strip()
        if not text:
            return ParsedCommand(type=CommandType.UNKNOWN, args=[], raw=text, flags={})

        # Check for slash commands first
        if text.startswith("/"):
            return self._parse_slash_command(text[1:])

        # Split into words
        parts = text.split()
        first_word = parts[0].lower()

        # Check for explicit command
        if first_word in self.ALIASES:
            return self._parse_explicit_command(first_word, parts[1:], text)

        # Check for natural language query patterns
        lower_text = text.lower()
        for pattern in self.QUERY_PATTERNS:
            if lower_text.startswith(pattern):
                return ParsedCommand(type=CommandType.QUERY, args=[text], raw=text, flags={})

        # Check if it looks like a question (ends with ?)
        if text.endswith("?"):
            return ParsedCommand(type=CommandType.QUERY, args=[text], raw=text, flags={})

        # Default: treat as a query
        return ParsedCommand(type=CommandType.QUERY, args=[text], raw=text, flags={})

    def _parse_slash_command(self, text: str) -> ParsedCommand:
        """Parse a slash command like /help or /quit."""
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in self.ALIASES:
            return self._parse_explicit_command(cmd, parts[1:], "/" + text)

        return ParsedCommand(type=CommandType.UNKNOWN, args=parts, raw="/" + text, flags={})

    def _parse_explicit_command(self, cmd: str, args: list[str], raw: str) -> ParsedCommand:
        """Parse an explicit command with arguments."""
        cmd_type = self.ALIASES[cmd]
        flags: dict[str, str | bool] = {}
        positional_args: list[str] = []

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if "=" in key:
                    k, v = key.split("=", 1)
                    flags[k] = v
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    flags[key] = args[i + 1]
                    i += 1
                else:
                    flags[key] = True
            elif arg.startswith("-") and not arg.startswith("--"):
                if len(arg) == 2:
                    # Single short flag: -f or -f value
                    key = arg[1]
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        flags[key] = args[i + 1]
                        i += 1
                    else:
                        flags[key] = True
                else:
                    # Combined short flags: -rf -> -r -f (all boolean)
                    for char in arg[1:]:
                        flags[char] = True
            else:
                positional_args.append(arg)
            i += 1

        return ParsedCommand(type=cmd_type, args=positional_args, raw=raw, flags=flags)
