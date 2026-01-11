"""Tests for the command parser."""

import pytest

from isotope.tui.commands.parser import CommandParser, CommandType


class TestCommandParser:
    """Tests for CommandParser."""

    @pytest.fixture
    def parser(self) -> CommandParser:
        return CommandParser()

    # Explicit commands
    def test_parse_ingest(self, parser: CommandParser) -> None:
        result = parser.parse("ingest ./docs")
        assert result.type == CommandType.INGEST
        assert result.args == ["./docs"]

    def test_parse_ingest_alias(self, parser: CommandParser) -> None:
        result = parser.parse("add ./docs")
        assert result.type == CommandType.INGEST
        assert result.args == ["./docs"]

    def test_parse_query(self, parser: CommandParser) -> None:
        result = parser.parse("query how does auth work")
        assert result.type == CommandType.QUERY
        assert result.args == ["how", "does", "auth", "work"]

    def test_parse_status(self, parser: CommandParser) -> None:
        result = parser.parse("status")
        assert result.type == CommandType.STATUS
        assert result.args == []

    def test_parse_status_alias(self, parser: CommandParser) -> None:
        result = parser.parse("stats")
        assert result.type == CommandType.STATUS

    def test_parse_list(self, parser: CommandParser) -> None:
        result = parser.parse("list")
        assert result.type == CommandType.LIST

    def test_parse_list_alias(self, parser: CommandParser) -> None:
        result = parser.parse("ls")
        assert result.type == CommandType.LIST

    def test_parse_config(self, parser: CommandParser) -> None:
        result = parser.parse("config")
        assert result.type == CommandType.CONFIG

    def test_parse_delete(self, parser: CommandParser) -> None:
        result = parser.parse("delete ./docs/readme.md")
        assert result.type == CommandType.DELETE
        assert result.args == ["./docs/readme.md"]

    def test_parse_delete_alias(self, parser: CommandParser) -> None:
        result = parser.parse("rm ./docs/readme.md")
        assert result.type == CommandType.DELETE

    def test_parse_init(self, parser: CommandParser) -> None:
        result = parser.parse("init")
        assert result.type == CommandType.INIT
        assert result.args == []

    def test_parse_init_aliases(self, parser: CommandParser) -> None:
        assert parser.parse("setup").type == CommandType.INIT
        assert parser.parse("configure").type == CommandType.INIT

    def test_parse_help(self, parser: CommandParser) -> None:
        result = parser.parse("help")
        assert result.type == CommandType.HELP

    def test_parse_help_alias(self, parser: CommandParser) -> None:
        result = parser.parse("?")
        assert result.type == CommandType.HELP

    def test_parse_quit(self, parser: CommandParser) -> None:
        result = parser.parse("quit")
        assert result.type == CommandType.QUIT

    def test_parse_quit_aliases(self, parser: CommandParser) -> None:
        assert parser.parse("exit").type == CommandType.QUIT
        assert parser.parse("q").type == CommandType.QUIT

    def test_parse_clear(self, parser: CommandParser) -> None:
        result = parser.parse("clear")
        assert result.type == CommandType.CLEAR

    # Slash commands
    def test_parse_slash_help(self, parser: CommandParser) -> None:
        result = parser.parse("/help")
        assert result.type == CommandType.HELP

    def test_parse_slash_quit(self, parser: CommandParser) -> None:
        result = parser.parse("/quit")
        assert result.type == CommandType.QUIT

    def test_parse_slash_ingest(self, parser: CommandParser) -> None:
        result = parser.parse("/ingest ./docs")
        assert result.type == CommandType.INGEST
        assert result.args == ["./docs"]

    # Natural language queries
    def test_parse_what_question(self, parser: CommandParser) -> None:
        result = parser.parse("what is the login flow?")
        assert result.type == CommandType.QUERY
        assert result.args == ["what is the login flow?"]

    def test_parse_how_question(self, parser: CommandParser) -> None:
        result = parser.parse("how does authentication work?")
        assert result.type == CommandType.QUERY
        assert result.args == ["how does authentication work?"]

    def test_parse_why_question(self, parser: CommandParser) -> None:
        result = parser.parse("why is the cache invalidated?")
        assert result.type == CommandType.QUERY

    def test_parse_explain_question(self, parser: CommandParser) -> None:
        result = parser.parse("explain the retry logic")
        assert result.type == CommandType.QUERY

    def test_parse_question_mark(self, parser: CommandParser) -> None:
        result = parser.parse("database connection?")
        assert result.type == CommandType.QUERY

    def test_parse_default_to_query(self, parser: CommandParser) -> None:
        # Anything that's not a command should be treated as a query
        result = parser.parse("some random text")
        assert result.type == CommandType.QUERY

    # Flags parsing
    def test_parse_long_flag(self, parser: CommandParser) -> None:
        result = parser.parse("status --detailed")
        assert result.type == CommandType.STATUS
        assert result.flags.get("detailed") is True

    def test_parse_short_flag(self, parser: CommandParser) -> None:
        result = parser.parse("query -k 10 something")
        assert result.type == CommandType.QUERY
        assert result.flags.get("k") == "10"

    def test_parse_flag_with_value(self, parser: CommandParser) -> None:
        # --force is a boolean flag when placed after the positional arg
        result = parser.parse("delete ./docs --force")
        assert result.type == CommandType.DELETE
        assert result.flags.get("force") is True
        assert result.args == ["./docs"]

    def test_parse_multiple_flags(self, parser: CommandParser) -> None:
        # --raw is a boolean flag, -k takes a value
        result = parser.parse("query --raw -k 5 something")
        assert result.type == CommandType.QUERY
        assert result.flags.get("k") == "5"
        assert result.flags.get("raw") is True

    # Edge cases
    def test_parse_empty_input(self, parser: CommandParser) -> None:
        result = parser.parse("")
        assert result.type == CommandType.UNKNOWN
        assert result.args == []

    def test_parse_whitespace_only(self, parser: CommandParser) -> None:
        result = parser.parse("   ")
        assert result.type == CommandType.UNKNOWN

    def test_parse_preserves_raw(self, parser: CommandParser) -> None:
        result = parser.parse("ingest ./docs")
        assert result.raw == "ingest ./docs"

    def test_parse_case_insensitive_commands(self, parser: CommandParser) -> None:
        result = parser.parse("INGEST ./docs")
        assert result.type == CommandType.INGEST

        result = parser.parse("Status")
        assert result.type == CommandType.STATUS
