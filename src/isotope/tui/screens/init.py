# src/isotope/tui/screens/init.py
"""Interactive init screen for TUI."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from isotope.commands.init import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    EMBEDDING_MODEL_CHOICES,
    LLM_MODEL_CHOICES,
    generate_config_content,
    get_settings_for_init,
    save_api_key_to_env,
    update_gitignore_for_env,
)
from isotope.config import is_local_model


class InitScreen(Screen[bool]):
    """Interactive initialization screen."""

    CSS = """
    InitScreen {
        align: center middle;
    }

    #init-container {
        width: 80;
        max-height: 90%;
        background: $surface;
        padding: 1 2;
    }

    #init-title {
        text-align: center;
        text-style: bold;
        color: #ff8700;
        margin-bottom: 1;
    }

    .step-title {
        color: #5fafaf;
        margin-top: 1;
        margin-bottom: 0;
    }

    .step-description {
        color: $text-muted;
        margin-bottom: 1;
    }

    RadioSet {
        margin-bottom: 1;
        height: auto;
        max-height: 10;
    }

    Input {
        margin-bottom: 1;
    }

    #api-key-input {
        margin-bottom: 1;
    }

    #button-row {
        margin-top: 1;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }

    #btn-next {
        background: #ff8700;
    }

    #btn-back {
        background: $surface-darken-1;
    }

    #btn-cancel {
        background: $error;
    }

    .hidden {
        display: none;
    }

    #step-1, #step-2, #step-3, #step-4, #step-5 {
        height: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._step = 1
        self._llm_model: str = DEFAULT_LLM_MODEL
        self._embedding_model: str = DEFAULT_EMBEDDING_MODEL
        self._is_local = False
        self._rate_limited: bool | None = None
        self._priority = "balanced"
        self._llm_api_key: str | None = None
        self._embed_api_key: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the init screen."""
        with VerticalScroll(id="init-container"):
            yield Static("Initialize Isotope", id="init-title")

            # Step 1: LLM Model Selection
            with Container(id="step-1"):
                yield Label("Step 1: Select LLM Model", classes="step-title")
                yield Label(
                    "Choose the model for question generation and synthesis",
                    classes="step-description",
                )
                with RadioSet(id="llm-model-select"):
                    for model in LLM_MODEL_CHOICES:
                        yield RadioButton(model, value=model == DEFAULT_LLM_MODEL)
                yield Label("Or enter custom model:", classes="step-description")
                yield Input(placeholder="e.g., anthropic/claude-3-haiku", id="llm-custom")

            # Step 2: Embedding Model Selection
            with Container(id="step-2", classes="hidden"):
                yield Label("Step 2: Select Embedding Model", classes="step-title")
                yield Label(
                    "Choose the model for generating embeddings",
                    classes="step-description",
                )
                with RadioSet(id="embedding-model-select"):
                    for model in EMBEDDING_MODEL_CHOICES:
                        yield RadioButton(model, value=model == DEFAULT_EMBEDDING_MODEL)
                yield Label("Or enter custom model:", classes="step-description")
                yield Input(placeholder="e.g., cohere/embed-english-v3.0", id="embed-custom")

            # Step 3: Rate Limit Settings (only for non-local models)
            with Container(id="step-3", classes="hidden"):
                yield Label("Step 3: API Rate Limits", classes="step-title")
                yield Label(
                    "Are you on a rate-limited or free tier API?",
                    classes="step-description",
                )
                with RadioSet(id="rate-limit-select"):
                    yield RadioButton("Yes - configure for rate limits", id="rate-yes")
                    yield RadioButton("No - I have high rate limits", id="rate-no")
                    yield RadioButton("Not sure - use safe defaults", value=True, id="rate-unsure")

            # Step 4: Priority Settings (only for non-local models)
            with Container(id="step-4", classes="hidden"):
                yield Label("Step 4: Priority", classes="step-title")
                yield Label("What's your priority?", classes="step-description")
                with RadioSet(id="priority-select"):
                    yield RadioButton(
                        "Retrieval quality (slower, more API calls)", id="prio-quality"
                    )
                    yield RadioButton("Speed & cost savings (faster, fewer calls)", id="prio-speed")
                    yield RadioButton("Balanced", value=True, id="prio-balanced")

            # Step 5: API Key
            with Container(id="step-5", classes="hidden"):
                yield Label("Step 5: API Key", classes="step-title")
                yield Label(
                    "Enter your API key (optional - leave empty to set later)",
                    classes="step-description",
                )
                yield Input(
                    placeholder="sk-...",
                    password=True,
                    id="api-key-input",
                )
                yield Label(
                    "The key will be saved to .env (which is gitignored)",
                    classes="step-description",
                )

            # Navigation buttons
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="btn-cancel", variant="error")
                yield Button("Back", id="btn-back", disabled=True)
                yield Button("Next", id="btn-next", variant="primary")

    def on_mount(self) -> None:
        """Focus the first radio set."""
        self.query_one("#llm-model-select", RadioSet).focus()

    def _show_step(self, step: int) -> None:
        """Show a specific step and hide others."""
        for i in range(1, 6):
            container = self.query_one(f"#step-{i}", Container)
            if i == step:
                container.remove_class("hidden")
            else:
                container.add_class("hidden")

        # Update button states
        back_btn = self.query_one("#btn-back", Button)
        next_btn = self.query_one("#btn-next", Button)

        back_btn.disabled = step == 1

        # Determine max step based on whether model is local
        max_step = 2 if self._is_local else 5
        next_btn.label = "Create Config" if step == max_step else "Next"

        self._step = step

    def _get_selected_llm_model(self) -> str:
        """Get the selected LLM model."""
        custom_input = self.query_one("#llm-custom", Input)
        if custom_input.value.strip():
            return custom_input.value.strip()

        radio_set = self.query_one("#llm-model-select", RadioSet)
        if radio_set.pressed_button:
            return str(radio_set.pressed_button.label)
        return DEFAULT_LLM_MODEL

    def _get_selected_embedding_model(self) -> str:
        """Get the selected embedding model."""
        custom_input = self.query_one("#embed-custom", Input)
        if custom_input.value.strip():
            return custom_input.value.strip()

        radio_set = self.query_one("#embedding-model-select", RadioSet)
        if radio_set.pressed_button:
            return str(radio_set.pressed_button.label)
        return DEFAULT_EMBEDDING_MODEL

    def _get_rate_limited(self) -> bool | None:
        """Get rate limited setting."""
        radio_set = self.query_one("#rate-limit-select", RadioSet)
        if radio_set.pressed_button:
            btn_id = radio_set.pressed_button.id
            if btn_id == "rate-yes":
                return True
            elif btn_id == "rate-no":
                return False
        return True  # Default to rate limited for safety

    def _get_priority(self) -> str:
        """Get priority setting."""
        radio_set = self.query_one("#priority-select", RadioSet)
        if radio_set.pressed_button:
            btn_id = radio_set.pressed_button.id
            if btn_id == "prio-quality":
                return "quality"
            elif btn_id == "prio-speed":
                return "speed"
        return "balanced"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id

        if btn_id == "btn-cancel":
            self.dismiss(False)
            return

        if btn_id == "btn-back":
            if self._step > 1:
                # For local models, skip steps 3-4
                if self._is_local and self._step == 5:
                    self._show_step(2)
                else:
                    prev_step = self._step - 1
                    # Skip steps 3-4 if local
                    if self._is_local and prev_step in (3, 4):
                        prev_step = 2
                    self._show_step(prev_step)
            return

        if btn_id == "btn-next":
            await self._handle_next()

    async def _handle_next(self) -> None:
        """Handle next button press."""
        if self._step == 1:
            # Get LLM model and check if local
            self._llm_model = self._get_selected_llm_model()
            self._is_local = is_local_model(self._llm_model)
            self._show_step(2)
            self.query_one("#embedding-model-select", RadioSet).focus()

        elif self._step == 2:
            # Get embedding model
            self._embedding_model = self._get_selected_embedding_model()

            if self._is_local:
                # Skip to creating config for local models
                await self._create_config()
            else:
                self._show_step(3)
                self.query_one("#rate-limit-select", RadioSet).focus()

        elif self._step == 3:
            # Get rate limit setting
            self._rate_limited = self._get_rate_limited()
            self._show_step(4)
            self.query_one("#priority-select", RadioSet).focus()

        elif self._step == 4:
            # Get priority
            self._priority = self._get_priority()
            self._show_step(5)
            self.query_one("#api-key-input", Input).focus()

        elif self._step == 5:
            # Get API key and create config
            api_key_input = self.query_one("#api-key-input", Input)
            self._llm_api_key = api_key_input.value.strip() or None
            if self._llm_api_key:
                self._embed_api_key = self._llm_api_key  # Same key for both
            await self._create_config()

    async def _create_config(self) -> None:
        """Create the configuration file."""
        from pathlib import Path

        config_path = Path("isotope.yaml")
        env_path = Path(".env")
        gitignore_path = Path(".gitignore")

        # Get settings based on choices
        settings = get_settings_for_init(
            self._is_local,
            self._rate_limited,
            self._priority,
        )

        # Generate and write config
        content = generate_config_content(
            self._llm_model,
            self._embedding_model,
            settings,
        )
        config_path.write_text(content, encoding="utf-8")

        # Save API key if provided
        if self._llm_api_key:
            save_api_key_to_env("ISOTOPE_LLM_API_KEY", self._llm_api_key, env_path)
            import os

            os.environ["ISOTOPE_LLM_API_KEY"] = self._llm_api_key
            update_gitignore_for_env(gitignore_path)

        self.dismiss(True)
