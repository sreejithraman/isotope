"""Isotope TUI color theme and constants."""

# Primary brand colors (based on Makefile ANSI 256 #208)
ISOTOPE_ORANGE = "#ff8700"  # Primary brand color
ISOTOPE_AMBER = "#ffaf5f"  # Lighter accent
ISOTOPE_RUST = "#d75f00"  # Darker accent for depth

# Complementary colors
ISOTOPE_TEAL = "#5fafaf"  # Complementary accent (for contrast)
ISOTOPE_CYAN = "#87d7d7"  # Lighter teal

# Background colors
BG_DARK = "#1c1c1c"  # Main background
BG_SURFACE = "#262626"  # Elevated surfaces (panels, cards)
BG_BOOST = "#303030"  # Hover/focus states

# Semantic colors (themed to match)
SUCCESS = "#87d787"  # Green with warmth
WARNING = "#ffaf5f"  # Uses amber
ERROR = "#ff8787"  # Soft red
INFO = "#87afd7"  # Muted blue

# Text colors
TEXT_PRIMARY = "#ffffff"  # Primary text
TEXT_MUTED = "#808080"  # Secondary/dim text
TEXT_ACCENT = ISOTOPE_ORANGE

# CSS variable names for consistency
CSS_VARS = """
$isotope-orange: #ff8700;
$isotope-amber: #ffaf5f;
$isotope-rust: #d75f00;
$isotope-teal: #5fafaf;
$isotope-cyan: #87d7d7;

$bg-dark: #1c1c1c;
$bg-surface: #262626;
$bg-boost: #303030;

$success: #87d787;
$warning: #ffaf5f;
$error: #ff8787;
$info: #87afd7;

$text-primary: #ffffff;
$text-muted: #808080;
"""
