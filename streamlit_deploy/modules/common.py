"""
Common utilities shared across analysis modules
"""

import os
import locale
import plotly.graph_objects as go

# Detect user's locale for consistent number formatting
def get_user_locale_decimal():
    """Detect if user's system uses comma as decimal separator."""
    try:
        # Check current locale
        current_locale = locale.getlocale(locale.LC_NUMERIC)
        if current_locale and current_locale[0]:
            loc = current_locale[0].lower()
            # European locales that use comma
            european_locales = ['nl', 'de', 'fr', 'es', 'it', 'pt', 'be', 'pl', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sl', 'et', 'lv', 'lt', 'fi', 'sv', 'da', 'no', 'el', 'tr', 'ru', 'uk']
            for euro_loc in european_locales:
                if euro_loc in loc:
                    return ','
    except:
        pass
    
    # Also check environment
    for env_var in ['LC_NUMERIC', 'LC_ALL', 'LANG']:
        env_val = os.environ.get(env_var, '').lower()
        if env_val:
            european_locales = ['nl', 'de', 'fr', 'es', 'it', 'pt', 'be', 'pl', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sl', 'et', 'lv', 'lt', 'fi', 'sv', 'da', 'no', 'el', 'tr', 'ru', 'uk']
            for euro_loc in european_locales:
                if euro_loc in env_val:
                    return ','
    return '.'

# Store the decimal separator to use throughout the app
DECIMAL_SEP = get_user_locale_decimal()

def format_number(value, decimals=1):
    """Format a number with the appropriate decimal separator for consistency."""
    if DECIMAL_SEP == ',':
        return f"{value:.{decimals}f}".replace('.', ',')
    return f"{value:.{decimals}f}"

def format_number_auto(value):
    """Format a number with automatic decimal places."""
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        return str(int(value))
    if DECIMAL_SEP == ',':
        return f"{value:.2f}".replace('.', ',')
    return f"{value:.2f}"

# Common Plotly configuration for interactive charts
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": [
        "zoom2d",
        "pan2d",
        "autoScale2d",
        "resetScale2d"
    ],
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d"
    ]
}


def render_interactive_chart(st, fig, caption=None):
    """
    Render a Plotly figure with consistent interactive controls.
    
    Args:
        st: Streamlit module
        fig: Plotly figure object
        caption: Optional caption to display below chart
    """
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    if caption:
        st.caption(caption)
