"""
Common utilities shared across analysis modules
"""

import plotly.graph_objects as go

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


def render_interactive_chart(st, fig, caption="Use the toolbar to zoom, pan, or reset (double-click)."):
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
