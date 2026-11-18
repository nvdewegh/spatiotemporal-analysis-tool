from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


_viewer = components.declare_component(
    "samtool_local_a2_viewer", path=str(Path(__file__).with_suffix(""))
)


def render():
    """Mount the viewer. The component has no file upload or return value."""
    with st.expander("A2 video and trajectory viewer"):
        _viewer(key="local_a2_viewer")
