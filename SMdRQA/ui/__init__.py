"""
SMdRQA interactive UI.

Launch with ``smdrqa-ui`` or ``python -m SMdRQA.ui`` (requires the
``ui`` extra: ``pip install SMdRQA[ui]``).

This package deliberately avoids importing streamlit at package level so
that ``SMdRQA.ui.recorder`` and ``SMdRQA.ui.sensitivity`` stay usable
(and testable) without the UI dependencies installed.
"""

from SMdRQA.ui.recorder import ScriptRecorder
from SMdRQA.ui.sensitivity import window_size_sensitivity

__all__ = ['ScriptRecorder', 'window_size_sensitivity', 'launch']


def launch():
    """Start the Streamlit app (console entry point ``smdrqa-ui``)."""
    import os
    import sys

    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:
        raise SystemExit(
            "The SMdRQA UI requires streamlit and plotly. "
            "Install them with: pip install SMdRQA[ui]") from exc

    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
