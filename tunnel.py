from __future__ import annotations

import os

import streamlit as st
from pyngrok import conf, ngrok


@st.cache_resource(show_spinner=False)
def start_ngrok_tunnel(port: int) -> tuple[str | None, str | None]:
    existing_public_url = os.getenv("APP_PUBLIC_URL")
    if existing_public_url:
        return existing_public_url, None

    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        return None, (
            "Set NGROK_AUTHTOKEN before launching Streamlit to expose this app "
            "publicly with ngrok."
        )

    try:
        conf.get_default().auth_token = token
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        print(f"ngrok public URL: {public_url}")
        return public_url, None
    except Exception as exc:
        return None, f"Could not start ngrok tunnel: {exc}"
