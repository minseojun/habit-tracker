import streamlit as st
from typing import Optional


def kpi_card(title: str, value: str, subtitle: Optional[str] = None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(f"<div style='font-size:28px; font-weight:700'>{value}</div>", unsafe_allow_html=True)
        if subtitle:
            st.caption(subtitle)


def info_card(title: str, body: str):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(body)
