import streamlit as st


def centered_title(title):
    st.markdown(
        f"""
        <div style='box-sizing: border-box; display:flex;flex-direction:column;align-items:center;justify-content:center; height: 70vh;'>
        <div style='box-sizing: border-box;display: flex; justify-content: center; align-items: center'>
            <h1 style='max-width:400px; text-align:center'>{title}</h1>
        </div>
        <p style='margin-top:25px'>Designed and Created by : <span style='color:#FF4B4B'> Younes Qara </span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def centered_text(text,font_size):
    st.markdown(
        f"""
        <div style='box-sizing: border-box; display:flex;flex-direction:column;align-items:center;justify-content:center;'>
            <p style:'font-size:{font_size}px'>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )