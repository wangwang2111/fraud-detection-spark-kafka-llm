import streamlit as st

def load_css(file_css):
    with open(f"{file_css}") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    if file_css == "style.css":
        st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

def styled_badge(text, bg_color):
    return f"""
    <span style="
        background-color:{bg_color};
        color:white;
        padding:4px 10px;
        border-radius:12px;
        font-weight:bold;
        font-size:0.9rem;
    ">
        {text}
    </span>
    """