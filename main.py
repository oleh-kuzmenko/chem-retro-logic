import streamlit as st
import asyncio
from app.service.bart_planner import BartPlannerService
from app.service.groq_planner import GroqPlannerService
from app.utils.chem_utils import canonicalize, draw_retro_reaction

@st.cache_resource
def load_bart():
    return BartPlannerService()

groq_service = GroqPlannerService()

with st.sidebar:
    st.title("Settings")
    engine = st.radio("Engine", ["Local BART", "Cloud Groq"])
    user_api_key = st.text_input("Groq API Key", type="password") if engine == "Cloud Groq" else None
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)

st.title("AI Retrosynthesis")
from streamlit_ketcher import st_ketcher
smiles = st_ketcher("c1ccccc1")

if st.button("Analyze"):
    target = canonicalize(smiles)
    
    if engine == "Cloud Groq" and not user_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        with st.spinner("Calculating path..."):
            if engine == "Local BART":
                model = load_bart()
                raw = model.predict(target)
                result = {"name": "BART Path", "reactants": raw.split(".")}
            else:
                res_obj = asyncio.run(groq_service.predict_step(target, user_api_key, temp))
                result = res_obj.model_dump()

            st.image(draw_retro_reaction(result['reactants'], target))
            st.json(result)
            