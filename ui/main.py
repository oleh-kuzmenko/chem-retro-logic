import requests
from groq import Groq
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from streamlit_ketcher import st_ketcher
from PIL import Image
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="AI Retrosynthesis Planner",
    layout="wide",
    page_icon="🧪"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stButton>button { 
        height: 3em; 
        background-color: #FF4B4B; 
        color: white; 
    }
    .reportview-container .main .block-container { 
        padding-top: 2rem; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Retro-Synthesis Planner")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Вибір двигуна моделі
    model_source = st.radio(
        "Select AI Engine",
        ["Local BART (My Model)", "Cloud Groq (Llama-3.3)"],
        help="Local model is your trained BART. Groq requires an API key."
    )

    if model_source == "Cloud Groq (Llama-3.3)":
        api_key = st.text_input("Groq API Key", type="password")
        st.markdown("[Get Free Groq Key](https://console.groq.com/keys)")
    else:
        model_url = "http://localhost:8000/predict"

    st.divider()
    temperature = st.slider(
        "Creativity (Temperature)",
        0.0, 1.0, 0.2, 0.1
    )


# --- Helper Functions ---
def clean_and_canonicalize(smiles):
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def generate_reaction_image(reactants_smiles, product_smiles):
    try:
        # Split reactants if they are combined with dots (A.B.C)
        if len(reactants_smiles) == 1 and "." in reactants_smiles[0]:
            reactants_smiles = reactants_smiles[0].split(".")

        # Convert SMILES to RDKit Mol objects
        reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
        product = Chem.MolFromSmiles(product_smiles)

        # Validate all molecules
        if not all(reactants) or not product:
            return None

        # Create a fake reaction image using reactants -> product
        # RDKit expects lists of reactants/products per reaction template
        reaction_img = Draw.MolsToGridImage(
            reactants + [product],
            legends=[f"R{i+1}" for i in range(len(reactants))] + ["Product"],
            subImgSize=(300, 300)
        )
        return reaction_img

    except Exception as e:
        st.error(f"Rendering Error: {e}")
        return None


def call_local_model(target_smiles, url):
    """Call your local BART model"""
    try:
        response = requests.post(
            url,
            json={"smiles": target_smiles, "num_predictions": 1},
            timeout=10
        )
        prediction = response.json()["predictions"][0]
        return {
            "reaction_name": "BART Predicted Transformation",
            "reactants": prediction.split("."),  # BART returns "A.B"
            "reagents": "Check scientific literature for conditions",
            "reasoning": "Prediction based on USPTO-50k trained Transformer model."
        }
    except Exception as e:
        st.error(f"Local Model Error: {e}")
        return None


def call_groq_service(target_smiles, api_url, temperature, api_key):
    """Call Groq FastAPI service on another host"""
    try:
        payload = {
            "target_smiles": target_smiles,
            "temperature": temperature,
            "api_key": api_key
        }
        response = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Groq Service Error: {e}")
        return None


# --- Main UI ---
smiles_input = st_ketcher("CC(=O)Oc1ccccc1C(=O)O", height=450)
canonical_input = clean_and_canonicalize(smiles_input)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Selected Target")
    if canonical_input:
        st.code(canonical_input)
        st.success("Structure Validated ✅")

    analyze_btn = st.button("🚀 Plan Synthesis", type="primary", use_container_width=True)

with col2:
    if analyze_btn:
        if model_source == "Cloud Groq (Llama-3.3)" and not api_key:
            st.warning("⚠️ Provide Groq API Key")
        elif not canonical_input:
            st.error("Draw a molecule first")
        else:
            with st.spinner(f"⚗️ Running {model_source}..."):
                if model_source == "Local BART (My Model)":
                    result = call_local_model(canonical_input, model_url)
                else:
                    groq_service_url = "http://localhost:8080"
                    result = call_groq_service(canonical_input, groq_service_url, temperature, api_key)

            if result:
                st.subheader(f"Proposed Path: {result.get('reaction_name')}")
                st.info(f"**Conditions:** {result.get('reagents')}")
                st.write(f"**Reasoning:** {result.get('reasoning')}")

                img = generate_reaction_image(result['reactants'], canonical_input)
                if img:
                    st.image(img, caption="Retrosynthetic Step")
                else:
                    st.warning("Visualization failed. Raw SMILES:")
                    st.code(result['reactants'])
