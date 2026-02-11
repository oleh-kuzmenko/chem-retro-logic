## AI Retro-Synthesis Planner

A Streamlit web app for retrosynthesis planning using either a local BART model or Groq Llama-3.3.


### Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Make sure you have:
    - ```model/``` folder with your trained BART model
    - ```rdkit```, ```transformers```, ```torch```, ```streamlit```, ```groq```, ```streamlit-ketcher```, ```PIL installed```


### Running the App

1. Start the local BART API
```bash
uvicorn core-models.trained-model-api:app --host 0.0.0.0 --port 8000 --reload
```

- API endpoint: ```http://localhost:8000/predict```
- Send JSON like:

```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

2. Start Groq AI Retrosynthesis Service
```bash 
uvicorn ai-providers.groq-api:app --host 0.0.0.0 --port 8080 --reload
```

- API endpoint: ```http://localhost:8080/predict```

4. Start Frontend (Streamlit)
```bash 
streamlit run ui/main.py
```
- Draw molecules or paste SMILES
- Choose AI engine: Local BART or Cloud Groq
- Visualize predicted reactants and retrosynthesis pathway