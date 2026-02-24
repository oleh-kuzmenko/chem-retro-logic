## AI Retro-Synthesis Planner

A Streamlit web app for retrosynthesis planning using either a local BART model or Groq Llama-3.3.


### Local Development Setup

To avoid dependency conflicts and build errors, follow these steps to set up a stable, isolated environment on your machine.

1. Prerequisites
    - Python 3.11
    - pyenv (for managing Python versions).
    - Model Weights: Ensure your trained BART model files are placed in the model/ directory.

2. Environment Initialization
Navigate to the project root and run:

```bash
# 1. Set the local Python version
pyenv local 3.11.9

# 2. Create an isolated virtual environment
python -m venv venv

# 3. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

3. Install Dependencies
Once the environment is active (indicated by (venv) in your terminal), install the required packages:

Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Running the Application
You can now launch the full suite with a single command:

```bash
streamlit run main.py
```

### Project Structure
The app is organized into a modular service-based architecture:
````text
retrosynthesis-app/
├── app/
│   ├── service/        # Business logic (BART & Groq classes)
│   └── utils/          # Chemistry & drawing helpers
├── model/              # Local BART weights (config/vocab/bin)
├── main.py             # Streamlit UI Entrypoint
└── requirements.txt    # Dependency list
````