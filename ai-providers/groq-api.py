from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from groq import Groq
import uvicorn

app = FastAPI(title="Groq AI Retrosynthesis Service")

# --- Request / Response Models ---
class RetrosynthesisRequest(BaseModel):
    target_smiles: str
    temperature: float = 0.2
    api_key: str


class RetrosynthesisResponse(BaseModel):
    reaction_name: str
    reactants: list[str]
    reagents: str
    reasoning: str


@app.post("/predict", response_model=RetrosynthesisResponse)
def predict_retro(req: RetrosynthesisRequest):
    try:
        client = Groq(api_key=req.api_key)

        system_prompt = """
        You are an expert Organic Chemist. Propose a one-step retrosynthesis for the target molecule.
        Return ONLY a JSON object with the following keys:
        - reaction_name: string
        - reactants: list of SMILES
        - reagents: string (include conditions)
        - reasoning: string (brief explanation)
        """

        user_prompt = f"Target Molecule SMILES: {req.target_smiles}"

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=req.temperature,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            result = {}

        return RetrosynthesisResponse(
            reaction_name=result.get("reaction_name", "Groq Predicted Transformation"),
            reactants=result.get("reactants", []),
            reagents=result.get("reagents", ""),
            reasoning=result.get("reasoning", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
