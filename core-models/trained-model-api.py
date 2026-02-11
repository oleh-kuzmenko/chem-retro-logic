from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import uvicorn
from pathlib import Path

app = FastAPI(title="BART Inference Service")

MODEL_PATH = Path(__file__).parent / "model"
MODEL_PATH = str(MODEL_PATH.resolve())

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class PredictRequest(BaseModel):
    smiles: str

@app.post("/predict")
async def predict(request: PredictRequest):
    inputs = tokenizer(request.smiles, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=128)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prediction)
    return {"predictions": [prediction]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
