import os
import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration

class BartPlannerService:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, "model")

        self.tokenizer = BartTokenizerFast.from_pretrained(
            model_path, 
            local_files_only=True
        )
        
        self.model = BartForConditionalGeneration.from_pretrained(
            model_path, 
            local_files_only=True
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def predict(self, smiles: str):
        with torch.no_grad():
            inputs = self.tokenizer(smiles, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"], 
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
