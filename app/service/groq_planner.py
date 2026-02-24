import json
from groq import AsyncGroq
from pydantic import BaseModel

class RetrosynthesisResult(BaseModel):
    reaction_name: str
    reactants: list[str]
    reagents: str
    reasoning: str

class GroqPlannerService:
    def __init__(self):
        self.model_name = "llama-3.3-70b-versatile"

    async def predict_step(self, target: str, api_key: str, temperature: float):
        client = AsyncGroq(api_key=api_key)

        system_prompt = """
        You are an expert Organic Chemist. Propose a one-step retrosynthesis for the target molecule.
        Return ONLY a JSON object with the following keys:
        - reaction_name: string
        - reactants: list of SMILES
        - reagents: string
        - reasoning: string
        """

        user_prompt = f"Target Molecule SMILES: {target}"

        try:
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            response_content = completion.choices[0].message.content
            
            try:
                data = json.loads(response_content)
            except json.JSONDecodeError:
                data = {}

            return RetrosynthesisResult(
                reaction_name=data.get("reaction_name", "Groq Predicted Transformation"),
                reactants=data.get("reactants", []),
                reagents=data.get("reagents", "No specific reagents provided"),
                reasoning=data.get("reasoning", "No detailed reasoning provided")
            )

        except Exception as e:
            return RetrosynthesisResult(
                reaction_name="Error",
                reactants=[],
                reagents="",
                reasoning=f"Failed to get response from Groq: {str(e)}"
            )