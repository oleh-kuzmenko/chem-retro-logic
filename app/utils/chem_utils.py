import logging
from rdkit import Chem
from rdkit import rdBase as RdBase 
from rdkit.Chem import Draw, AllChem
from typing import List, Optional, Union

RdBase.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)

def canonicalize(smiles: str) -> Optional[str]:
    """
    Validates and converts a SMILES string to its canonical form.
    Returns None if the SMILES is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol:
            # Remove stereochemistry if you want a more general match, 
            # or keep it for precision (default: keep)
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.error(f"Canonicalization error: {e}")
    return None

def draw_retro_reaction(reactants: List[str], product: str, img_size: tuple = (600, 300)):
    """
    Generates a high-quality chemical reaction image.
    Uses ChemicalReaction objects for proper 'Reactants -> Product' rendering.
    """
    try:
        # 1. Clean and validate inputs
        valid_reactants = [r for r in reactants if Chem.MolFromSmiles(r)]
        if not valid_reactants or not Chem.MolFromSmiles(product):
            logger.warning("Invalid reactants or product SMILES provided for drawing.")
            return None

        # 2. Construct Reaction SMARTS
        # Format: Reactant1.Reactant2>>Product
        rxn_smarts = f"{'.'.join(valid_reactants)}>>{product}"
        
        # 3. Create Reaction Object
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        
        # 4. Render to Image
        # useSVG=False returns a PIL Image object (best for Streamlit st.image)
        img = Draw.ReactionToImage(
            rxn, 
            subImgSize=(img_size[0]//2, img_size[1]),
            useSVG=False 
        )
        return img

    except Exception as e:
        logger.error(f"Failed to draw reaction: {e}")
        return None

def get_mol_weight(smiles: str) -> float:
    """Calculates molecular weight for metadata displays."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.CalcExactMolWt(mol)
    return 0.0
