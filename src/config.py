"""
Configuration file for payment instructions classification project.
This prevents us for not using all 90 enum values in the training dataset.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

CSV_DATASET_PATH = PROCESSED_DIR / "dataset.csv"
MODEL_PATH = MODEL_DIR / "classify_document.joblib"

# (model, type) pairs with PerfectAPP names for future API integration.
CLASSES = [
    ("SimplesNacional", "GuiaPagamento"),
    ("FGTSDigital", "GuiaPagamento"),
    ("DARF", "GuiaPagamento"),
    ("DCTFWEB", "GuiaPagamento"),
    ("GNRE", "GuiaPagamento"),
    ("GuiaGPS", "GuiaPagamento"),
    ("BalanceteMensal", "Relatorio"),
    ("SPEDFiscal", "Recibo"),
    ("ISSPoa", "GuiaPagamento"),
    ("ISSCanoas", "GuiaPagamento"),
    ("BoletoSindicato", "GuiaPagamento"),
    ("Desconhecido", "Desconhecido"),
]

#Hyperparameters for the model training.
CONFIDENCE_THRESHOLD = 0.75 # Minimum confidence required for a prediction to be considered valid
PER_CLASS_SAMPLES = 80 # Number of samples per class to be used for training the model
SEED = 42 # Random seed for reproducibility
OCR_NOISE_THRESHOLD = 0.15 # Amount of noise to add to the OCR data emulation, to simulate real-world OCR errors and improve model robustness
TEST_SIZE = 0.2 # Proportion of the dataset to be used for testing the model