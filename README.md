# Document Classifier — Perfect Importa

## O problema (em uma frase)

Today, PerfectAPP decides the PDF model with hundreds of `if (texto.Contains(...))`.
Each new payment instruction need a new code.
Here, I try a classifier: **extracted text → model**.

## Steps

1. Generate anonymized dataset: `python -m src.prepare_dataset`
2. Train and measure: `python -m src.train`
   - Terminal report + `data/processed/confusion_matrix.png`
   - Model in `models/classify_document.joblib`
3. API: `python -m src.api` → open http://127.0.0.1:8000

## Run API 

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.api
```

Browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Or `curl`:

```powershell
curl -s http://127.0.0.1:8000/v1/identify -H "Content-Type: application/json" -d "{\"text\": \"Documento de Arrecadação de Receitas Federais DARF código da receita 2089\"}"
```

Typical Response:

```json
{
  "type": "GuiaPagamento",
  "model": "DARF",
  "confidence": 0.91,
  "needs_review": false,
  "text_origin": "json"
}
```

`needs_review` is `true` if the confidence is less than 0.75.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.prepare_dataset
python -m src.train
```

CSV goes to `data/processed/documents.csv` (`text`, `model`, `type`).
CNPJs, dates and values are `[CNPJ]`, `[DATE]`, `[VALUE]`.