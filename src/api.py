"""
API for the machine learning model that PerfectAPP will use.
"""

from __future__ import annotations

from pathlib import Path
from urllib import request

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import identify_text

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

app = FastAPI(
    title="Identificador de documentos",
    description="API para classificar documentos de pagamento e relatórios contábeis.",
    version="0.1.0",
)

class IdentifyRequest(BaseModel):
    """
    Request model for identifying the type of a document based on its text content.
    """
    text: str = Field(..., description="Texto extraído do PDF para classificação.")

@app.get("/")
def page() -> FileResponse:
    """
    Serve the static HTML page for the API.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static index.html not found.")
    return FileResponse(index_path)

@app.post("/v1/identify")
def identify(body: IdentifyRequest) -> dict:
    """
    Identify the type of a document based on its text content.
    """
    return {"type": identify_text(body.text, text_origin="json")}

@app.post("/v1/identify/file")
async def identify_file(file: UploadFile = File(...)) -> dict:
    """
    Identify the type of a document based on the text content extracted from an uploaded file.
    """
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são suportados.")

    raw = await file.read()
    try:
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar o PDF: {str(e)}") from e

    if not text.strip():
        return {
            "type": "Desconhecido",
            "model": "Desconhecido",
            "confidence": 0.0,
            "needs_review": True,
            "origin": "pdf",
            "error": "Não foi possível extrair texto do PDF.",
        }
    return identify_text(text, text_origin="pypdf")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, log_level="info", reload=False)

if __name__ == "__main__":
    main()