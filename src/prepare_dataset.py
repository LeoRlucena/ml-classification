"""
Generate a CSV file containing the dataset for training and testing the model.
The dataset is inspired in real layouts from "Perfect Importa".
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.anonimizer import anonymize
from src.config import (
    PER_CLASS_SAMPLES,
    CSV_DATASET_PATH,
    PROCESSED_DIR,
    SEED,
    OCR_NOISE_THRESHOLD,
)

HEADERS = [
    "Contribuinte: Empresa Exemplo 104 LTDA CNPJ 12.345.678/0001-90",
    "Identificação do sujeito passivo 98.765.432/0001-10",
    "Período de apuração 01/03/2026 a 31/03/2026",
    "Vencimento 20/04/2026 valor total R$ 1.234,56",
    "Autenticação bancária 00000.00000 11111.111111 22222.222222",
    "Código de barras 85640000001234567890123456789012345678901234",
]

OCR_NOISE = str.maketrans({"O": "0", "I": "1", "l": "1", "S": "5"})

def _wrapup(pins: list[str], rng: random.Random) -> str:
    extras = rng.sample(HEADERS, k=rng.randint(2, 4))
    block = extras + pins
    rng.shuffle(block)
    text = "\n".join(block)
    if rng.random() < OCR_NOISE_THRESHOLD:
        text = text.translate(OCR_NOISE) # Emulate bad OCR recognition
    return anonymize(text)

def _simples_nacional(rng: random.Random) -> str:
    return _wrapup(
        [
            "Documento de Arrecadação do Simples Nacional",
            "Programa Gerador do DAS",
            "Receita Federal do Brasil DAS",
        ],
        rng,
    )

def _fgts(rng: random.Random) -> str:
    return _wrapup(
        [
            "GFD - Guia do FGTS Digital",
            "Fundo de Garantia do Tempo de Serviço",
            "Competência da guia FGTS Digital",
        ],
        rng,
    )

def _darf(rng: random.Random) -> str:
    return _wrapup(
        [
            "Documento de Arrecadação de Receitas Federais",
            "Ministério da Fazenda Secretaria da Receita Federal do Brasil",
            "DARF código da receita 2089",
        ],
        rng,
    )

def _dctfweb(rng: random.Random) -> str:
    return _wrapup(
        [
            "Documento de Arrecadação de Receitas Federais",
            "Contribuição previdenciária CONTRPREV",
            "IRRF - Rendimento do trabalho assalariado DCTFWeb",
        ],
        rng,
    )

def _gars(rng: random.Random) -> str:
    return _wrapup(
        [
            "Estado do Rio Grande Do Sul",
            "Guia de Arrecadação - GA",
            "Guia de Arrecadação",
        ],
        rng,
    )

def _gps(rng: random.Random) -> str:
    return _wrapup(
        [
            "Ministério da Previdência Social",
            "INSS Guia da Previdência Social GPS",
            "Código de pagamento 2100",
        ],
        rng,
    )

def _dre(rng: random.Random) -> str:
    return _wrapup(
        [
            "Demonstrativo de Resultado",
            "Demonstração de Resultado",
        ],
        rng,
    )

def _sped(rng: random.Random) -> str:
    return _wrapup(
        [
            "Sistema Público de Escrituração Digital SPED",
            "Recibo de entrega de Escrituração Fiscal Digital",
            "EFD ICMS IPI número do recibo",
        ],
        rng,
    )

def _iss_poa(rng: random.Random) -> str:
    return _wrapup(
        [
            "Prefeitura de Porto Alegre",
            "ISSQN Imposto Sobre Serviços de Qualquer Natureza",
            "Guia de recolhimento municipal",
        ],
        rng,
    )

def _contracheque(rng: random.Random) -> str:
    return _wrapup(
        [
            "Adiantamento",
            "Adiantamento Salarial",
            "Declaro ter recebido a importância líquida",
        ],
        rng,
    )

def _sindicato(rng: random.Random) -> str:
    return _wrapup(
        [
            "Boleto de contribuição assistencial",
            "Sindicato dos empregados contribuição confederativa",
            "Cedente sindicato código do beneficiário",
        ],
        rng,
    )

def _desconhecido(rng: random.Random) -> str:
    options = [
        [
            "Contrato de prestação de serviços cláusula primeira",
            "As partes acima identificadas têm entre si justo e contratado",
        ],
        [
            "Pedido de compra ordem 4521",
            "Quantidade 10 unidades descrição material de escritório",
        ],
        [
            "Ata de reunião pauta 1 aprovação das contas",
            "Presentes os sócios na sede da sociedade",
        ],
    ]
    return _wrapup(rng.choice(options), rng)

GENERATORS = {
    "SimplesNacional": ("GuiaPagamento", _simples_nacional),
    "FGTSDigital": ("GuiaPagamento", _fgts),
    "DARF": ("GuiaPagamento", _darf),
    "DCTFWEB": ("GuiaPagamento", _dctfweb),
    "GARS": ("GuiaPagamento", _gars),
    "GuiaGPS": ("GuiaPagamento", _gps),
    "DREMensal": ("Relatorio", _dre),
    "SPEDFiscal": ("Recibo", _sped),
    "ISSPoa": ("GuiaPagamento", _iss_poa),
    "Contracheque": ("Recibo", _contracheque),
    "BoletoSindicato": ("GuiaPagamento", _sindicato),
    "Desconhecido": ("Desconhecido", _desconhecido),
}

def generate_dataset(n_per_class: int = PER_CLASS_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    lines: list[dict] = []
    for model, (type, fn) in GENERATORS.items():
        for _ in range(n_per_class):
            lines.append({
                "text": fn(rng),
                "model": model,
                "type": type,
                "origin": "synthetic",
            }
        )
    df = pd.DataFrame(lines)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True) # Shuffle the dataset

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(CSV_DATASET_PATH, index=False, encoding="utf-8")
    print(f"Generated dataset with {len(df)} samples and saved to {CSV_DATASET_PATH}")
    print(df["model"].value_counts().to_string())
    print("\nAnonymized sample (first line):")
    print(df.iloc[0]["text"][:400])

if __name__ == "__main__":
    main()














