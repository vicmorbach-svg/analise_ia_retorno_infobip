from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os

API_KEY = os.getenv("API_KEY", "")
app = FastAPI()

class ItemIn(BaseModel):
    messageId: str
    texto: str = ""

class BatchIn(BaseModel):
    items: List[ItemIn]

@app.get("/health")
def health():
    return {"ok": True}

def classify_rules(texto: str):
    t = (texto or "").lower()
    if "falta" in t and ("agua" in t or "água" in t):
        return "Falta_dagua", "Cliente relata falta de água", "Abrir_verificacao_rede", 0.93
    if "vazamento" in t:
        return "Vazamento", "Cliente relata vazamento", "Abrir_verificacao_rede", 0.92
    if "conta" in t or "cobran" in t or "segunda via" in t:
        return "Cobranca", "Solicitação sobre cobrança", "Direcionar_comercial", 0.90
    return "Outros", (texto[:120] if texto else "Mensagem sem texto"), "Solicitar_dados_complementares", 0.60

@app.post("/classify-batch")
def classify_batch(payload: BatchIn, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    results = []
    for it in payload.items:
        categoria, resumo, acao, confianca = classify_rules(it.texto)
        results.append({
            "messageId": it.messageId,
            "categoria": categoria,
            "resumo": resumo,
            "acao": acao,
            "confianca": confianca
        })

    return {"ok": True, "provider": "rules", "results": results, "errors": []}
