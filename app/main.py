from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ItemIn(BaseModel):
    messageId: str
    texto: str = ""
    from_: Optional[str] = None
    receivedAt: Optional[str] = None
    numLigacao: Optional[str] = None

class BatchIn(BaseModel):
    items: List[ItemIn]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/classify-batch")
def classify_batch(payload: BatchIn):
    results = []
    for it in payload.items:
        results.append({
            "messageId": it.messageId,
            "categoria": "Outros",
            "resumo": it.texto[:120],
            "acao": "Solicitar_dados_complementares",
            "confianca": 0.5
        })
    return {"ok": True, "provider": "rules", "results": results, "errors": []}
