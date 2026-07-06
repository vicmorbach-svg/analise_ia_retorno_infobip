import os
import re
import logging
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

# =========================
# Configuração
# =========================
API_KEY = os.getenv("API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ia-classifier")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# Categorias e ações permitidas (fechadas)
CATEGORIAS = {
  
  "Pagamento Imediato",
  "Promessa de Pagamento",
  "Negociação",
  "Incapacidade",
  "Outros / Sem Contexto",
  "Já pagou",
  "Não reconhece a dívida",
  "Questiona o valor",
  "Já fez contato",
  "Outros serviços",
  "Débito em conta",
  "Saudação",
  "Canais críticos",
  "Cadastro atualizado",
  "Problemas nos canais",
  "Golpe"
}
ACOES = {
    "Abrir_verificacao_rede",
    "Direcionar_comercial",
    "Solicitar_dados_complementares",
    "Encerrar_sem_acao",
}

# =========================
# App
# =========================
app = FastAPI(
    title="IA Classifier API",
    version="1.1.0",
    description="Classificador de mensagens para Power Automate"
)

# =========================
# Schemas
# =========================
class ItemIn(BaseModel):
    messageId: str = Field(..., min_length=1)
    texto: Optional[str] = ""
    from_: Optional[str] = Field(default=None, alias="from")
    receivedAt: Optional[str] = None
    numLigacao: Optional[str] = None

    @field_validator("texto", mode="before")
    @classmethod
    def normalize_texto(cls, v):
        if v is None:
            return ""
        return str(v)

class BatchIn(BaseModel):
    items: List[ItemIn]

class ItemOut(BaseModel):
    messageId: str
    categoria: str
    resumo: str
    acao: str
    confianca: float

class BatchOut(BaseModel):
    ok: bool
    provider: str
    results: List[ItemOut]
    errors: List[Dict[str, Any]]

# =========================
# Utilitários
# =========================
def clean_text(texto: str) -> str:
    t = (texto or "").strip()
    # normaliza espaços
    t = re.sub(r"\s+", " ", t)
    return t

def clamp_conf(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.5
    return max(0.0, min(1.0, v))

def summarize(texto: str, max_len: int = 120) -> str:
    t = clean_text(texto)
    if not t:
        return "Mensagem sem texto"
    return t if len(t) <= max_len else t[:max_len - 3] + "..."

def classify_rules(texto: str) -> Tuple[str, str, str, float]:
    t = clean_text(texto).lower()

    if not t:
        return ("Outros", "Mensagem sem texto", "Solicitar_dados_complementares", 0.35)

    # Falta de água
    if re.search(r"\b(sem água|sem agua|falta de água|falta de agua)\b", t):
        return ("Falta_dagua", "Cliente relata falta de água", "Abrir_verificacao_rede", 0.93)

    # Vazamento
    if re.search(r"\bvazamento\b|\bvaza(mento|ndo|r)?\b", t):
        return ("Vazamento", "Cliente relata vazamento", "Abrir_verificacao_rede", 0.92)

    # Cobrança / conta
    if re.search(r"\b(cobrança|cobranca|conta|2a via|segunda via|fatura|boleto)\b", t):
        return ("Cobranca", "Solicitação relacionada à cobrança/conta", "Direcionar_comercial", 0.90)

    # Religação / corte
    if re.search(r"\b(religa(ção|cao|r)|corte|ligação cortada|ligacao cortada)\b", t):
        return ("Religacao", "Solicitação sobre religação/corte", "Direcionar_comercial", 0.88)

    # Atendimento geral
    if re.search(r"\b(obrigad[oa]|agradeço|agradeco|bom dia|boa tarde|boa noite)\b", t):
        return ("Atendimento", "Interação geral de atendimento", "Encerrar_sem_acao", 0.78)

    return ("Outros", summarize(texto), "Solicitar_dados_complementares", 0.60)

def ensure_allowed(categoria: str, acao: str, confianca: float):
    cat = categoria if categoria in CATEGORIAS else "Outros"
    act = acao if acao in ACOES else "Solicitar_dados_complementares"
    conf = clamp_conf(confianca)
    return cat, act, conf

def auth_or_401(x_api_key: str):
    # Se API_KEY estiver configurada, exige autenticação.
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    # Endpoint leve para healthcheck do Railway
    return {"ok": True, "service": SERVICE_NAME}

@app.get("/ready")
def ready():
    # Prontidão básica (pode evoluir depois com checagens extras)
    return {"ok": True, "ready": True}

@app.post("/classify-batch", response_model=BatchOut)
def classify_batch(payload: BatchIn, x_api_key: str = Header(default="")):
    auth_or_401(x_api_key)

    if not payload.items:
        return BatchOut(ok=True, provider="rules", results=[], errors=[])

    # Deduplicação por messageId no próprio lote
    seen = set()
    dedup_items: List[ItemIn] = []
    dup_count = 0
    for item in payload.items:
        if item.messageId in seen:
            dup_count += 1
            continue
        seen.add(item.messageId)
        dedup_items.append(item)

    results: List[ItemOut] = []
    errors: List[Dict[str, Any]] = []

    for it in dedup_items:
        try:
            categoria, resumo, acao, confianca = classify_rules(it.texto or "")
            categoria, acao, confianca = ensure_allowed(categoria, acao, confianca)

            results.append(ItemOut(
                messageId=it.messageId,
                categoria=categoria,
                resumo=resumo,
                acao=acao,
                confianca=confianca
            ))
        except Exception as e:
            logger.exception("Erro ao classificar messageId=%s", it.messageId)
            errors.append({"messageId": it.messageId, "erro": str(e)})

    logger.info(
        "Batch processado | recebidos=%d deduplicados=%d duplicados=%d sucesso=%d erros=%d",
        len(payload.items), len(dedup_items), dup_count, len(results), len(errors)
    )

    return BatchOut(
        ok=True,
        provider="rules",
        results=results,
        errors=errors
    )

# Ponto de entrada local (útil para rodar fora do Railway)
# Boas práticas do __main__ seguem a convenção do Python. <sources>[2]</sources>
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
