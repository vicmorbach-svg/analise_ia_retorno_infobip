import os
import re
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

import threading
import requests

MODEL_PATH = Path(os.getenv("LLAMA_MODEL_PATH", "/data/models/model.gguf"))
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

model_bootstrap = {
    "started": False,
    "done": False,
    "ok": False,
    "error": None
}

def download_model_if_needed():
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # já existe e não está vazio
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            model_bootstrap["ok"] = True
            model_bootstrap["done"] = True
            return

        if not HF_MODEL_URL:
            raise RuntimeError("HF_MODEL_URL não definido")

        tmp_path = MODEL_PATH.with_suffix(".part")
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        with requests.get(HF_MODEL_URL, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(MODEL_PATH)  # atomic move
        model_bootstrap["ok"] = True
    except Exception as e:
        model_bootstrap["error"] = str(e)
        model_bootstrap["ok"] = False
    finally:
        model_bootstrap["done"] = True

# =========================
# Configuração
# =========================
API_KEY = os.getenv("API_KEY", "")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ia-classifier")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# provider: local_llm | rules
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local_llm").lower()

# LLM local (llama-cpp)
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "/app/models/model.gguf")
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "4"))

# Inferência
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "260"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "1.2"))

# Prompt versionado
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
PROMPTS_DIR = Path(__file__).parent / "prompts"

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
    "Golpe",
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
    version="2.0.0",
    description="Classificador de mensagens para Power Automate (LLM local)"
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
    promptVersion: str
    results: List[ItemOut]
    errors: List[Dict[str, Any]]

# =========================
# Prompt (arquivos)
# =========================
DEFAULT_SYSTEM_PROMPT = """Você é um classificador de mensagens de cobrança.
Responda APENAS JSON válido, sem markdown e sem texto extra.

Formato obrigatório:
{
  "categoria": "string",
  "resumo": "string",
  "acao": "string",
  "confianca": 0.0
}

Categorias permitidas:
- Pagamento Imediato
- Promessa de Pagamento
- Negociação
- Incapacidade
- Outros / Sem Contexto
- Já pagou
- Não reconhece a dívida
- Questiona o valor
- Já fez contato
- Outros serviços
- Débito em conta
- Saudação
- Canais críticos
- Cadastro atualizado
- Problemas nos canais
- Golpe

Ações permitidas:
- Abrir_verificacao_rede
- Direcionar_comercial
- Solicitar_dados_complementares
- Encerrar_sem_acao

Regras:
- Se houver dúvida, use "Outros / Sem Contexto".
- resumo objetivo em até 140 caracteres.
- confianca entre 0 e 1.
- nunca invente dados.
"""

DEFAULT_USER_TEMPLATE = """Classifique a mensagem abaixo.

messageId: {messageId}
from: {from_}
receivedAt: {receivedAt}
numLigacao: {numLigacao}
texto: {texto}
"""

def load_prompt_file(filename: str, fallback: str) -> str:
    p = PROMPTS_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return fallback

SYSTEM_PROMPT = load_prompt_file(f"{PROMPT_VERSION}_system.txt", DEFAULT_SYSTEM_PROMPT)
USER_TEMPLATE = load_prompt_file(f"{PROMPT_VERSION}_user.txt", DEFAULT_USER_TEMPLATE)

def build_user_prompt(item: ItemIn) -> str:
    return USER_TEMPLATE.format(
        messageId=item.messageId or "",
        from_=(item.from_ or ""),
        receivedAt=(item.receivedAt or ""),
        numLigacao=(item.numLigacao or ""),
        texto=(item.texto or "")
    )

# =========================
# LLM local (lazy)
# =========================
_llm = None
_llm_error = None

def get_llm():
    global _llm, _llm_error
    if _llm is not None:
        return _llm
    try:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_N_CTX,
            n_threads=LLAMA_N_THREADS,
            verbose=False
        )
        return _llm
    except Exception as e:
        _llm_error = str(e)
        logger.exception("Falha ao carregar LLM local")
        return None

# =========================
# Utilitários
# =========================
def clean_text(texto: str) -> str:
    t = (texto or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def clamp_conf(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.5
    return max(0.0, min(1.0, v))

def summarize(texto: str, max_len: int = 140) -> str:
    t = clean_text(texto)
    if not t:
        return "Mensagem sem texto"
    return t if len(t) <= max_len else t[:max_len - 3] + "..."

def ensure_allowed(categoria: str, acao: str, confianca: float):
    cat = categoria if categoria in CATEGORIAS else "Outros / Sem Contexto"
    act = acao if acao in ACOES else "Solicitar_dados_complementares"
    conf = clamp_conf(confianca)
    return cat, act, conf

def auth_or_401(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # tentativa direta
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # extração por bloco { ... }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

# =========================
# Regras de fallback
# =========================
def classify_rules(texto: str) -> Tuple[str, str, str, float]:
    t = clean_text(texto).lower()

    if not t:
        return ("Outros / Sem Contexto", "Mensagem sem texto", "Solicitar_dados_complementares", 0.35)

    if re.search(r"\b(pago agora|vou pagar agora|pix agora|já vou pagar)\b", t):
        return ("Pagamento Imediato", "Cliente indica pagamento imediato", "Encerrar_sem_acao", 0.93)

    if re.search(r"\b(vou pagar|prometo pagar|pago dia|até dia)\b", t):
        return ("Promessa de Pagamento", "Cliente promete pagamento em data futura", "Encerrar_sem_acao", 0.88)

    if re.search(r"\b(parcelar|negociar|acordo|desconto)\b", t):
        return ("Negociação", "Cliente solicita negociação de dívida", "Direcionar_comercial", 0.90)

    if re.search(r"\b(desempregado|sem dinheiro|não tenho como pagar|incapaz de pagar)\b", t):
        return ("Incapacidade", "Cliente relata incapacidade de pagamento", "Direcionar_comercial", 0.86)

    if re.search(r"\b(já paguei|pagamento efetuado|paguei)\b", t):
        return ("Já pagou", "Cliente afirma já ter pago", "Solicitar_dados_complementares", 0.91)

    if re.search(r"\b(não reconheço|não fiz|não é minha dívida|nao reconheco)\b", t):
        return ("Não reconhece a dívida", "Cliente não reconhece a dívida", "Direcionar_comercial", 0.90)

    if re.search(r"\b(valor errado|valor indevido|juros abusivos|questiono o valor)\b", t):
        return ("Questiona o valor", "Cliente questiona valor cobrado", "Direcionar_comercial", 0.89)

    if re.search(r"\b(já falei|já entrei em contato|já liguei)\b", t):
        return ("Já fez contato", "Cliente informa contato anterior", "Solicitar_dados_complementares", 0.80)

    if re.search(r"\b(débito em conta|debito em conta)\b", t):
        return ("Débito em conta", "Assunto relacionado a débito em conta", "Direcionar_comercial", 0.85)

    if re.search(r"\b(bom dia|boa tarde|boa noite|olá|ola|oi)\b", t):
        return ("Saudação", "Mensagem de saudação", "Encerrar_sem_acao", 0.75)

    if re.search(r"\b(procon|reclame aqui|justiça|advogado|processo)\b", t):
        return ("Canais críticos", "Cliente menciona canal crítico", "Direcionar_comercial", 0.94)

    if re.search(r"\b(atualizei cadastro|cadastro atualizado|dados atualizados)\b", t):
        return ("Cadastro atualizado", "Cliente informa cadastro atualizado", "Encerrar_sem_acao", 0.82)

    if re.search(r"\b(site fora|app não funciona|não consigo acessar|erro no sistema)\b", t):
        return ("Problemas nos canais", "Cliente relata falha em canal digital", "Direcionar_comercial", 0.87)

    if re.search(r"\b(golpe|fraude|mensagem falsa|link suspeito)\b", t):
        return ("Golpe", "Cliente relata suspeita de golpe/fraude", "Direcionar_comercial", 0.95)

    if re.search(r"\b(segunda via|religação|vazamento|falta de água|corte|conta de água)\b", t):
        return ("Outros serviços", "Mensagem sobre serviço fora de cobrança", "Direcionar_comercial", 0.78)

    return ("Outros / Sem Contexto", summarize(texto), "Solicitar_dados_complementares", 0.60)

# =========================
# Classificação LLM local
# =========================
def call_local_llm(messages: List[Dict[str, str]]) -> str:
    llm = get_llm()
    if llm is None:
        raise RuntimeError(f"LLM indisponível: {_llm_error or 'erro desconhecido'}")

    resp = llm.create_chat_completion(
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()

def call_local_llm_with_retry(messages: List[Dict[str, str]]) -> str:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_local_llm(messages)
        except Exception as e:
            last_error = e
            wait_s = min(8.0, RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.3))
            logger.warning("Falha LLM tentativa %d/%d: %s", attempt + 1, MAX_RETRIES, str(e))
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_s)
    raise RuntimeError(f"Falha LLM após retries: {last_error}")

def classify_llm(item: ItemIn) -> Tuple[str, str, str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)},
    ]
    raw = call_local_llm_with_retry(messages)
    obj = extract_json_object(raw)

    if obj is None:
        # fallback se o modelo devolver texto fora do formato
        return classify_rules(item.texto or "")

    categoria = obj.get("categoria", "Outros / Sem Contexto")
    resumo = obj.get("resumo", summarize(item.texto or ""))
    acao = obj.get("acao", "Solicitar_dados_complementares")
    confianca = obj.get("confianca", 0.5)

    categoria, acao, confianca = ensure_allowed(categoria, acao, confianca)
    resumo = summarize(str(resumo), max_len=140)

    return categoria, resumo, acao, confianca

def classify_item(item: ItemIn) -> Tuple[str, str, str, float, str]:
    if MODEL_PROVIDER == "local_llm":
        if not MODEL_PATH.exists():
            c, r, a, conf = classify_rules(item.texto or "")
            return c, r, a, conf, "rules-waiting-model"

    try:
        c, r, a, conf = classify_llm(item)
        return c, r, a, conf, "local_llm"
    except Exception as e:
        logger.error("LLM falhou, fallback para rules. messageId=%s erro=%s", item.messageId, str(e))
        c, r, a, conf = classify_rules(item.texto or "")
        return c, r, a, conf, "rules-fallback"

# =========================
# Endpoints
# =========================
@app.on_event("startup")
def startup_event():
    if AUTO_DOWNLOAD and not model_bootstrap["started"]:
        model_bootstrap["started"] = True
        t = threading.Thread(target=download_model_if_needed, daemon=True)
        t.start()

@app.get("/health")
def health():
    # Endpoint leve para healthcheck do Railway
    return {"ok": True, "service": SERVICE_NAME, "provider": MODEL_PROVIDER}

@app.get("/ready")
def ready():
    model_exists = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0
    llm_loaded = get_llm() is not None if model_exists and MODEL_PROVIDER == "local_llm" else False

    return {
        "ok": True,
        "ready": bool(model_exists and llm_loaded),
        "provider": MODEL_PROVIDER,
        "promptVersion": PROMPT_VERSION,
        "modelExists": model_exists,
        "bootstrap": model_bootstrap,
        "llmLoaded": llm_loaded,
        "llmError": _llm_error
    }

@app.post("/classify-batch", response_model=BatchOut)
def classify_batch(payload: BatchIn, x_api_key: str = Header(default="")):
    auth_or_401(x_api_key)

    if not payload.items:
        return BatchOut(ok=True, provider=MODEL_PROVIDER, promptVersion=PROMPT_VERSION, results=[], errors=[])

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
    providers_used = set()

    for it in dedup_items:
        try:
            categoria, resumo, acao, confianca, provider_used = classify_item(it)
            categoria, acao, confianca = ensure_allowed(categoria, acao, confianca)

            results.append(ItemOut(
                messageId=it.messageId,
                categoria=categoria,
                resumo=resumo,
                acao=acao,
                confianca=confianca
            ))
            providers_used.add(provider_used)
        except Exception as e:
            logger.exception("Erro ao classificar messageId=%s", it.messageId)
            errors.append({"messageId": it.messageId, "erro": str(e)})

    provider_label = ",".join(sorted(providers_used)) if providers_used else MODEL_PROVIDER

    logger.info(
        "Batch processado | recebidos=%d deduplicados=%d duplicados=%d sucesso=%d erros=%d provider=%s prompt=%s",
        len(payload.items), len(dedup_items), dup_count, len(results), len(errors), provider_label, PROMPT_VERSION
    )

    return BatchOut(
        ok=True,
        provider=provider_label,
        promptVersion=PROMPT_VERSION,
        results=results,
        errors=errors
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
