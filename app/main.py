from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os

API_KEY = os.getenv("API_KEY", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "rules")  # rules | llama
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_CTX = int(os.getenv("N_CTX", "4096"))
app = FastAPI()

class ItemIn(BaseModel):
    messageId: str
    texto: str = ""

class BatchIn(BaseModel):
    items: List[ItemIn]

@app.get("/health")
def health():
    return {"ok": True}

# Categorias e ações permitidas
CATEGORIAS = [
  
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
]

ACOES = [
    "Abrir_verificacao_rede",
    "Direcionar_comercial",
    "Solicitar_dados_complementares",
    "Encerrar_sem_acao"
]

_llm = None
_llm_load_error = None

def try_load_llm():
    global _llm, _llm_load_error
    if MODEL_PROVIDER != "llama":
        return
    try:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False
        )
    except Exception as e:
        _llm_load_error = str)
        _llm = None

try_load_llm()

# =========================
# Schemas
# =========================
class ItemIn(BaseModel):
    messageId: str
    texto: str = ""
    from_: Optional[str] = Field(default=None, alias="from")
    receivedAt: Optional[str] = None
    numLigacao: Optional[str] = None

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
# Classificação por regras (rápida e barata)
# =========================
def classify_rules(texto: str) -> Dict[str, Any]:
    t = (texto or "").strip().lower()

    if not t:
        return {
            "categoria": "Outros",
            "resumo": "Mensagem sem texto",
            "acao": "Solicitar_dados_complementares",
            "confianca": 0.35
        }

    # Regras simples (ajuste conforme seu negócio)
    if re.search(r"\bfalta\b.*\bágua\b|\bsem água\b|\bsem agua\b", t):
        return {
            "categoria": "Falta_dagua",
            "resumo": "Cliente relata falta de água",
            "acao": "Abrir_verificacao_rede",
            "confianca": 0.93
        }

    if re.search(r"\bvazamento\b|\bvaza\b", t):
        return {
            "categoria": "Vazamento",
            "resumo": "Cliente relata vazamento",
            "acao": "Abrir_verificacao_rede",
            "confianca": 0.92
        }

    if re.search(r"\bconta\b|\bcobrança\b|\bcobranca\b|\b2a via\b|\bsegunda via\b", t):
        return {
            "categoria": "Cobranca",
            "resumo": "Dúvida/solicitação sobre cobrança",
            "acao": "Direcionar_comercial",
            "confianca": 0.90
        }

    if re.search(r"\breliga(?:ção|cao|r)\b|\bcorte\b", t):
        return {
            "categoria": "Religacao",
            "resumo": "Solicitação sobre religação/corte",
            "acao": "Direcionar_comercial",
            "confianca": 0.88
        }

    if re.search(r"\bobrigad[oa]\b|\bagrade", t):
        return {
            "categoria": "Atendimento",
            "resumo": "Interação de atendimento geral",
            "acao": "Encerrar_sem_acao",
            "confianca": 0.80
        }

    return {
        "categoria": "Outros",
        "resumo": (texto[:120] + "...") if len(texto) > 120 else texto,
        "acao": "Solicitar_dados_complementares",
        "confianca": 0.60
    }

# =========================
# Classificação por LLM (opcional)
# =========================
def classify_one(texto: str):
    prompt = f"""
 Você é um agente especialista em análise de dados e recuperação de crédito.

TAREFA
Classifique a mensagem do cliente em APENAS UMA categoria da lista abaixo e retorne somente JSON válido.

CATEGORIAS PERMITIDAS (texto exato):
- "Pagamento Imediato"
- "Promessa de Pagamento"
- "Negociação"
- "Incapacidade"
- "Outros / Sem Contexto"
- "Já pagou"
- "Não reconhece a dívida"
- "Questiona o valor"
- "Já fez contato"
- "Outros serviços"
- "Débito em conta"
- "Saudação"
- "Canais críticos"
- "Cadastro atualizado"
- "Problemas nos canais"
- "Golpe"


REGRAS DE CLASSIFICAÇÃO
1) Pagamento Imediato: pede PIX, boleto atualizado, código de barras, diz que vai pagar hoje, pede fatura para pagar, pede reenvio da fatura.
2) Promessa de Pagamento: informa data futura para pagar.
3) Negociação: pede parcelamento ou desconto.
4) Incapacidade: diz que não tem dinheiro, está desempregado, não consegue pagar.
5) Outros / Sem Contexto: mensagem sem sentido, automática, emoji isolado, assunto fora de cobrança, opção inválida, “não entendemos sua mensagem”.
6) Já pagou: informa que já realizou o pagamento.
7) Não reconhece a dívida: diz que não é cliente, não está no nome dele, não é mais o usuário do imóvel.
8) Questiona o valor: diz que a conta está errada, revisão de fatura.
9) Já fez contato: informa contato anterior por outro canal.
10) Outros serviços: vazamento, esgoto, buraco na rua, religação, solicitações fora de cobrança.
11) Débito em conta: menciona cadastro de débito em conta.
12) Saudação: bom dia, boa tarde, boa noite, oi, olá. 
13) Canais críticos: processo, justiça, procon, agência reguladora, agergs, agesan, advogado.
14) Cadastro atualizado: quando a mensagem é 'Está atualizado'

15) Fala que o site, app, aplicativo, agência virtual, whatsapp, chat ou call center não funcionam, são ruins, não consegue usar, cita problemas ou reclama desses canais.

16) Fala em golpe, não acredita na mensagem, acha que está sendo enganado.

DESEMPATE (ordem de prioridade)
- Se mencionar processo/justiça/procon => "Canais críticos"
- Se disser que já pagou => "Já pagou"
- Se pedir PIX/boleto/código para pagar agora => "Pagamento Imediato"
- Se informar pagamento futuro => "Promessa de Pagamento"

Responda SOMENTE JSON válido no formato:
{{"categoria":"...","resumo":"...","acao":"...","confianca":0.0}}

Categorias permitidas: {CATEGORIAS}

Mensagem:
{texto}
""".strip()

    out = _llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=220
    )

    content = out["choices"][0]["message"]["content"].strip()

    try:
        obj = json.loads(content)
    except Exception:
        return classify_rules(texto)

    categoria = obj.get("categoria", "Outros")
    acao = obj.get("acao", "Solicitar_dados_complementares")
    resumo = obj.get("resumo", "")
    confianca = obj.get("confianca", 0.5)

    if categoria not in CATEGORIAS:
        categoria = "Outros"
    if acao not in ACOES:
        acao = "Solicitar_dados_complementares"

    try:
        confianca = float(confianca)
    except Exception:
        confianca = 0.5

    confianca = max(0.0, min(1.0, confianca))

    return {
        "categoria": categoria,
        "resumo": resumo if isinstance(resumo, str) and resumo.strip() else classify_rules(texto)["resumo"],
        "acao": acao,
        "confianca": confianca
    }

def classify(texto: str) -> Dict[str, Any]:
    if MODEL_PROVIDER == "llama":
        return classify_llm(texto)
    return classify_rules(texto)

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "provider": MODEL_PROVIDER,
        "llmLoaded": _llm is not None,
        "llmError": _llm_load_error
    }

@app.post("/classify-batch", response_model=BatchOut)
def classify_batch(payload: BatchIn, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    results: List[ItemOut] = []
    errors: List[Dict[str, Any]] = []

    for it in payload.items:
        try:
            cls = classify(it.texto or "")
            results.append(ItemOut(
                messageId=it.messageId,
                categoria=cls["categoria"],
                resumo=cls["resumo"],
                acao=cls["acao"],
                confianca=cls["confianca"]
            ))
        except Exception as e:
            errors.append({"messageId": it.messageId, "erro": str(e)})

    return BatchOut(
        ok=True,
        provider=MODEL_PROVIDER if (_llm is not None or MODEL_PROVIDER == "rules") else "rules",
        results=results,
        errors=errors
    )
