from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, json
from llama_cpp import Llama

API_KEY = os.getenv("API_KEY", "")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.gguf")

app = FastAPI(title="Classifier API")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=int(os.getenv("N_THREADS", "4")),
    verbose=False
)

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

class ItemIn(BaseModel):
    messageId: str
    texto: str
    from_: Optional[str] = None
    receivedAt: Optional[str] = None
    numLigacao: Optional[str] = None

class BatchIn(BaseModel):
    items: List[ItemIn]

@app.get("/health")
def health():
    return {"ok": True}

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

    out = llm.create_chat_completion(
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        max_tokens=220
    )
    content = out["choices"][0]["message"]["content"].strip()

    # fallback defensivo
    try:
        obj = json.loads(content)
    except:
        obj = {
            "categoria": "Outros",
            "resumo": (texto[:120] + "...") if len(texto) > 120 else texto,
            "acao": "Solicitar_dados_complementares",
            "confianca": 0.3
        }

    if obj.get("categoria") not in CATEGORIAS:
        obj["categoria"] = "Outros"
    if obj.get("acao") not in ACOES:
        obj["acao"] = "Solicitar_dados_complementares"

    try:
        conf = float(obj.get("confianca", 0.5))
    except:
        conf = 0.5
    obj["confianca"] = max(0.0, min(1.0, conf))

    return obj

@app.post("/classify-batch")
def classify_batch(payload: BatchIn, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    results = []
    errors = []

    for it in payload.items:
        try:
            cls = classify_one(it.texto or "")
            results.append({
                "messageId": it.messageId,
                "categoria": cls["categoria"],
                "resumo": cls["resumo"],
                "acao": cls["acao"],
                "confianca": cls["confianca"]
            })
        except Exception as e:
            errors.append({"messageId": it.messageId, "erro": str(e)})

    return {"ok": True, "results": results, "errors": errors}
