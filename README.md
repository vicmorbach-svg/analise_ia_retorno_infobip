# IA Classifier API

## Endpoints
- GET /health
- POST /classify-batch

## Request exemplo
```json
{
  "items": [
    {
      "messageId": "abc-1",
      "texto": "Estou sem água desde ontem",
      "from": "555199999999",
      "receivedAt": "2026-07-06T12:00:00Z",
      "numLigacao": "2235474"
    }
  ]
}
```

## Response exemplo
```json
{
  "ok": true,
  "provider": "rules",
  "results": [
    {
      "messageId": "abc-1",
      "categoria": "Falta_dagua",
      "resumo": "Cliente relata falta de água",
      "acao": "Abrir_verificacao_rede",
      "confianca": 0.93
    }
  ],
  "errors": []
}
```
