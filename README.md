# Motor Agéntico ReAct

Práctica Calificada de Laboratorio N° 05 — Agentes Inteligentes.

Implementación de un agente **ReAct** (Reasoning + Acting) sobre Gemini con:
- Catálogo de herramientas (inventario, descuentos, etc.)
- Parseo robusto de las acciones del modelo
- *Self-healing*: el agente recibe el error de una herramienta y corrige su llamada

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y coloca tu clave de https://aistudio.google.com/apikey
```

## Uso

```bash
python practica_semana05_react.py                 # demos contra Gemini
python practica_semana05_react.py "pregunta"      # pregunta libre
python practica_semana05_react.py --test          # pruebas deterministas (sin API)
```

Variables de entorno: `GEMINI_API_KEY` (obligatoria), `GEMINI_MODEL` (opcional, por defecto `gemini-3.5-flash`).
