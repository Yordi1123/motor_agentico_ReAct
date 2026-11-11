"""
Práctica Calificada de Laboratorio N° 05 — Agentes Inteligentes
Motor Agéntico ReAct: Razonamiento, Herramientas, Self-Healing y Parseo Robusto

Requisitos:
    pip install google-genai
    export GEMINI_API_KEY="tu_clave"   # https://aistudio.google.com/apikey
    (opcional) export GEMINI_MODEL="gemini-3.5-flash"
"""

import re
import os
import sys
import json
import math
import time
import inspect
import datetime
from typing import Dict, Any, Tuple, Optional, List

STOP_SEQUENCES = ["Observation:"]
MODELO_POR_DEFECTO = "gemini-3.5-flash"

# ==========================================
# 1. CATÁLOGO DE HERRAMIENTAS Y SIMULACIÓN
# ==========================================

BD_INVENTARIO = {
    "PROD-001": {"nombre": "Laptop Gamer", "stock": 5, "precio": 1200.0},
    "PROD-002": {"nombre": "Monitor 4K", "stock": 0, "precio": 350.0}
}


def consultar_inventario(producto_id: str) -> str:
    """Consulta el stock y precio de un producto por su ID."""
    if not isinstance(producto_id, str):
        raise TypeError(f"'producto_id' debe ser texto (str), se recibió {type(producto_id).__name__}.")
    producto_id = producto_id.strip().strip("'\"`").upper()
    if not producto_id:
        raise ValueError("'producto_id' no puede estar vacío. Ejemplo válido: PROD-001.")
    if producto_id not in BD_INVENTARIO:
        raise KeyError(f"El producto con ID '{producto_id}' no existe en la base de datos. "
                       f"IDs válidos: {list(BD_INVENTARIO.keys())}")
    info = BD_INVENTARIO[producto_id]
    return f"Producto: {info['nombre']} | Stock: {info['stock']} | Precio: ${info['precio']}"


def calcular_descuento(precio: str, porcentaje: str) -> str:
    """Calcula el precio final aplicando un porcentaje de descuento."""
    try:
        p = float(str(precio).strip().replace("$", "").replace("%", ""))
        d = float(str(porcentaje).strip().replace("%", ""))
    except ValueError:
        raise ValueError("Los argumentos 'precio' y 'porcentaje' deben ser números válidos.")

    if not (math.isfinite(p) and math.isfinite(d)):
        raise ValueError("Los argumentos deben ser números finitos (no NaN ni infinito).")
    if p < 0 or d < 0 or d > 100:
        raise ValueError("El precio debe ser positivo y el descuento debe estar entre 0 y 100.")

    precio_final = p * (1 - d / 100)
    return f"Precio original: ${p:.2f} | Descuento: {d}% | Precio Final: ${precio_final:.2f}"


HERRAMIENTAS_DISPONIBLES = {
    "consultar_inventario": consultar_inventario,
    "calcular_descuento": calcular_descuento
}


# ==========================================
# 1.b CLIENTES LLM (google-genai real + simulado para pruebas)
# ==========================================

class ErrorConfiguracionLLM(Exception):
    """Falta el SDK o la API key."""


GUIA_CONFIGURACION = """
No se pudo inicializar Gemini (google-genai). Pasos:
  1. pip install google-genai          (o: .venv/bin/pip install google-genai)
  2. Crear una clave en https://aistudio.google.com/apikey  (botón 'Create API key')
  3. export GEMINI_API_KEY="tu_clave"   (o escribir GEMINI_API_KEY=tu_clave en un archivo .env junto al script)
  4. Volver a ejecutar. Para probar sin API: python practica_semana05_react.py --test
"""


def cargar_env_local(ruta: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) -> None:
    """Carga variables KEY=VALUE de un .env local (si existe) sin sobrescribir las del entorno."""
    if not os.path.isfile(ruta):  # si no existe o es una carpeta, se ignora
        return
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


class ClienteLLMGemini:
    """Cliente real con el SDK oficial google-genai. Pausa la generación en 'Observation:'."""

    def __init__(self, modelo: Optional[str] = None, max_reintentos: int = 3):
        cargar_env_local()
        try:
            from google import genai
            from google.genai import types, errors
        except ImportError as e:
            raise ErrorConfiguracionLLM(f"SDK google-genai no instalado ({e}).")
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise ErrorConfiguracionLLM("Variable GEMINI_API_KEY no definida.")
        self._types = types
        self._errors = errors
        self.client = genai.Client()
        self.modelo = modelo or os.environ.get("GEMINI_MODEL", MODELO_POR_DEFECTO)
        self.max_reintentos = max_reintentos
        self.cuota_diaria_agotada = False

    def generar(self, system_prompt: str, historial: str, stop_sequences: List[str]) -> str:
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            stop_sequences=stop_sequences,   # <- el modelo cede el control al entorno aquí
            temperature=0.0,
            max_output_tokens=2048,
        )
        if self.cuota_diaria_agotada:
            raise RuntimeError(f"Cuota diaria gratuita agotada para {self.modelo}; "
                               "usa otro modelo con GEMINI_MODEL o intenta mañana.")
        espera = 10
        for intento in range(1, self.max_reintentos + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.modelo, contents=historial, config=config
                )
                return resp.text or ""
            except self._errors.APIError as e:
                codigo = getattr(e, "code", None)
                if codigo == 429 and "PerDay" in str(e):
                    # Cuota diaria: reintentar solo gastaría más solicitudes.
                    self.cuota_diaria_agotada = True
                    raise
                # 429 por minuto / 5xx transitorios: reintento corto con espera creciente.
                if codigo in (429, 500, 503) and intento < self.max_reintentos:
                    print(f"   [API] Error {codigo}; reintentando en {espera}s...")
                    time.sleep(espera)
                    espera *= 2
                    continue
                raise
        return ""


class ClienteLLMSimulado:
    """LLM guionizado para pruebas deterministas. También respeta las stop sequences."""

    def __init__(self, respuestas: List[str]):
        self.respuestas = list(respuestas)
        self.llamadas: List[Dict[str, Any]] = []

    def generar(self, system_prompt: str, historial: str, stop_sequences: List[str]) -> str:
        self.llamadas.append({"stop_sequences": stop_sequences, "system_prompt": system_prompt})
        texto = self.respuestas.pop(0) if self.respuestas else "Thought: sigo pensando...\nAction: consultar_inventario\nAction Input: PROD-001"
        # Simula el comportamiento de la API: corta en la primera stop sequence.
        for stop in stop_sequences:
            idx = texto.find(stop)
            if idx != -1:
                texto = texto[:idx]
        return texto


# ==========================================
# 2. MOTOR DE PARSEO Y AUTOCORRECCIÓN
# ==========================================

_ETIQUETAS = r"(?:Thought|Action\s*Input|Action|Observation|Final\s*Answer)"


def _limpiar_valor(valor: Optional[str]) -> Optional[str]:
    """Quita espacios, backticks y comillas envolventes accidentales."""
    if valor is None:
        return None
    v = valor.strip()
    v = re.sub(r"^`+|`+$", "", v).strip()
    while len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v or None


class MotorReActEstudiante:
    def __init__(self, max_pasos: int = 5, cliente_llm: Any = None, verbose: bool = True):
        self.max_pasos = max_pasos
        self.historial_contexto: str = ""
        self.cliente_llm = cliente_llm
        self.verbose = verbose

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # ---------- reloj_sistema ----------
    def obtener_reloj_sistema(self) -> str:
        """TAREA 1 (Ej 2): Obtener timestamp actual del sistema."""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def construir_system_prompt(self) -> str:
        timestamp = self.obtener_reloj_sistema()
        return f"""Eres un Agente Autónomo ReAct. Tu deber es resolver consultas respondiendo
estrictamente en este formato:

Thought: <tu razonamiento sobre qué hacer>
Action: <nombre de la herramienta a usar>
Action Input: <argumentos para la herramienta>
Observation: <resultado que el entorno te devolverá>

Cuando tengas la respuesta final, responde exactamente con:
Thought: <razonamiento final>
Final Answer: <respuesta clara y directa al usuario>

Reglas:
- Propón UNA sola Action por turno y detente; NUNCA escribas tú la Observation.
- Action Input: valores separados por comas (ej: 1200, 15) o JSON (ej: {{"precio": 1200, "porcentaje": 15}}).
- Si recibes [ERROR DE EJECUCIÓN], reflexiona en el siguiente Thought y corrige la llamada.
- Para preguntas sobre fecha u hora usa el [ESTADO_SISTEMA]; no necesitas herramientas.

Herramientas disponibles:
- consultar_inventario: Argumento (producto_id: str)
- calcular_descuento: Argumentos (precio, porcentaje)

[ESTADO_SISTEMA]: Timestamp actual = {timestamp}
"""

    # ---------- Parseo robusto ----------
    def parsear_salida_llm(self, texto_salida: str) -> Dict[str, Optional[str]]:
        """TAREA 2 (Ej 2): Parseo mediante RegEx tolerante a Markdown y variaciones."""
        res = {"thought": None, "action": None, "action_input": None, "final_answer": None}
        if not texto_salida:
            return res

        texto = texto_salida.replace("\r\n", "\n")
        # Eliminar cercados de código ```lang ... ``` (se conserva el contenido)
        texto = re.sub(r"```[a-zA-Z0-9_-]*[ \t]*", "", texto)
        # Normalizar etiquetas en negrita/cursiva/encabezado: **Action:** / *Thought*: / ## Thought:
        texto = re.sub(r"^[ \t>#-]*[*_]{0,2}[ \t]*(" + _ETIQUETAS + r")[ \t]*[*_]{0,2}[ \t]*:[ \t]*[*_]{0,2}",
                       lambda m: m.group(1) + ": ", texto, flags=re.M | re.I)

        def extraer(etiqueta: str) -> Optional[str]:
            patron = (r"^[ \t]*" + etiqueta + r"[ \t]*:[ \t]*(.*?)"
                      r"(?=^[ \t]*" + _ETIQUETAS + r"[ \t]*:|\Z)")
            m = re.search(patron, texto, flags=re.S | re.M | re.I)
            return _limpiar_valor(m.group(1)) if m else None

        res["final_answer"] = extraer(r"Final\s*Answer")
        res["thought"] = extraer(r"Thought")
        if res["final_answer"] is not None:
            return res

        accion = extraer(r"Action(?!\s*Input)")
        if accion:
            # Solo la primera palabra identificador (tolera "consultar_inventario()" o texto extra)
            m = re.match(r"[`'\"]*([a-zA-Z_][a-zA-Z0-9_]*)", accion)
            res["action"] = m.group(1) if m else accion
        res["action_input"] = extraer(r"Action\s*Input")
        return res

    # ---------- Argumentos ----------
    @staticmethod
    def _parsear_argumentos(args_str: Optional[str]) -> Tuple[list, dict]:
        """Convierte Action Input en (args, kwargs). Acepta JSON (dict/list/escalar) o CSV."""
        if args_str is None:
            return [], {}
        s = args_str.strip()
        try:
            dato = json.loads(s)
            if isinstance(dato, dict):
                return [], dato
            if isinstance(dato, list):
                return dato, {}
            return [dato], {}
        except (json.JSONDecodeError, ValueError):
            pass
        # Formato tipo llamada: f(a, b) -> a, b
        m = re.fullmatch(r"[a-zA-Z_]\w*\s*\((.*)\)", s, flags=re.S)
        if m:
            s = m.group(1)
        args, kwargs = [], {}
        for parte in (p.strip() for p in s.split(",")):
            if not parte:
                continue
            kv = re.fullmatch(r"([a-zA-Z_]\w*)\s*[=:]\s*(.+)", parte)
            if kv:
                kwargs[kv.group(1)] = _limpiar_valor(kv.group(2))
            else:
                args.append(_limpiar_valor(parte))
        return args, kwargs

    # ---------- Self-Healing ----------
    def ejecutar_herramienta_con_self_healing(self, nombre_herramienta: Optional[str], args_str: Optional[str]) -> str:
        """TAREA 3 (Ej 1): Ejecutar herramienta y capturar excepciones para Self-Healing."""
        sufijo = "Por favor corrige los argumentos o intenta otra herramienta."
        if not nombre_herramienta or nombre_herramienta not in HERRAMIENTAS_DISPONIBLES:
            return (f"Observation: [ERROR DE EJECUCIÓN] La herramienta '{nombre_herramienta}' no existe. "
                    f"Herramientas válidas: {list(HERRAMIENTAS_DISPONIBLES.keys())}. {sufijo}")
        try:
            func = HERRAMIENTAS_DISPONIBLES[nombre_herramienta]
            args, kwargs = self._parsear_argumentos(args_str)
            firma = inspect.signature(func)
            try:
                firma.bind(*args, **kwargs)
            except TypeError as e:
                raise TypeError(f"Firma inválida para {nombre_herramienta}{firma}: {e}")
            resultado = func(*args, **kwargs)
            return f"Observation: [ÉXITO] {resultado}"
        except Exception as e:  # KeyError, ValueError, TypeError, cualquier otra
            mensaje = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            return f"Observation: [ERROR DE EJECUCIÓN] {type(e).__name__}: {str(mensaje).rstrip('.')}. {sufijo}"

    # ---------- Bucle ReAct ----------
    def ejecutar_bucle_react(self, pregunta_usuario: str) -> str:
        """Bucle ReAct con stop sequences, self-healing y Failsafe por límite de pasos."""
        if self.cliente_llm is None:
            raise ErrorConfiguracionLLM("No se configuró un cliente LLM.")
        self._log(f"\n=== PROCESANDO PREGUNTA: '{pregunta_usuario}' ===")
        system_prompt = self.construir_system_prompt()
        estado = next(l for l in system_prompt.splitlines() if l.startswith("[ESTADO_SISTEMA]"))
        self._log(f"Inyección sensorial -> {estado}")
        self.historial_contexto = f"Usuario: {pregunta_usuario}\n"
        ultima_observacion = "ninguna"

        for paso in range(1, self.max_pasos + 1):
            self._log(f"\n--- PASO {paso} de {self.max_pasos} ---")
            try:
                salida = self.cliente_llm.generar(system_prompt, self.historial_contexto, STOP_SEQUENCES)
            except Exception as e:
                respuesta = (f"Final Answer: [FAILSAFE ACTIVADO] Error al comunicarse con el LLM "
                             f"({type(e).__name__}: {e}). No se pudo completar la consulta.")
                self._log(respuesta)
                return respuesta

            salida = salida.rstrip()
            self._log(f"[LLM]\n{salida}")
            self.historial_contexto += salida + "\n"
            parseo = self.parsear_salida_llm(salida)

            if parseo["final_answer"] is not None:
                respuesta = f"Final Answer: {parseo['final_answer']}"
                self._log(f"\n>>> {respuesta}")
                return respuesta

            if parseo["action"] is None:
                observacion = ("Observation: [ERROR DE EJECUCIÓN] Formato inválido: no se encontró 'Action:' "
                               "ni 'Final Answer:'. Responde con Thought/Action/Action Input o con Final Answer. "
                               "Por favor corrige los argumentos o intenta otra herramienta.")
            else:
                observacion = self.ejecutar_herramienta_con_self_healing(parseo["action"], parseo["action_input"])

            ultima_observacion = observacion
            self._log(f"[ENTORNO] {observacion}")
            self.historial_contexto += observacion + "\n"

        respuesta = (f"Final Answer: [FAILSAFE ACTIVADO] Se ha alcanzado el límite máximo de "
                     f"{self.max_pasos} pasos sin una respuesta final. Última observación: {ultima_observacion}")
        self._log(f"\n>>> {respuesta}")
        return respuesta


# ==========================================
# 3. PRUEBAS DETERMINISTAS (sin API)
# ==========================================

def ejecutar_pruebas() -> None:
    motor = MotorReActEstudiante(verbose=False)

    # --- Parseo robusto ---
    casos = [
        ("Thought: busco\nAction: consultar_inventario\nAction Input: PROD-001",
         "consultar_inventario", "PROD-001"),
        ("```\nThought: busco\nAction: `consultar_inventario`\nAction Input: \"PROD-001\"\n```",
         "consultar_inventario", "PROD-001"),
        ("**Thought:** pienso\n**Action:** calcular_descuento\n**Action Input:**   1200, 15  ",
         "calcular_descuento", "1200, 15"),
        ("```text\n  thought :  pienso\n  action :   calcular_descuento  \n  action input : {\"precio\": 1200, \"porcentaje\": 15}\n```",
         "calcular_descuento", '{"precio": 1200, "porcentaje": 15}'),
    ]
    for texto, accion, entrada in casos:
        r = motor.parsear_salida_llm(texto)
        assert r["action"] == accion, (texto, r)
        assert r["action_input"] == entrada, (texto, r)
        assert r["thought"], (texto, r)
    r = motor.parsear_salida_llm("```\nThought: listo\nFinal Answer: 'El precio es $1020.00'\n```")
    assert r["final_answer"] == "El precio es $1020.00", r
    print("[OK] Parseo robusto (Markdown, negritas, comillas, espacios, JSON, Final Answer)")

    # --- Self-Healing ---
    eh = motor.ejecutar_herramienta_con_self_healing
    assert eh("consultar_inventario", "PROD-001").startswith("Observation: [ÉXITO]")
    assert "KeyError" in eh("consultar_inventario", "PROD-999")
    assert "ValueError" in eh("calcular_descuento", "mil, 10")
    assert "ValueError" in eh("calcular_descuento", "100, 150")
    assert "TypeError" in eh("calcular_descuento", "100")                 # firma inválida
    assert "no existe" in eh("buscar_producto", "PROD-001")              # herramienta inexistente
    assert "[ÉXITO]" in eh("calcular_descuento", '{"precio": 1200, "porcentaje": 15}')
    assert "1020.00" in eh("calcular_descuento", "precio=1200, porcentaje=15")
    for obs in (eh("consultar_inventario", "X"), eh("nada", "")):
        assert obs.startswith("Observation: [ERROR DE EJECUCIÓN]") and "corrige" in obs
    print("[OK] Self-Healing (KeyError, ValueError, firma inválida, herramienta inexistente)")

    # --- Bucle completo con autocorrección + stop sequences ---
    sim = ClienteLLMSimulado([
        "Thought: busco el producto\nAction: buscar_producto\nAction Input: PROD-001\nObservation: inventado",
        "Thought: esa herramienta no existe\nAction: consultar_inventario\nAction Input: PROD-1",
        "Thought: el ID era incorrecto\nAction: consultar_inventario\nAction Input: PROD-001",
        "```\nThought: aplico descuento\n**Action:** calcular_descuento\nAction Input: 1200, 15\n```",
        "Thought: ya tengo todo\nFinal Answer: La Laptop Gamer cuesta $1020.00 con 15% de descuento.",
    ])
    motor = MotorReActEstudiante(max_pasos=5, cliente_llm=sim, verbose=False)
    r = motor.ejecutar_bucle_react("Precio de PROD-001 con 15%")
    assert r == "Final Answer: La Laptop Gamer cuesta $1020.00 con 15% de descuento.", r
    assert all(c["stop_sequences"] == ["Observation:"] for c in sim.llamadas)
    assert "inventado" not in motor.historial_contexto        # el LLM no pudo autocompletar Observation
    assert motor.historial_contexto.count("[ERROR DE EJECUCIÓN]") == 2
    assert "[ESTADO_SISTEMA]: Timestamp actual = " in sim.llamadas[0]["system_prompt"]
    assert re.search(r"Timestamp actual = \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", sim.llamadas[0]["system_prompt"])
    print("[OK] Bucle ReAct con stop_sequences=['Observation:'], autocorrección y reloj_sistema")

    # --- Failsafe ---
    sim = ClienteLLMSimulado([])  # nunca emite Final Answer
    motor = MotorReActEstudiante(max_pasos=5, cliente_llm=sim, verbose=False)
    r = motor.ejecutar_bucle_react("pregunta infinita")
    assert r.startswith("Final Answer: [FAILSAFE ACTIVADO] Se ha alcanzado el límite máximo"), r
    assert len(sim.llamadas) == 5
    print("[OK] Failsafe: interrupción limpia en el paso 5")

    # --- Fallo de la API -> degradación elegante ---
    class ClienteRoto:
        def generar(self, *a):
            raise ConnectionError("sin red")
    r = MotorReActEstudiante(cliente_llm=ClienteRoto(), verbose=False).ejecutar_bucle_react("x")
    assert "[FAILSAFE ACTIVADO]" in r
    print("[OK] Error de API degradado sin traceback")
    print("\nTodas las pruebas pasaron.")


# ==========================================
# 4. DEMOSTRACIÓN CONTRA GEMINI
# ==========================================

PREGUNTAS_DEMO = [
    ("¿Cuánto cuesta la Laptop Gamer (PROD-001) si le aplico un 15% de descuento?", 5),
    ("Consulta el producto PROD-999; si no existe, revisa el PROD-002 y dime si hay stock.", 5),
    ("Aplica un descuento del 150% al Monitor 4K (PROD-002) y dime el precio final.", 5),
    ("¿Qué fecha y hora es ahora mismo según el sistema?", 5),
    # Failsafe: requiere más pasos de los permitidos
    ("Consulta PROD-001, luego PROD-002, y calcula el precio de ambos con 10%, 20% y 30% de descuento.", 2),
]


def main() -> None:
    argumentos = sys.argv[1:]
    if "--test" in argumentos:
        ejecutar_pruebas()
        return

    try:
        cliente = ClienteLLMGemini()
    except ErrorConfiguracionLLM as e:
        print(f"[CONFIGURACIÓN] {e}")
        print(GUIA_CONFIGURACION)
        sys.exit(1)

    print(f"Modelo Gemini: {cliente.modelo} | stop_sequences={STOP_SEQUENCES}")
    if argumentos:
        preguntas = [(" ".join(argumentos), 5)]
    else:
        preguntas = PREGUNTAS_DEMO

    for pregunta, pasos in preguntas:
        agente = MotorReActEstudiante(max_pasos=pasos, cliente_llm=cliente)
        agente.ejecutar_bucle_react(pregunta)
        print("=" * 70)


if __name__ == "__main__":
    main()
