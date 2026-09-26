# Especificación — FAQ RAG Assistant

Documento vivo. Se actualiza **antes** de escribir el código de cada milestone. Las decisiones marcadas como *propuesta* se confirman o cambian al empezar el milestone correspondiente.

## 1. Problema

Soporte de una empresa de HR SaaS recibe más de 200 preguntas repetitivas por día que ya están respondidas en la documentación interna. El sistema responde esas preguntas a partir de un documento de texto plano, usando RAG: primero **recupera** los fragmentos relevantes del documento y después **genera** la respuesta con un LLM usando solo ese contexto.

## 2. Alcance

Incluido:

- Pipeline de indexación: documento → chunks → embeddings → almacenamiento.
- Pipeline de consulta: pregunta → embedding → búsqueda vectorial → contexto → LLM → JSON.
- Agente evaluador (bonus): puntaje 0–10 + justificación.
- CLI (`python main.py ...`), tests deterministas, README.

Opcional (después de cumplir la consigna): comparación ANN vs. k-NN, endpoint HTTP con FastAPI, backend de vector store con pgvector.

Fuera de alcance: UI, múltiples documentos, historial de conversación, actualización incremental del índice.

## 3. Arquitectura

```
                 ┌──────────── INDEXACIÓN (offline, una vez) ────────────┐
data/faq_document.txt → load_document → chunk_document → generate_embeddings → save_index → data/index/
                 └───────────────────────────────────────────────────────┘

                 ┌──────────── CONSULTA (cada pregunta) ─────────────────┐
pregunta → embed_query → search_similar_chunks → build_context → generate_answer → QueryResult (JSON)
                 └───────────────────────────────────────────────────────┘
                                                        │
                                               (bonus) evaluate_answer → Evaluation
```

## 4. Estructura del repositorio (propuesta)

```
data/faq_document.txt        # documento fuente (≥1000 palabras)
data/index/                  # generado por build_index (no se commitea)
outputs/sample_queries.json  # ≥3 ejemplos de punta a punta
src/config.py                # variables de entorno y parámetros
src/schemas.py               # modelos Pydantic: Chunk, QueryResult, Evaluation
src/embeddings.py            # wrapper de la API de embeddings (lo usan ambos pipelines)
src/vector_store.py          # guardar/cargar índice + búsqueda por similitud
src/build_index.py           # pipeline de indexación
src/query.py                 # pipeline de consulta
src/evaluator.py             # agente evaluador (bonus)
src/api.py                   # (opcional, M10) endpoint FastAPI
docker-compose.yml           # (opcional, M11) Postgres + pgvector
main.py                      # CLI: `build`, `ask`
tests/
.env.example
requirements.txt             # versiones fijadas
README.md
```

## 5. Contratos

### 5.0 Formato del documento fuente

`data/faq_document.txt` (UTF-8, ~3.400 palabras) sigue una estructura fija que el chunking puede aprovechar:

```
NEXO HR — PREGUNTAS FRECUENTES Y GUÍA DE USO      ← encabezado + párrafo de introducción
...

## Vacaciones y licencias                          ← sección: línea que empieza con "## "

P: ¿Cómo solicito vacaciones?                      ← pregunta: línea que empieza con "P: "
R: Las vacaciones se solicitan ...                 ← respuesta: empieza con "R: ", puede tener
1. Ve a Ausencias > Nueva solicitud.                  varias líneas y pasos numerados
...
```

- 11 secciones y 33 pares pregunta/respuesta (entre 129 y 192 tokens cada uno).
- Mezcla preguntas de "cómo hacer" (con pasos numerados), de sí/no y de datos puntuales (plazos, precios, límites), para cubrir los tipos de pregunta que menciona la rúbrica.
- El encabezado y el párrafo de introducción no pertenecen a ningún par: el chunking también tiene que capturarlos (100% del contenido).

### 5.1 Salida de una consulta (`QueryResult`)

Exactamente tres claves en el nivel superior (la rúbrica lo exige):

```json
{
  "user_question": "¿Cuántos días de vacaciones tengo?",
  "system_answer": "…",
  "chunks_related": [
    {"chunk_id": 7, "section": "Vacaciones", "similarity": 0.62, "text": "…"}
  ]
}
```

- `chunks_related`: entre 2 y 5 elementos.
- Validado con Pydantic (`extra="forbid"`).

### 5.2 Salida del evaluador (`Evaluation`)

```json
{"score": 8, "reason": "Puntaje 8: responde usando 3 chunks relevantes, pero …"}
```

- `score`: entero 0–10. `reason`: string de ≥50 caracteres.
- Dimensiones: relevancia de los chunks, fidelidad al contexto (sin alucinaciones), completitud.
- Se guarda **aparte** del `QueryResult` para no romper la regla de las tres claves.

## 6. Decisiones técnicas

| Tema | Propuesta | Alternativas | Se decide en |
|---|---|---|---|
| Chunking | Por estructura: un chunk por par pregunta/respuesta, con el título de la sección como prefijo. Límites 50–500 tokens; si un bloque supera el máximo se divide por oraciones. | Tamaño fijo con solapamiento; por párrafos; semántico. | M2 |
| Conteo de tokens | `tiktoken` con el encoding del modelo de embeddings. | Aproximar por palabras. | M2 |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensiones). | Sentence-Transformers local. | M3 |
| Almacenamiento | Archivos locales: `chunks.json` (texto + metadatos) y `embeddings.npy` (matriz NumPy). | FAISS, Chroma, pgvector. | M3 |
| Búsqueda | k-NN exacto con similitud coseno calculada explícitamente en NumPy, más un umbral mínimo de similitud (filtro por rango), acotado a 2–5 resultados. | ANN (FAISS/HNSW); híbrida con BM25. | M4 |
| LLM | `gpt-4o-mini` vía Responses API con Structured Output. | Otro modelo de OpenAI. | M5 |
| Evaluador | LLM-as-judge con Structured Output y rúbrica en el prompt. | Heurísticas de solapamiento de palabras. | M7 |

## 7. Parámetros (propuesta)

| Parámetro | Valor inicial | Dónde |
|---|---|---|
| `MIN_CHUNK_TOKENS` | 50 | `src/config.py` |
| `MAX_CHUNK_TOKENS` | 500 | `src/config.py` |
| `TOP_K` | 5 | `src/config.py` |
| `MIN_RESULTS` | 2 | `src/config.py` |
| `SIMILARITY_THRESHOLD` | a calibrar en M4 | `src/config.py` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `.env` |
| `OPENAI_MODEL` | `gpt-4o-mini` | `.env` |

## 8. Reglas de código (de la rúbrica)

- Funciones con un único propósito, nombres descriptivos, ≤30 líneas.
- En cada archivo: imports arriba, funciones en el medio, `main` al final.
- API key leída con `os.getenv()`; nunca hardcodeada.
- Errores explícitos: documento inexistente, encoding inválido, índice no construido, API key faltante, fallas de la API.
- Tests sin llamadas reales a OpenAI.

## 9. Milestones

Cada milestone sigue el mismo ciclo: **concepto → contrato → implementación → tests → checkpoint**.

| # | Milestone | Concepto que se aprende | Entregable | Criterio de "terminado" |
|---|---|---|---|---|
| M0 | Spec y plan | Diseño antes de código | Este documento | Decisiones revisadas |
| M1 | Setup + documento fuente | Qué hace a un documento "chunkeable" | `requirements.txt`, `.env.example`, `src/config.py`, `data/faq_document.txt` | Documento ≥1000 palabras con estructura clara |
| M2 | Carga y chunking | Tokens, estrategias de chunking, trade-off tamaño vs. contexto | `load_document`, `chunk_document` + tests | ≥20 chunks, todos entre 50 y 500 tokens, 100% del texto cubierto |
| M3 | Embeddings y almacenamiento | Qué es un embedding, dimensionalidad, normalización | `src/embeddings.py`, `save_index`/`load_index`, `build_index.py` | N embeddings = N chunks; índice en disco |
| M4 | Búsqueda vectorial | Similitud coseno, k-NN, filtro por rango, top-k | `search_similar_chunks` + tests con vectores de juguete | Devuelve 2–5 chunks relevantes en preguntas de prueba |
| M5 | Generación | Ensamblado de contexto, grounding, prompt anti-alucinación | `build_context`, `generate_answer`, `QueryResult` | JSON válido con las 3 claves |
| M6 | CLI y ejemplos | Orquestación de punta a punta | `main.py`, `outputs/sample_queries.json`, chequeo de palabras clave | ≥3 ejemplos; ≥80% de chunks relevantes |
| M7 | Evaluador (bonus) | LLM-as-judge, fidelidad, detección de alucinaciones | `src/evaluator.py` | `score` 0–10 y `reason` ≥50 caracteres, ≥2 dimensiones |
| M8 | README y cierre | Documentar decisiones técnicas | `README.md` | Cumple cada indicador de la rúbrica (sección 10) |
| M9 | Opcional: ANN vs. k-NN | Índices aproximados (HNSW), recall@k | Script comparativo | Tabla de velocidad y recall |
| M10 | Opcional: API con FastAPI | Endpoints, validación con Pydantic, códigos HTTP, Swagger | `src/api.py`: `POST /ask`, `GET /health` | El endpoint devuelve el mismo `QueryResult` que la CLI; 400 si la pregunta está vacía, 503 si no hay índice |
| M11 | Opcional: pgvector | Interfaz de vector store, bases vectoriales, índices en Postgres | Interfaz `VectorStore` (`add`, `search`), `NumpyVectorStore`, `PgVectorStore`, `docker-compose.yml` | Mismos resultados de búsqueda con ambos backends; `VECTOR_STORE=numpy` sigue siendo el default y no requiere Docker |

Estado: M0 ✅ · M1 ✅ · M2 siguiente.

Regla para los opcionales: toda la lógica RAG vive en `answer_question(question) -> QueryResult`. La CLI y la API son dos puertas de entrada a esa misma función, y el backend de almacenamiento se elige por configuración sin tocar los pipelines.

## 10. Trazabilidad con la rúbrica

| Requisito de la rúbrica | Milestone | Cómo se verifica |
|---|---|---|
| Parsing y chunking (≥20 chunks, 50–500 tokens, 100% cubierto, estrategia documentada) | M2 | Tests + resumen impreso por `build_index` |
| Embeddings (uno por chunk, OpenAI, guardados) | M3 | Assert `len(embeddings) == len(chunks)` |
| JSON con exactamente 3 claves | M5 | Validación Pydantic + test |
| Búsqueda vectorial con similitud explícita | M4 | Fórmula de coseno visible en `vector_store.py` |
| Calidad de recuperación (≥80% relevantes, 2–5 chunks) | M4, M6 | Chequeo de palabras clave sobre `sample_queries.json` |
| Pipeline de indexación modular (4 etapas) | M2, M3 | Una función por etapa en `build_index.py` |
| Pipeline de consulta (4 etapas conectadas) | M4, M5 | Una función por etapa en `query.py` |
| Comprensión de RAG (dos pasos, beneficios) | M5, M8 | Código + sección "Por qué RAG" del README |
| Código modular (≥4 funciones, ≤30 líneas) | Todos | Revisión en cada checkpoint |
| Documentación | M8 | Checklist del README |
| Dependencias y entorno (versiones fijadas, `os.getenv`) | M1 | `requirements.txt`, `.env.example` |
| Agente evaluador | M7 | Test del schema `Evaluation` |
