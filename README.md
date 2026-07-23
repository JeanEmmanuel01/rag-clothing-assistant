# rag-clothing-assistant

A retrieval-augmented chatbot for Fashion Forward Hub, a fictional online clothing
store. It answers policy questions from a FAQ set and product questions from a
Weaviate-indexed catalog, combining metadata filtering with semantic search before
generating a response through Together AI.

## Architecture

Each user query is routed through a sequence of classification and retrieval steps
before generation:

```
query
  │
  ▼
FAQ / Product router ──── FAQ ───────────────────────────► FAQ context (prompt-injected) ─┐
  │                                                                                        │
  ▼                                                                                        │
Product                                                                                    │
  │                                                                                        │
  ▼                                                                                        │
technical / creative classifier ──► sampling profile (temperature / top_p)                 │
  │                                                                                        │
  ▼                                                                                        │
metadata extraction (LLM → JSON filter: gender, masterCategory, articleType,               │
                     baseColour, season, usage, price)                                     │
  │                                                                                        │
  ▼                                                                                        │
Weaviate filtered vector search (near_text + filters, with progressive                     │
                                  filter relaxation if too few results)                     │
  │                                                                                        │
  ▼                                                                                        │
retrieved items → context block                                                            │
  │                                                                                        │
  ▼                                                                                        ▼
                        generation prompt (context + query) ──► Together AI completion
```

FAQ queries skip retrieval entirely: the FAQ set is small enough to be serialized
and injected directly into the prompt. Product queries go through metadata
extraction and filtered vector search, with filters dropped in a fixed order
(season, usage, baseColour, masterCategory, gender) when the result set is too
small, falling back to unfiltered semantic search if no combination works.

## Stack

- **Weaviate** — vector database holding the product catalog, queried with
  metadata filters combined with `near_text` semantic search.
- **Together AI** — chat completion API used for query routing, metadata
  extraction, and response generation.
- **sentence-transformers** (`BAAI/bge-base-en-v1.5`) — embedding model used for
  vectorizing queries and product text.
- **ipywidgets** — chat interface rendered inline in the notebook.
- **Flask** — local `/vectors` and `/rerank` endpoints (`flask_app.py`) that
  Weaviate's `text2vec-transformers` / reranker modules call into.

## Project structure

```
.
├── rag_clothing_assistant.ipynb   # Main notebook: routing, retrieval, generation, chat UI
├── utils.py                       # Shared helpers: LLM calls, ChatBot/ChatWidget, embeddings
├── flask_app.py                   # Local Flask server exposing /vectors and /rerank to Weaviate
├── weaviate_server.py             # Starts an embedded Weaviate instance
├── dataset/
│   ├── clothes.csv                # Raw product catalog (tabular)
│   ├── clothes_json.joblib        # Product catalog, pre-processed as a list of dicts
│   ├── faq.joblib                 # FAQ entries (question / answer / type)
│   └── clothing_ft_format.csv     # Product descriptions in a fine-tuning text format
├── .gitignore
└── README.md
```

## Setup

### Environment variables

| Variable             | Purpose                                                              | Default                        |
|-----------------------|-----------------------------------------------------------------------|---------------------------------|
| `TOGETHER_API_KEY`   | Together AI API key. When unset, calls fall back to an HTTP POST against `TOGETHER_BASE_URL`. | — |
| `TOGETHER_BASE_URL`  | Base URL for the Together-compatible chat completions endpoint.       | `https://api.together.xyz/`    |
| `PRODUCT_IMAGES_DIR` | Directory containing product images named `<product_id>.jpg`, used by the chat widget to display images inline. | `dataset/images` |
| `WEAVIATE_DATA_PATH` | Local directory where the embedded Weaviate instance persists its data. | `weaviate_data` |

The repository ships no product images. The chat widget only displays an image for
a given product ID if a matching `<product_id>.jpg` file is present under
`PRODUCT_IMAGES_DIR`; otherwise it silently skips that image. To enable image
previews, add the desired files to that directory (or point `PRODUCT_IMAGES_DIR`
at wherever they're stored).

### Running the stack

1. Start the embedded Weaviate instance and the local Flask server (`weaviate_server.py`
   and `flask_app.py` are imported directly from the notebook, which also serves
   the two ports Weaviate calls into for vectorization and reranking).
2. Populate the `products` Weaviate collection with the catalog data before running
   the notebook — `weaviate_server.py` only starts the instance, it does not ingest
   data.

   **Gap:** no standalone ingestion script exists in this repo yet. The notebook
   assumes a `products` collection already populated with `dataset/clothes_json.joblib`
   is reachable; getting it there is currently a manual/undocumented step.
3. Open `rag_clothing_assistant.ipynb` and run the cells in order.

## Limitations and next steps

- The FAQ set is injected into the prompt rather than indexed; this stops scaling
  once the FAQ grows beyond a few dozen entries.
- Classification relies on two chained LLM calls, which adds latency to every
  turn. A single call returning both labels, or a small local classifier, would
  cut it.
- Filter relaxation drops fields in a fixed order rather than by measured
  selectivity on the actual catalog.
- No evaluation set: routing accuracy and retrieval quality are only checked by
  hand.
