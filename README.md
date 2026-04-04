# ragsearch

`ragsearch` is a Python library designed for building a Retrieval-Augmented Generation (RAG) application that enables natural language querying over both structured and unstructured data. This tool leverages embedding models and a vector database (FAISS or ChromaDB) to provide an efficient and scalable search engine.

## Features
- Seamless integration with the Cohere AI LLM for generating embeddings.
- Utilizes FAISS for fast, in-memory vector storage and similarity search.
- Optional ChromaDB backend for persistent, scalable vector search using SQLite.
- Unstructured ingestion support through LiteParse (with Python fallback parsers).
- Incremental indexing support for FAISS setup runs with changed-file detection and skip-unchanged behavior.
- Easy setup and configuration for different use cases.
- Simple web interface for user interaction.

## Installation
To install `ragsearch`, run the following command:

```bash
pip install ragsearch
```

Alternatively, for local development, use:

```bash
pip install /path/to/ragsearch
```

Ensure that you have all necessary dependencies installed:

```bash
pip install pandas faiss-cpu flask cohere chromadb
```

For richer unstructured parsing support (HTML/PDF/DOCX fallback):

```bash
pip install beautifulsoup4 pypdf python-docx
```

For LiteParse support, ensure Node.js 18+ and npx are available:

```bash
node --version
npx --version
npx --yes @run-llama/liteparse --help
```

## Basic Setup
### Step 1: Prepare Your Data
Ensure you have your data in a supported format. Structured files (CSV/JSON/Parquet) are loaded with pandas. Unstructured files are parsed via LiteParse when available, otherwise fallback parsers are used.

**Example data (`sample_data.csv`)**:
```csv
name,id,minutes,contributor_id,submitted,tags,n_steps,steps,description,ingredients,n_ingredients,average_rating,votes,Score,calories,total fat (PDV),sugar (PDV),sodium (PDV),protein (PDV),saturated fat (PDV),carbohydrates (PDV),category,meal_type,cuisine,difficulty
baked ham glazed with pineapple and chipotle peppers,146558,85,58104,2005-11-28,"['ham', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'occasion', 'north-american', 'lunch', 'main-dish', 'pork', 'american', 'mexican', 'southwestern-united-states', 'tex-mex', 'oven', 'holiday-event', 'easter', 'stove-top', 'spicy', 'christmas', 'meat', 'taste-mood', 'sweet', 'equipment', 'presentation', 'served-hot', '4-hours-or-less']",7,"['mix cornstarch with a little cold water to dissolve', 'place all ingredients except for ham in a blender and blend smooth , in a small saucepan over medium heat bring to a boil them simmer till thickened', 'preheat oven to 375 f', 'place ham , cut end down , in a large baking pan and score skin', 'bake ham for 15 minutes', 'brush glaze over ham and bake for another hour or until internal temperature reads 140 f', 'baste half way through baking']","sweet, smokey and spicy! go ahead and leave the seeds in if you enjoy the heat.","['smoked ham', 'brown sugar', 'crushed pineapple', 'chipotle chile in adobo', 'adobo sauce', 'nutmeg', 'fresh ginger', 'cornstarch', 'salt']",9,5.0,27,4.852754009963201,712.5,50.0,127.0,207.0,131.0,55.0,12.0,Non-veg,Lunch,North-American,2.65
chocolate raspberry  or strawberry  tall cake,90774,60,117781,2004-05-05,"['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'occasion', 'desserts', 'oven', 'dinner-party', 'holiday-event', 'cakes', 'chocolate', 'dietary', 'equipment']",20,"['prepare the cake according to package directions , using three greased and floured 9 in round cake pans', 'bake at 350 degrees for 25- 30 minutes or until a toothpick inserted in center comes out clean', 'cool for 10 minutes', 'remove from pans to wire racks to cool completely', 'in a mixing bowl , beat cream cheese until fluffy', 'combine milk and pudding mix', 'add to cream cheese and mix well', 'fold in whipped topping and raspberries', 'reserve a large dollop of filling for garnish', 'place one cake layer on a serving plate', 'spread with half of the filling', 'do not cover sides of cake , just the top as this is made to look like a torte , not a frosted cake', 'repeat layers', 'top with remaining cake', ""dust with confectioner's sugar"", 'mound the reserved filling in the center and arrange raspberries in the middle', 'garnish with fresh mint on top if desired', 'store in refrigerator', 'this is because the strawberries will""bleed"" into the filling and become mushy', 'it will still taste great , it will just look a bit unattractive']","you won't believe how easy this cake is to make. when you present it to your guests, i promise you will receive ""oohs and aahs"". it is beautiful to look at and absolutely scrumptious to eat. my father is not quick to dole out compliments on food, he said this was the best cake he has ever had. when strawberries are in peak season i often substitute them for the raspberries.","['chocolate cake mix', 'eggs', 'oil', 'water', 'cream cheese', 'milk', 'vanilla instant pudding mix', 'frozen whipped topping', 'fresh raspberries', ""confectioners' sugar"", 'of fresh mint', 'raspberries']",12,4.945945945945946,37,4.841285746927722,433.1,40.0,120.0,23.0,12.0,53.0,15.0,Non-veg,Dinner,Other,3.25
rr s caramelized onions,209735,35,145489,2007-02-06,"['60-minutes-or-less', 'time-to-make', 'main-ingredient', 'preparation', 'low-protein', 'vegetables', 'dietary', 'low-sodium', 'low-calorie', 'low-carb', 'low-in-something', 'onions']",4,"['in a large skillet , melt the butter in the olive oil over medium-high heat', 'add the onions , the salt and pepper', 'cook , stirring constantly , until the onions begin to soften , about 5 minutes', 'stir in the sugar and cook , scraping the browned bits off the bottom of the pan frequently , until the onions are golden brown , about 20 minutes']","these are a great condiment - scatter over a pizza, chop & stir into mashed potatoes, toss with pasta & parmesan cheese!  the possibilities are endless.","['butter', 'extra virgin olive oil', 'onions', 'salt', 'pepper', 'sugar']",6,4.966666666666667,30,4.838439598940391,129.0,12.0,28.0,4.0,3.0,16.0,4.0,Veg,Other,Other,2.05
mexican coffee  caf mexicano,171163,5,242766,2006-06-02,"['15-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'cuisine', 'preparation', 'north-american', 'for-1-or-2', 'low-protein', 'healthy', 'beverages', 'mexican', 'easy', 'low-fat', 'chocolate', 'dietary', 'low-sodium', 'low-cholesterol', 'low-saturated-fat', 'low-in-something', 'number-of-servings', '3-steps-or-less']",4,"['place kahla , brandy , chocolate syrup and cinnamon in a coffee cup or mug', 'fill with hot coffee', 'stir to blend', 'top with sweetened whipped cream']","posted for the zaar world tour 2006-mexico.
this drink is so yummy and definitely warms you up on a cold day.","['kahlua', 'brandy', 'chocolate syrup', 'ground cinnamon', 'hot coffee', 'sweetened whipped cream']",6,4.944444444444445,36,4.837758763526116,156.4,0.0,61.0,1.0,0.0,1.0,5.0,Veg,Beverage,North-American,1.75
magic white sauce  and variations,92008,20,121684,2004-05-27,"['30-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'sauces', 'condiments-etc', 'eggs-dairy', 'stove-top', 'dietary', 'savory-sauces', 'equipment']",16,"['pour milk into a saucepan', 'add all other ingredients', 'place pan over a medium heat , and , using a wire balloon whisk , whisk sauce constantly until butter melts', 'be sure to work the whisk into the edges of the pan to incorporate all the flour', 'whisk frequently until the mixture comes to a boil', 'reduce heat to low and simmer for about 5 minutes , stirring occasionally , until the sauce reaches the desired consistency', 'taste and add extra seasoning if required', 'variations: mustard sauce make sauce as per recipe , but use 1 cups milk and cup chicken stock', 'along with the salt , add 2 teaspoons mustard powder , teaspoon onion powder and substitute a good pinch of cayenne pepper for the black pepper', 'stir in 1 teaspoon lemon juice when sauce is completed', 'cheese sauce make sauce as per recipe , but use 1 cup milk and 1 cup cream', 'along with the salt , add a good pinch of nutmeg and substitute a good pinch of cayenne pepper for black pepper', 'at the simmering stage stir in 75g- 100g grated tasty cheese', 'stir in 1 teaspoon lemon juice when sauce is completed', 'parsley sauce make sauce as per recipe , but use 1 cups milk and cup cream', 'when sauce is finished , mix in 4 tablespoons finely chopped parsley and 1 teaspoon lemon juice']","sick of lumpy sauce? hate making that flour and butter roux? here's the answer! this is the easiest version of white sauce ever. you won’t believe it works until you try it! you will need a wire balloon whisk for this recipe and you must make sure that all ingredients are cold (or at least at room temperature) to begin with. thanks to english food writer, delia smith, for discovering this all-in-one method. the following are my simplified adaptations for basic white sauce, along with variations for mustard sauce, cheese sauce (mornay sauce) and parsley sauce.","['milk', 'butter', 'plain flour', 'salt', 'black pepper']",5,5.0,23,4.834348261208601,627.0,76.0,0.0,44.0,23.0,155.0,11.0,Veg,Other,Other,2.65
```


### Step 2: Initialize `ragsearch`
Use the `setup()` function to set up the `ragsearch` with your data and configuration.

**Example code (FAISS, default)**:
```python
from pathlib import Path
from ragsearch import setup

# Define your data path and configuration parameters
data_path = Path("path/to/your/sample_data.csv")
llm_api_key = "your-cohere-api-key"

# Initialize the RagSearchEngine (FAISS backend)
rag_engine = setup(data_path, llm_api_key)
```

**Example code (ChromaDB backend)**:
```python
from pathlib import Path
from ragsearch import setup

data_path = Path("path/to/your/sample_data.csv")
llm_api_key = "your-cohere-api-key"
chromadb_sqlite_path = "/path/to/chroma.sqlite3"
chromadb_collection_name = "your_collection_name"

# Initialize the RagSearchEngine (ChromaDB backend)
rag_engine = setup(
    data_path,
    llm_api_key,
    use_chromadb=True,
    chromadb_sqlite_path=chromadb_sqlite_path,
    chromadb_collection_name=chromadb_collection_name
)
```

### Step 2b: Unstructured File Ingestion (PDF/DOCX/TXT/HTML)

Use the same `setup()` API for unstructured files:

```python
from pathlib import Path
from ragsearch import setup

data_path = Path("path/to/your/report.pdf")
llm_api_key = "your-cohere-api-key"

rag_engine = setup(data_path, llm_api_key)
```

Parser selection behavior:
- LiteParse is preferred when available (Node.js + npx installed).
- If LiteParse is unavailable, fallback parser is used for supported file types.
- If LiteParse is selected but fails at runtime, `setup()` retries with fallback parser when the file type is fallback-supported.
- Unsupported types raise `UnsupportedFileTypeError`.

Ingestion diagnostics:
- After `setup()`, the engine exposes `rag_engine.ingestion_diagnostics` with deterministic per-file fields:
    - `source_path`: input file path used for setup.
    - `selected_parser`: `structured/pandas`, `liteparse`, or `fallback`.
    - `status`: `success` or `recovered_with_fallback`.
    - `failure_reason`: empty string on success; primary parser error message when fallback recovery is used.
    - `observability`: structured setup metrics for deterministic telemetry checks:
        - `stage`: `ingestion`
        - `event`: `setup_completed`
        - `metrics.setup_latency_ms`: setup latency for current run
        - `metrics.loaded_records`: records loaded into index for current run
        - `metrics.selected_parser`: parser chosen by setup
        - `metrics.fallback_recovered`: boolean fallback recovery flag

Note: supported extensions can be backend-dependent. LiteParse supports additional types such as `.doc`, `.png`, `.jpg`, and `.jpeg`, while fallback parsing is intentionally narrower.

Incremental indexing behavior (FAISS backend):
- `setup()` persists an embedding manifest in the embeddings directory and reuses cached embeddings for unchanged records.
- New or changed records are re-embedded, while unchanged records are skipped for embedding generation.
- After setup, `rag_engine.ingestion_diagnostics["indexing"]` reports deterministic counters:
    - `manifest_version`: manifest schema version.
    - `manifest_path`: on-disk manifest file path.
    - `total_records`: total records considered for indexing.
    - `embedded_records`: records embedded in the current run.
    - `reused_records`: records reused from manifest cache.
    - `new_records`: records seen for the first time.
    - `changed_records`: previously-seen records with changed content hash.

Optional setup parameter:
- `embeddings_dir`: custom directory for embedding artifacts and incremental manifest cache.

Retrieval quality hooks (optional):
- `chunking_strategy`: controls how each record is split before embedding/indexing.
- `reranker`: post-processes retrieval results before they are returned.
- Defaults are backward-compatible:
    - row-level chunking (`RowChunkingStrategy`)
    - no-op reranking (`NoOpReranker`)

Example hook usage:

```python
from ragsearch import setup
from ragsearch.chunking import FixedWordChunkingStrategy

class ReverseReranker:
    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        return list(reversed(results))

rag_engine = setup(
    data_path,
    llm_api_key,
    chunking_strategy=FixedWordChunkingStrategy(words_per_chunk=120),
    reranker=ReverseReranker(),
)
```

Notes:
- Keep hooks disabled unless you are actively tuning retrieval quality.
- Chunking strategy changes can alter indexed record boundaries and retrieval behavior.

### Step 3: Run a Search Query
Once the `ragsearch` is initialized, you can perform natural language searches.

**Example code**:
```python
query = "Find recipes with chicken"
results = rag_engine.search(query, top_k=5)

for result in results:
    print("Metadata:", result["metadata"])
    print("Citation:", result["citation"])
    print("Similarity:", result["similarity"])

# citation fields:
# - record_id: row/chunk index in the indexed dataset
# - source_path: source file path when available
# - parser_name: parser used during ingestion when available
# - excerpt: up to 200 chars from text/combined_text
```

### Generate a Grounded Answer
Use the retrieval-to-generation pipeline when you want a direct answer with preserved citations:

```python
response = rag_engine.answer("What does the document say about the accident location?", top_k=3)

print(response["answer"])
for citation in response["citations"]:
    print(citation)
```

Answer response fields:
- `question`: original query string
- `answer`: generated response text
- `results`: full retrieval results, including `metadata`, `citation`, and `similarity`
- `citations`: citation list preserved from retrieval
- `context`: grounded retrieval context supplied to the LLM

HTTP API note:
- `POST /answer` returns the same structured payload as `rag_engine.answer(...)`
- `POST /query` remains backward-compatible and continues to return search results only

Observability events:
- `rag_engine.observability_events` stores structured events emitted during retrieval and generation.
- Indexing event emission is backend-dependent: FAISS/in-process indexing emits `indexing_completed`; ChromaDB mode does not emit an indexing event because indexing is handled outside the in-process FAISS embedding path.
- Retrieval events include deterministic payload fields: `query`, `top_k`, `results_count`, `latency_ms`.
- Generation events include deterministic payload fields: `query`, `top_k`, `results_count`, `citations_count`, `latency_ms`.
- Configure `observability_max_events` in `setup(...)` to cap retained in-memory events for long-lived processes.

Evaluation harness baseline:
- Use `ragsearch.evaluation.run_regression_gates(...)` to run deterministic pass/fail gates over a fixed case set.
- Default thresholds are configurable via `EvaluationThresholds(min_results=..., min_citations=...)`.
- Baseline unit gate command:

```bash
poetry run pytest libs/tests/test_evaluation.py -q
```

Evaluation quickstart:

```python
from pathlib import Path
from ragsearch.evaluation import EvaluationThresholds, load_cases, run_regression_gates

cases = load_cases(Path("libs/tests/eval_cases.json"))
summary = run_regression_gates(
        rag_engine,
        cases,
        EvaluationThresholds(min_results=1, min_citations=1),
)

print(summary["pass"], summary["passed_cases"], summary["failed_cases"])
```

Evaluation CLI:

```bash
python -m ragsearch.evaluation \
    --engine-factory your_project.bootstrap.build_engine \
    --cases libs/tests/eval_cases.json \
    --summary-only
```

## Running the Web Interface
### Step 1: Start the Flask Server
Run the following command to start the Flask server:
```bash
rag_engine.run()
```

### Step 2: Access the Web Interface
Open your browser and navigate to:

```
http://localhost:8080/
```

### Step 3: Interact with the Web Interface
- Enter a search query in the input field.
- Click the **Submit** button.
- View the results displayed on the page.

API contract note for `/query`:
- Default behavior is backward compatible and returns metadata-only results.
- Set `include_details=true` in request JSON to return full objects with `metadata`, `citation`, and `similarity`.

## Testing the Package
### Running Unit Tests
Ensure your package functions as expected by running `pytest`:

```bash
poetry run pytest
```

## Advanced Usage and Customization

### Using ChromaDB Backend
To use ChromaDB, set `use_chromadb=True` and provide the path to your ChromaDB SQLite file and collection name. This enables persistent, scalable vector search.

### Changing the Embedding Model
`setup()` now uses an embedding-model contract internally.

Provider selection (config-driven via `setup()` params):
- `embedding_provider="cohere"` (default)
- `embedding_provider="sentence_transformers"`
- `embedding_provider="openai"`
- `embedding_provider="ollama"`

Optional provider settings:
- `embedding_model_name`: provider-specific model id
- `embedding_api_key`: embedding provider key (defaults to `llm_api_key`)
- `embedding_base_url`: custom endpoint URL (OpenAI-compatible/Ollama host)

Example:
```python
rag_engine = setup(
    data_path,
    llm_api_key,
    embedding_provider="openai",
    embedding_model_name="text-embedding-3-small",
    embedding_api_key="your-openai-key",
)
```

Expected embed response contract:
- The embedding provider must support `embed(texts=[...])`.
- The response must contain an `embeddings` attribute.
- `embeddings` must be a non-empty sequence of numeric vectors.

Dimension behavior:
- `setup()` probes the embedding model to infer vector dimension automatically.
- If probe-time inference fails (invalid response shape or transient provider error), `setup()` falls back to legacy dimension `4096` for backward compatibility.

Migration note for custom providers:
- If you use a custom embedding client, ensure response shape follows the contract above to avoid `ValueError` during indexing/search normalization.

Optional dependencies for non-default providers:
- `sentence-transformers` for `sentence_transformers`
- `openai` for `openai`
- `ollama` for `ollama`

### Changing the LLM Provider
`setup()` now uses a configurable LLM contract internally.

Provider selection:
- `llm_provider="cohere"` (default)
- `llm_provider="openai"`
- `llm_provider="ollama"`

Optional provider settings:
- `llm_model_name`: provider-specific chat model id
- `llm_base_url`: custom endpoint URL (OpenAI-compatible/Ollama host)

Example:
```python
rag_engine = setup(
    data_path,
    llm_api_key,
    llm_provider="openai",
    llm_model_name="gpt-4o-mini",
    llm_base_url="https://api.openai.com/v1",
)
```

Optional dependencies for non-default providers:
- `openai` for `openai`
- `ollama` for `ollama`

Migration note:
- Custom generation clients should return string output through the `generate(prompt, **kwargs)` contract to avoid runtime adaptation errors.

### Adding More Metadata
Include additional columns in your data for more detailed results.

### Customizing the Web Interface
Edit `index.html` in the `templates` directory to adjust the UI layout or add more user features.

## Troubleshooting
- **`AssertionError: d == self.d`**: Embedding/vector dimensions are typically inferred automatically. If this appears with custom providers, verify your embed response contains consistent numeric vectors in `response.embeddings`.
- **`TypeError: embed() takes 1 positional argument`**: Use the correct keyword argument format for `embed()` based on your `cohere` version.
- **`ValueError: Embedding response must contain an 'embeddings' attribute`**: Your embedding provider response shape does not match the A1 contract; return an object with an `embeddings` sequence.

### Parser Pipeline Troubleshooting (Issue #18 Slice 3)

| Error | Typical Cause | Resolution |
| --- | --- | --- |
| `ParserUnavailableError: LiteParse CLI not found` | Node.js/npx not installed, or custom CLI path invalid | Install Node.js 18+, verify `npx` works, or set `RAGSEARCH_LITEPARSE_CLI` to a valid executable |
| `ParseTimeoutError` | Large/complex document exceeded parse timeout | In default `setup()` flow, timeout may be recovered automatically via fallback parser for supported types; otherwise retry with smaller file and inspect parser logs |
| `ParseCorruptError` | Corrupt file or invalid parser output payload | In default `setup()` flow, corruption may be recovered automatically via fallback parser for supported types; if both parsers fail, the primary LiteParse error is surfaced |
| `UnsupportedFileTypeError` | Extension not supported by the active parser backend | Convert to a supported format; fallback supports `.txt/.md/.html/.htm/.pdf/.docx`, LiteParse supports additional formats |
| `NoDataFoundError` | File parsed but content was empty/whitespace only | Verify source file contains readable text content |

### Performance Notes

- Structured files (CSV/JSON/Parquet) are typically fastest because they bypass parser dispatch.
- Unstructured parsing performance depends on document size and parser backend.
- Very large PDFs/DOCX files can trigger timeout paths; use smaller batches/files where possible.
- Empty/whitespace-only parsed documents are filtered before indexing to preserve retrieval quality.

## Deployment Tips
- **Deploying to a Server**: Use services like Heroku, AWS, or Docker.

## Contributing
Feel free to contribute to this project by submitting issues, feature requests, or pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

