import os
import pytest
from pathlib import Path
from libs.ragsearch.setup import setup

chromadb_sqlite_path = os.environ.get("CHROMADB_SQLITE_PATH", "chroma.sqlite3")
chromadb_collection_name = os.environ.get("CHROMADB_COLLECTION_NAME", "test_collection")
data_path = Path(os.environ.get("CHROMADB_SAMPLE_DATA_PATH", "libs/tests/sample_data.csv"))
llm_api_key = os.environ.get("LLM_API_KEY", "test-api-key")

@pytest.mark.skipif(
	not os.path.exists(chromadb_sqlite_path),
	reason=f"ChromaDB SQLite file not found at '{chromadb_sqlite_path}'. Set CHROMADB_SQLITE_PATH or run create_chromadb.py first."
)
def test_chromadb_setup_and_search():
	engine = setup(
		data_path,
		llm_api_key,
		use_chromadb=True,
		chromadb_sqlite_path=chromadb_sqlite_path,
		chromadb_collection_name=chromadb_collection_name
	)

	query = "Find recipes with chicken"
	results = engine.chromadb_search(query, top_k=3)
	assert isinstance(results, dict)
	assert "ids" in results
	assert len(results["ids"]) <= 3
	print(results)
