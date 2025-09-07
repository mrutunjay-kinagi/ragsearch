# ragsearch Samples

This folder contains example scripts to help new users get started with the ragsearch package.

## How to Use

1. **Install dependencies**
   - Make sure you have installed all dependencies using Poetry:
     ```bash
     poetry install
     ```

2. **Create a ChromaDB SQLite file (for ChromaDB samples)**
   - Run the provided script to generate a test database:
     ```bash
     poetry run python samples/create_chromadb.py
     ```

3. **Run the sample scripts**
   - To test ChromaDB queries:
     ```bash
     poetry run python samples/demo_chromadb_real.py
     ```
   - To run the hello world sample:
     ```bash
     poetry run python samples/demo_hello_world.py
     ```

## Sample Scripts

- `demo_chromadb_real.py`: Demonstrates querying a ChromaDB SQLite file with natural language queries.
- `demo_hello_world.py`: Basic hello world test for the package.

## Notes
- Make sure the ChromaDB file (`chroma.sqlite3`) exists before running ChromaDB samples.
- You can modify the sample scripts to test your own queries or data.

For more information, see the main project README or documentation.
