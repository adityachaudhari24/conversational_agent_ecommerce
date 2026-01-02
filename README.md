# conversational_agent_ecommerce - A Demo application to demonstrate 3 key pipeline in conversational agent 


### Environment preparation
```bash
crearing uv envirment  commands 

uv venv .venv --python cpython-3.11.13-macos-aarch64-none
source .venv/bin/activate
```

# Key Steps for the project

## Step 1 - Data Preparation and Feature Engineering

 Data is loaded from huge huggingface dataset here https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories
 For experiment only iphone data is extracted, all data loading from huggingface and data extraction steps are in the notebook "ecommerce_data.ipynb" in the codebase.


 ## Step 2 -  Data Ingestion pipeline
* Techno-Functional Specification for this pipeline is under ".kiro/specs/data-ingestion-pipeline" folder.
* using "RecursiveCharacterTextSplitter" for chunking - we have chosen this to keep sementic coherance and context preservation specially because recursive splitting split by separaters.
* For better visualizing what actually goes in into vector database check "scripts/ingestion" visualization python scripts this gives good idea on what goes in to the vector DB.

* use below commands to run entire data ingestion pipeline -
```bash
# below is for dry run
python -m src.pipelines.ingestion --dry-run

# below will actually create the embeddings in the pinecone
python -m src.pipelines.ingestion


# With custom data file and verbose logging
python -m src.pipelines.ingestion --data-file data/phones_reviews.csv --log-level DEBUG

# With custom configuration file
python -m src.pipelines.ingestion --config config/ingestion.yaml

# With custom parameters
python -m src.pipelines.ingestion --batch-size 50 --abort-threshold 0.3
```

 #### running tests for Data Ingestion Pipeline commands
 ```bash
 1. python -m pytest tests/unit/test_document_loader.py::TestDocumentLoaderUni
tTests -v
 2. python -m pytest tests/unit/test_document_loader.py -v
 #run all tests 
 3. uv run pytest tests/unit/ -v

# running single test example
uv run pytest tests/unit/test_vector_searcher.py::TestVectorSearcherIntegration -v
 4. 
 ```

## Step 3 -  Data Retrieval pipeline
* For retrieval pipeline design please check here 'https://github.com/adityachaudhari24/conversational_agent_ecommerce/blob/main/.kiro/specs/data-retrieval-pipeline/design.md'
* If Metadata filters are provided we are doing similarity search first and then on that MMR, however if no filters are provided we are doing MMR directly skipping similarity search.
* MMR algorithm (balance relevance + diversity).
* To test the retrieval pipeline use belwo commands (For details please check scripts/retrieval/README.md for detailed testing of this pipeline)
```bash
# Simple test to verify everything works
python scripts/retrieval/simple_test.py

# Run detailed demo
python scripts/retrieval/demo_retrieval_pipeline.py

# Interactive query testing
python scripts/retrieval/demo_retrieval_pipeline.py --mode interactive

```
## Step 4 -  Data Inference pipeline
* For detailed design for inference pipeline please check /kiro/specs/data-inference-pipeline/design.md.
* we are using LangGraph for this pipeline, please check the design for details.
* For testing the inference pipeline please check scripts/inference/README.md.
* start interacting with "python scripts/inference/interactive.py"


## Step 5 -  Backend and Frontend

```bash

# Terminal 1 - Backend
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Frontend  
uv run streamlit run src/frontend/app.py --server.port 8501

```