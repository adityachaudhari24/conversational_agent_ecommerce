# conversational_agent_ecommerce - a production grade demo application to demonstrate 3 key pipeline in conversational agent ** white paper
Conversational Agent E-Commerce demo application 


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


 ## Step 2 - Creating Data Ingestion pipeline
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
 ```