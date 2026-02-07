# conversational_agent_ecommerce - A Demo application to demonstrate 3 key pipeline in conversational agent 

## This demo project is to companion to this published article : https://www.linkedin.com/newsletters/7414417371697684480/


### Environment preparation
```bash
crearing uv envirment  commands 

uv venv .venv --python cpython-3.11.13-macos-aarch64-none
source .venv/bin/activate
```

# Key Steps for the project

## Step 1 - Data Preparation and Feature Engineering

 Data is loaded from huge huggingface dataset here https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories
 For experiment only iphone data is filtered, all data loading from huggingface and then keeping that data into csv is code is under the notebook "ecommerce_data.ipynb" in the codebase.


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
```

 #### running tests for Data Ingestion Pipeline commands
 ```bash
 1. python -m pytest tests/unit/test_document_loader.py::TestDocumentLoaderUni
tTests -v
 2. python -m pytest tests/unit/test_document_loader.py -v
 #run all tests 
 3. uv run pytest tests/unit/ -v

# running single test example
4. uv run pytest tests/unit/test_vector_searcher.py::TestVectorSearcherIntegration -v
 
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
* I am using LangGraph for this pipeline, please check the design for details.
* For testing the inference pipeline please check scripts/inference/README.md.
* start interacting with "python scripts/inference/interactive.py"


## Step 5 -  Backend and Frontend

```bash

# Terminal 1 - Backend
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Frontend  
uv run streamlit run src/frontend/app.py --server.port 8501

```



## Architectural trade-off decisions
<details>
<summary>🎯Q. Which model is chosen as Embedding model and why? </summary>

- We are using `text-embedding-3-large` model for embedding with 3072 dimensions. (Both ingestion and retrieval pipeline use this model)
- more dimensions represent a higher "resolution" for capturing the meaning of your text
- Excellent for product reviews and descriptions (As our data is phone reviews)
- CONS - `text-embedding-3-large` does not support image embeddings. Our RAG is not optimized to embed images in reviews its only text based.
</details>

<details>
<summary>🎯Q. What is your chunking strategy ? </summary>

- I am using using `RecursiveCharacterTextSplitter` chunking strategy.
- The recursive approach tries to split on natural boundaries (paragraphs, sentences, words) before falling back to character-level splitting.
- I have also have 200-character overlap ensures important context isn't lost between chunks.
</details>

<details>
<summary>🎯Q. Why you have chosen MMR search why not similarity search ? </summary>

- MMR prevents redundant results. Pure similarity search might return 5 documents all describing identical results because they're all highly similar to the query. MMR selects documents that are both relevant to the query AND different from each other, giving users a diverse set of results. This is especially valuable in our reviews use case where we want to  to find variety in reviews, not repetitions.
</details>

<details>
<summary>🎯Q. Why re-ranking is not used? </summary>

- The pipeline uses MMR (Maximal Marginal Relevance) and LLM-based contextual compression instead of reranking.
- Design Choice: The system prioritizes diversity (MMR) and relevance filtering (LLM compression) over score-based reranking
- we choose the LLM based reranker because
  - Quality over Speed: LLM provides better semantic understanding
  - Noise Reduction: Completely removes irrelevant documents rather than just reordering
  - Better customization control: Can adjust the relevance prompt for different use cases
  - E-commerce Focus: Better for product queries where relevance is more important than     
    ranking nuances.

- _execute_retrieval_workflow()_ function in src/pipelines/retrieval.py uses MMR and then LLM-based contextual compression to filter and rank the retrieved documents.
- CONS : this reranking approach(LLM based) is not cost optimized also has a bit more latency.
</details>

<details>
<summary>🎯Q. Are you doing validations before retrieval? </summary>

- yes we are doing validations before retrieval, meaning before passing request to vector DB.
 - 1. _validate(query)     - Validates query is not empty/whitespace
 - 2. _normalize(query)    - Normalizes whitespace and unicode
 - 3. _truncate(query)     - Truncates if too long
 - 4. _embed(query)        - Generates embedding

</details>