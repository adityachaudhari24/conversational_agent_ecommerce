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
 Techno-Functional Specification for this pipeline is under "specs/data-ingestion-pipeline" folder