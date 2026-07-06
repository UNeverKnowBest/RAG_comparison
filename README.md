ID-RAG for Multi-hop Question Answering with Small Language Models :

Setup
1. Clone and install dependencies
git clone https://github.com/UNeverKnowBest/RAG_comparison.git
```bash
cd RAG_comparison
uv sync
```
3. Start Neo4j
```bash
docker compose up -d
```
5. Create your .env file (copy from the template)
```bash
cp .env.example .env
```
The .env file must contain (values below match docker-compose.yml and a default local Ollama install):

Repository structure:

We use the dataset 2WikiMultihopQA (https://huggingface.co/datasets/framolfese/2WikiMultihopQA) to create our Knowledge Graph. Specially, we use the total dev set to create KG and sample the first 300 questions from the 
dataset to evlatuate our RAG system. 

The building process of KG and other preprocessing files are listed in the src/data_processing
.
The main pipeline of different RAG systems are implemented in src/.

Reproduce procedure:

If you want to reproduce this result, please firstly download the dataset using download.py and then run the build_corpus, build_kg and extract_kg to build corpus for Semantic RAG and ID-RAG. Then, run the command uv run python run.py to run the experiments.
```bash
# 1. Create data directories
mkdir -p dataset/raw dataset/processed dataset/kg dataset/corpus

# 2. Download 2WikiMultihopQA from Hugging Face
uv run python src/data_processing/download.py

# 3. Filter validation samples that contain evidence triples
uv run python src/data_processing/preprocess_data.py

# 4. Build the passage/sentence corpus
uv run python src/data_processing/build_corpus.py

# 5. Extract entity/relation tables from evidence triples
uv run python src/data_processing/extract_kg.py

# 6. Import the knowledge graph into Neo4j 
uv run python src/data_processing/build_kg.py

# 7. Build the Chroma vector index
uv run python src/build_index.py

# 8. Run the main experiment (first 300 questions by default)
uv run python run.py                 # use --sample 0 to run on the full set

# 9. Run the ablation study
uv run python ablation.py
```
Figures:

All figures are created manually by Lucidchart.
