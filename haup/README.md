# HAUP v3.0 — Hybrid Adaptive Unified Pipeline

A high-performance bidirectional data pipeline system for seamless data flow between PostgreSQL/Neon databases and ChromaDB vector stores, with an integrated RAG (Retrieval-Augmented Generation) engine.

## Features

- **Forward Pipeline**: PostgreSQL → ChromaDB with automatic embedding generation
- **Reverse Pipeline**: ChromaDB → PostgreSQL with constraint-aware data extraction
- **RAG Engine**: Interactive question-answering system with context retrieval
- **Hardware-Adaptive**: Auto-detects CPU, RAM, GPU/VRAM and optimizes worker configuration
- **Checkpoint System**: Crash-safe resume capability with SQLite WAL mode
- **Real-Time Monitoring**: Live progress bars, ETA, resource usage, and per-worker statistics
- **Multi-Format Support**: PostgreSQL, Excel output, and extensible architecture

## Architecture

### Forward Pipeline (main.py)
```
PostgreSQL/Neon → Stream Reader → Worker Pool → Embedding Generation → ChromaDB
                                      ↓
                              Schema Analyzer (column classification)
                                      ↓
                              Template Builder (text serialization)
```

### Reverse Pipeline (reverse_main.py)
```
ChromaDB → Vector Reader → Worker Pool → Text Parser → Constraint Reconciler → PostgreSQL/Excel
                                ↓
                        Heuristic Parser (data recovery)
                                ↓
                        Schema Loader (type inference)
```

### RAG Engine (rag_main.py)
```
User Query → Query Rewriter → Retriever → Reranker → Context Builder → LLM → Response
                                  ↓
                            ChromaDB Search
                                  ↓
                        Conversation Manager (session tracking)
```

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL or Neon database
- CUDA-capable GPU (optional, for faster embeddings)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd haup
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the `haup/` directory:
```env
# Database Configuration
DB_TYPE=postgresql

# Neon Connection String
NEON_CONNECTION_STRING=postgresql://user:password@host/database?sslmode=require

# PostgreSQL/Neon Configuration
PG_TABLE=users

# RAG Configuration (optional)
OPENAI_API_KEY=your_openai_api_key
CHROMA_PERSIST_DIR=./chroma_db
```

## Usage

### Forward Pipeline: PostgreSQL → ChromaDB

Embed data from PostgreSQL into ChromaDB vector store:

```bash
python main.py
```

The forward pipeline will:
- Connect to your Neon/PostgreSQL database
- Analyze table schema and classify columns (semantic, numeric, date, ID)
- Generate text templates for embedding
- Create embeddings using sentence-transformers
- Store vectors in ChromaDB with metadata
- Track progress with checkpoints for resume capability

### Reverse Pipeline: ChromaDB → PostgreSQL

Extract and reconstruct data from ChromaDB back to PostgreSQL:

```bash
python reverse_main.py
```

The reverse pipeline will:
- Read vectors from ChromaDB
- Parse embedded text to recover original data
- Infer SQL types and apply constraints
- Write to PostgreSQL or Excel
- Preserve data integrity with constraint reconciliation

### RAG Engine: Interactive Q&A

Query your embedded data using natural language:

```bash
# Interactive mode
python rag_main.py

# Single query mode
python rag_main.py --query "Find users with email domain gmail.com"
```

RAG commands:
- `/help` - Show available commands
- `/new` - Start new conversation session
- `/history` - View conversation history
- `/health` - Check system health
- `/clear` - Clear terminal
- `/quit` or `/exit` - Exit

## Configuration

### Hardware Detection

HAUP automatically detects and optimizes for your hardware:
- CPU cores (physical and logical)
- RAM capacity
- GPU availability (CUDA)
- VRAM capacity

Worker count and batch sizes are calculated based on available resources.

### Checkpoint System

Both pipelines use SQLite-based checkpoints with WAL mode for crash safety:

- **Forward**: `haup_checkpoint.db` tracks chunk processing
- **Reverse**: `reverse_job.db` tracks vector extraction

Resume from interruption by simply re-running the pipeline.

### Schema Analysis

The forward pipeline classifies columns into:
- **Semantic**: Text columns for embedding (name, email, address)
- **Numeric**: Numbers for metadata (age, price, quantity)
- **Date**: Temporal data (created_at, updated_at)
- **ID**: Primary keys and identifiers
- **Skip**: Binary, JSON, or excluded columns

All column types are included in the embedding template to ensure complete data recovery.

### Constraint Reconciliation

The reverse pipeline handles PostgreSQL constraints:
- NOT NULL constraints
- UNIQUE constraints
- PRIMARY KEY constraints
- Data type validation

Strategy: Relaxed mode allows NULL values during extraction to prevent insertion failures.

## Project Structure

```
haup/
├── main.py                      # Forward pipeline entry point
├── reverse_main.py              # Reverse pipeline entry point
├── rag_main.py                  # RAG engine CLI
├── rag_api.py                   # RAG REST API (optional)
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
│
├── forward_core/                # Forward pipeline components
│   ├── orchestrator.py          # Pipeline orchestration
│   ├── stream_reader.py         # PostgreSQL data streaming
│   ├── schema_analyzer.py       # Column classification
│   ├── worker_pool_manager.py   # Embedding workers
│   ├── vector_writer.py         # ChromaDB writer
│   ├── monitor.py               # Progress monitoring
│   └── hardware_detector.py     # System resource detection
│
├── reverse_core/                # Reverse pipeline components
│   ├── vect_batch_reader.py     # ChromaDB reader
│   ├── reverse_worker_pool.py   # Parsing workers
│   ├── reverse_writer.py        # PostgreSQL/Excel writer
│   ├── schema_loader.py         # Type inference
│   ├── schema_reconciler.py     # Constraint handling
│   ├── constraint_reader.py     # PostgreSQL constraint reader
│   ├── monitor.py               # Progress monitoring
│   └── text_filter/
│       └── heuristic_parser.py  # Text-to-data parser
│
├── rag_core/                    # RAG engine components
│   ├── rag_engine.py            # Main RAG orchestrator
│   ├── retriever.py             # Vector search
│   ├── reranker.py              # Result reranking
│   ├── context_builder.py       # Context assembly
│   ├── llm_client.py            # LLM integration
│   ├── query_rewriter.py        # Query optimization
│   ├── conversation_manager.py  # Session management
│   ├── guardrails.py            # Safety checks
│   ├── cache.py                 # Response caching
│   └── analytics.py             # Usage tracking
│
└── chroma_db/                   # ChromaDB persistent storage
```

## Monitoring

Both pipelines provide real-time monitoring:

### Forward Pipeline Monitor
```
[████████████████████████████░░] 93% | ETA: 00:02:15 | Elapsed: 00:12:45
Rows in DB: 9300/10000 | CPU: 45% | RAM: 2.1GB/16GB | VRAM: 1.2GB/8GB

Worker-0: 3100 rows | batch=32 | CPU: 22% | RAM: 512MB
Worker-1: 3050 rows | batch=32 | CPU: 23% | RAM: 498MB
Worker-2: 3150 rows | batch=32 | CPU: 24% | RAM: 520MB
```

### Reverse Pipeline Monitor
```
[████████████████████████████░░] 87% | ETA: 00:01:30 | Elapsed: 00:08:20
Rows in DB: 8700/10000 | CPU: 38% | RAM: 1.8GB/16GB

Worker-0: parsed=4350 fails=12 | CPU: 19% | RAM: 450MB
Worker-1: parsed=4350 fails=8  | CPU: 18% | RAM: 445MB
```

## Troubleshooting

### Forward Pipeline Issues

**Problem**: Embeddings are slow
- Check GPU availability: `nvidia-smi`
- Reduce batch size in hardware detector
- Use smaller embedding model

**Problem**: Out of memory
- Reduce worker count
- Decrease chunk size
- Close other applications

### Reverse Pipeline Issues

**Problem**: NULL constraint violations
- Schema reconciler uses relaxed mode by default
- Check constraint_reader.py for PostgreSQL constraints
- Verify all columns included in forward template

**Problem**: Data recovery incomplete
- Ensure forward pipeline includes all column types (ID, date, semantic, numeric)
- Check heuristic_parser.py for parsing rules
- Verify ChromaDB IDs match original row IDs

### RAG Engine Issues

**Problem**: No results returned
- Verify ChromaDB has embedded data
- Check collection name matches
- Increase top_k parameter

**Problem**: Slow responses
- Enable caching in config
- Reduce context window size
- Use faster reranking model

## Performance Tips

1. **GPU Acceleration**: Install CUDA for 10-50x faster embeddings
2. **Batch Tuning**: Larger batches = faster throughput (if memory allows)
3. **Worker Scaling**: More workers = better CPU utilization (up to core count)
4. **Checkpoint Frequency**: Balance between resume granularity and overhead
5. **Schema Optimization**: Exclude unnecessary columns to reduce embedding size

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]

## Support

For issues, questions, or contributions, please [contact information or issue tracker].
