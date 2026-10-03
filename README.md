# VectorDB: Vector Search Engine and RAG Pipeline in Java

![Java](https://img.shields.io/badge/Java-17%2B-orange)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A vector database written **from scratch in pure Java**, with no libraries and no frameworks. It implements three nearest-neighbour search algorithms side by side (**HNSW**, **KD-Tree**, **Brute Force**) behind a REST API and a web UI, and uses them to power a **Retrieval-Augmented Generation (RAG)** pipeline that answers questions about your own documents using a local LLM (Ollama).

It is the same core idea behind Pinecone, Weaviate, Chroma and Milvus, built small enough to read end to end.

<!-- Add a screenshot: save it as docs/screenshot.png, then replace this line with:  ![App screenshot](docs/screenshot.png) -->

## Highlights

- **HNSW index implemented from the paper** (Malkov & Yashunin): multilayer small-world graph, greedy descent, beam search, bidirectional linking with neighbour pruning.
- **Measured accuracy:** reaches **99.8% recall@10** against exact search on 5,000 vectors (64 dimensions) while running **3.1x faster** than brute force. Reproducible with one command (see [Benchmarks](#benchmarks)).
- **End-to-end RAG:** text is chunked, embedded (768-D, `nomic-embed-text`), indexed, retrieved by semantic similarity, and passed to `llama3.2` to generate a grounded answer. Everything runs locally; no API keys or cloud costs.
- **Zero dependencies:** HTTP server, HTTP client, JSON parser, concurrency control and all data structures use only the JDK.
- **Interactive web UI:** 2-D PCA scatter plot of the semantic space, side-by-side algorithm benchmark, document upload, and chat interface.

## Skills demonstrated

| Area | Where to look |
|---|---|
| **Data structures and algorithms** | `HNSW.java` (graph search, priority queues), `KDTree.java` (space partitioning, branch pruning), `BruteForce.java` |
| **Information retrieval / AI engineering** | `DocumentDB.java`, `TextChunker.java`, `OllamaClient.java`: embeddings, overlapping chunking, retrieval, prompt construction |
| **Backend and API design** | `Main.java`: 13 REST endpoints on the JDK `HttpServer`, CORS, routing, error handling, thread pool |
| **Concurrency** | `ReentrantReadWriteLock` in `VectorDB` and `DocumentDB`: concurrent reads, exclusive writes, and a thread pool so slow LLM calls never block searches |
| **Parsing and protocols** | `Json.java`: a recursive-descent JSON parser with full string-escape and Unicode support |
| **Performance engineering** | JIT warm-up, tight distance kernels over primitive `float[]` vectors, reproducible benchmark (`RecallBenchmark.java`) |
| **Software design** | Small single-purpose classes, functional `DistFn` interface for pluggable metrics, immutable `record` types |

## How it works

```
 Your text
    |  split into overlapping 250-word chunks
    v
 Ollama (nomic-embed-text)      -> 768-dimensional vector per chunk
    |
    v
 HNSW index (this project)      -> multilayer graph, O(log N) approximate search
    |
    v
 Top-k most similar chunks      -> retrieved context
    |
    v
 Ollama (llama3.2)              -> answer grounded in your documents
```

**Algorithms in this repo**

| Algorithm | Search cost | Result | Notes |
|---|---|---|---|
| Brute Force | O(N·d) | Exact | The baseline everything is measured against |
| KD-Tree | ~O(log N) in low dimensions | Exact for Euclidean | Degrades towards brute force as dimensions grow |
| HNSW | ~O(log N) | Approximate, tunable via `ef` | Scales to high-dimensional embeddings |

Three distance metrics are supported: cosine, Euclidean and Manhattan.

## Benchmarks

Run it yourself:

```bash
javac -d out src/vectordb/*.java
java -cp out vectordb.RecallBenchmark
```

Setup: 5,000 random 64-D vectors, cosine distance, k = 10, 200 queries, HNSW with M = 16 and efConstruction = 200. Recall is measured against exact brute-force results.

| ef (search beam width) | Recall@10 |
|---:|---:|
| 10 | 52.1% |
| 50 | 91.5% |
| 100 | 98.4% |
| 200 | 99.8% |
| 400 | 100.0% |

Latency at `ef = 50` (after JIT warm-up): **HNSW 308 µs vs brute force 966 µs per query (3.1x faster)**.

Recall climbing smoothly to 100% as `ef` grows is the expected signature of a correct HNSW. This is a single run in a Linux container, so absolute timings will differ on your machine. Random Gaussian data has no cluster structure, which makes it a hard case for graph indexes. Real text embeddings often do better at the same settings. The gap over brute force also widens as the collection grows, because HNSW's cost grows roughly logarithmically while brute force grows linearly.

## Quick start

**Requirements:** JDK 17+ ([adoptium.net](https://adoptium.net)). For the document / RAG features also install [Ollama](https://ollama.com) and run:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

**Build and run** (from the project root, where `index.html` is):

```bash
javac -d out src/vectordb/*.java
java -cp out vectordb.Main
```

Open **http://localhost:8080**.

| Tab | What it does |
|---|---|
| **Search** | Query 20 pre-loaded 16-D semantic vectors (CS, Math, Food, Sports) with any algorithm and metric. Click *Compare all algos* to benchmark all three live. |
| **Documents** | Paste text. It is chunked, embedded by Ollama, and indexed in HNSW. |
| **Ask AI** | Ask a question. The most relevant chunks are retrieved and the LLM answers from them. |

Search works without Ollama. Only the Documents and Ask AI tabs need it.

**Configuration** (environment variables): `OLLAMA_GEN_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_HOST`, `OLLAMA_PORT`, `PORT`. For example, a faster model on a slow laptop:

```powershell
$env:OLLAMA_GEN_MODEL = "llama3.2:1b"
java -cp out vectordb.Main
```

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=...&k=5&metric=cosine&algo=hnsw` | k-nearest-neighbour search |
| POST | `/insert` | Insert a 16-D demo vector |
| DELETE | `/delete/:id` | Delete a demo vector |
| GET | `/items` | List demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Time all three algorithms |
| GET | `/hnsw-info` | Graph structure: nodes, edges, per-layer statistics |
| GET | `/stats`, `/status` | Database and Ollama status |
| POST | `/doc/insert` | Chunk, embed and store a document |
| GET | `/doc/list` | List stored chunks |
| DELETE | `/doc/delete/:id` | Delete a chunk |
| POST | `/doc/search` | Retrieval only |
| POST | `/doc/ask` | Full RAG: embed question, retrieve, generate |

Example:

```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Engineering notes

Decisions that go beyond a straight translation of the algorithms:

- **Read/write locking.** Searches take a shared read lock and inserts/deletes take an exclusive write lock, so many queries run in parallel while the graph is never read mid-update.
- **Slow calls isolated.** LLM generation can take 10 to 30 seconds on a laptop CPU. The server uses a fixed thread pool so these requests never block fast vector searches.
- **Correct JSON handling.** Ollama can emit `\uXXXX` escapes in generated text, so the project includes a real recursive-descent parser instead of string scanning.
- **Locale-safe output.** Numbers are formatted with `Locale.ROOT`, so a machine set to a locale that uses decimal commas cannot produce invalid JSON.
- **JVM warm-up.** The JIT compiles hot loops lazily, so the first queries run in the interpreter and would make the algorithm comparison look wrong. A short warm-up burst at startup keeps the live benchmark fair.
- **HNSW delete handling.** Deleting removes the node from every neighbour list. If the entry point is deleted, the highest-layer remaining node takes over, so search always has a valid starting point.
- **Graceful degradation.** If Ollama is offline, the server still starts, vector search still works, and the document endpoints return clear error messages.

## Limitations and roadmap

Being upfront about the current scope:

- Data is held in memory, so it is lost on restart. *Next: snapshot the index to disk.*
- HNSW deletion is a hard delete without graph repair, so recall can drift down after many deletions. *Next: tombstones plus periodic rebuild.*
- The KD-Tree's pruning rule is only exact for Euclidean distance, so with cosine or Manhattan it can differ slightly from brute force.
- There is no automated test suite yet. *Next: JUnit tests for each index against brute-force ground truth.*
- Streaming LLM responses and metadata filtering are not implemented.

## Project structure

```
VectorDB-Java/
├── index.html                    web UI (PCA scatter plot, chat, benchmark)
└── src/vectordb/
    ├── Main.java                 HTTP server and REST routes
    ├── HNSW.java                 hierarchical navigable small-world graph
    ├── KDTree.java               k-d tree
    ├── BruteForce.java           exact baseline
    ├── VectorDB.java             demo database: one store, three indexes
    ├── DocumentDB.java           HNSW over real 768-D embeddings
    ├── OllamaClient.java         embeddings and generation over HTTP
    ├── TextChunker.java          overlapping word-window chunking
    ├── Json.java                 JSON parser and string quoting
    ├── Metrics.java              cosine / Euclidean / Manhattan
    ├── RecallBenchmark.java      reproducible accuracy and speed benchmark
    └── DemoData.java, VectorItem.java, Pair.java, DistFn.java
```

## Credits and license

Built by **Ankur Shrivastava** ([@Ankurshrivastavaa](https://github.com/Ankurshrivastavaa)).

Released under the **MIT License**.
