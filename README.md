# VectorDB: a Vector Database from Scratch in Java

A Java 17+ port of the C++ VectorDB. It implements **HNSW**, **KD-Tree** and **Brute Force** search side by side, plus a **RAG pipeline** powered by a local LLM via Ollama.

The REST API and JSON shapes are identical to the C++ version, so the original `index.html` works **unchanged**. There are **no dependencies**: no Maven, no Gradle, no libraries. It uses only the JDK (`com.sun.net.httpserver`, `java.net.http`).

## Prerequisites

1. **JDK 17 or newer** (a JDK, not just a JRE, because you need `javac`). Get one from https://adoptium.net
2. **Ollama** from https://ollama.com, then:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

Check your setup:
```
javac -version
ollama list
```

Ollama is only needed for the **Documents** and **Ask AI** tabs. The demo-vector search works without it.

## Build and run

From the project folder (the one containing `index.html`):

```powershell
# Windows PowerShell
javac -d out src/vectordb/*.java
java -cp out vectordb.Main
```

```bash
# macOS / Linux
javac -d out src/vectordb/*.java
java -cp out vectordb.Main
```

Then open **http://localhost:8080**.

> Run it from the project root. The server serves `index.html` from the current working directory.

You should see:
```
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

## Configuration (environment variables)

There is no recompiling to switch models any more.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | HTTP port |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_GEN_MODEL` | `llama3.2` | Answer-generation model |

Faster answers on a slow laptop:
```powershell
ollama pull llama3.2:1b
$env:OLLAMA_GEN_MODEL = "llama3.2:1b"
java -cp out vectordb.Main
```

## REST API

Same as the C++ version.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,...,f16&k=5&metric=cosine&algo=hnsw` | K-NN search (`algo`: hnsw / kdtree / bruteforce) |
| POST | `/insert` | Insert a demo vector `{"metadata","category","embedding":[16 floats]}` |
| DELETE | `/delete/:id` | Delete a demo vector |
| GET | `/items` | List demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Time all 3 algorithms |
| GET | `/hnsw-info` | HNSW layer and edge statistics |
| GET | `/stats`, `/status` | Database / Ollama status |
| POST | `/doc/insert` | `{"title","text"}`: chunk, embed, store |
| GET | `/doc/list` | List stored chunks |
| DELETE | `/doc/delete/:id` | Delete a chunk |
| POST | `/doc/search` | `{"question","k"}`: retrieval only |
| POST | `/doc/ask` | `{"question","k"}`: full RAG (retrieve + generate) |

## Source layout

```
VectorDB-Java/
├── index.html                    frontend (unchanged from the C++ project)
└── src/vectordb/
    ├── Main.java                 HTTP server and all routes
    ├── VectorDB.java             16D demo DB: one store, three indexes
    ├── DocumentDB.java           HNSW over real Ollama embeddings
    ├── HNSW.java                 multilayer small-world graph
    ├── KDTree.java               axis-aligned space partitioning
    ├── BruteForce.java           exact O(N*d) baseline
    ├── Metrics.java              cosine / euclidean / manhattan
    ├── OllamaClient.java         /api/embeddings and /api/generate
    ├── TextChunker.java          250-word overlapping chunks
    ├── Json.java                 small JSON parser and quoting helper
    ├── DemoData.java             the 20 built-in 16D vectors
    └── VectorItem.java, Pair.java, DistFn.java
```

## C++ to Java mapping

| C++ | Java |
|---|---|
| `httplib.h` | `com.sun.net.httpserver.HttpServer` |
| `httplib::Client` to Ollama | `java.net.http.HttpClient` |
| hand-rolled JSON string scanning | `Json.java` (a real parser) |
| `std::mutex` | `ReentrantReadWriteLock` (concurrent searches, exclusive writes) |
| `std::priority_queue` | `PriorityQueue<Pair>` |
| `std::vector<float>` | `float[]` |
| `std::pair<float,int>` | `record Pair(float, int)` |

## Differences from the C++ version

- **JSON parsing is more robust.** The C++ `extractStr` did not understand `\uXXXX` escapes, which Ollama can emit in generated text. `Json.java` does.
- **Deleting from HNSW re-picks the entry point.** The C++ code kept a stale `topLayer` after deleting the entry node. Here the entry point becomes the remaining node with the highest layer.
- **JIT warm-up at startup.** The JVM compiles hot loops lazily, so without warm-up the first queries are ~50x slower and the algorithm comparison looks wrong.
- **Random levels differ.** `java.util.Random(42)` is not `std::mt19937(42)`, so the HNSW graph is shaped slightly differently, with the same statistics.
- **Word counts in `/doc/list`** now count whitespace-separated words, not spaces.

## Known limitations (inherited from the original design)

- The KD-Tree prunes with an axis-distance bound that is only exact for Euclidean distance. With cosine or Manhattan its results can occasionally differ from brute force.
- HNSW deletion is a hard delete without graph repair, so recall can drift down after many deletions.
- Everything lives in memory. Restarting the server clears inserted data.

## License

MIT
