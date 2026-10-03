package vectordb;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * VectorDB server: REST API + static index.html.
 * Port of the original C++ main.cpp. Endpoints and JSON shapes are identical,
 * so the existing index.html works unchanged.
 */
public final class Main {
    private static final int DIMS = 16; // demo vectors; doc embedding dims come from Ollama
    private static final Pattern DELETE_DEMO = Pattern.compile("^/delete/(\\d+)$");
    private static final Pattern DELETE_DOC = Pattern.compile("^/doc/delete/(\\d+)$");

    private final VectorDB db = new VectorDB(DIMS);
    private final DocumentDB docDB = new DocumentDB();
    private final OllamaClient ollama;

    private Main(OllamaClient ollama) { this.ollama = ollama; }

    public static void main(String[] args) throws IOException {
        int port = Integer.parseInt(env("PORT", "8080"));
        OllamaClient ollama = new OllamaClient(
                env("OLLAMA_HOST", "127.0.0.1"),
                Integer.parseInt(env("OLLAMA_PORT", "11434")),
                env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                env("OLLAMA_GEN_MODEL", "llama3.2"));

        Main app = new Main(ollama);
        DemoData.load(app.db);
        warmUp(app.db);

        boolean up = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + port);
        System.out.println(app.db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (up ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (up) System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", app::handle);
        // Thread pool: slow LLM calls must not block other requests
        server.setExecutor(Executors.newFixedThreadPool(16));
        server.start();
    }

    /**
     * The JVM compiles hot loops lazily, so the very first queries run in the
     * slow interpreter and would make the UI's speed comparison look unfair.
     * A burst of throw-away searches at startup gets the JIT going.
     */
    private static void warmUp(VectorDB db) {
        float[] q = new float[DIMS];
        java.util.Arrays.fill(q, 0.5f);
        for (String metric : new String[]{"cosine", "euclidean", "manhattan"})
            for (int i = 0; i < 2000; i++) db.benchmark(q, 5, metric);
    }

    private static String env(String k, String def) {
        String v = System.getenv(k);
        return v == null || v.isBlank() ? def : v;
    }

    // =====================================================================
    //  ROUTER
    // =====================================================================

    private void handle(HttpExchange ex) throws IOException {
        try {
            addCors(ex.getResponseHeaders());
            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();

            if (method.equals("OPTIONS")) {
                ex.sendResponseHeaders(204, -1);
                return;
            }

            if (method.equals("DELETE")) {
                Matcher m;
                if ((m = DELETE_DEMO.matcher(path)).matches()) {
                    json(ex, "{\"ok\":" + db.remove(Integer.parseInt(m.group(1))) + "}");
                } else if ((m = DELETE_DOC.matcher(path)).matches()) {
                    json(ex, "{\"ok\":" + docDB.remove(Integer.parseInt(m.group(1))) + "}");
                } else {
                    notFound(ex);
                }
                return;
            }

            switch (method + " " + path) {
                case "GET /" -> serveIndex(ex);
                case "GET /search" -> search(ex);
                case "POST /insert" -> insert(ex);
                case "GET /items" -> items(ex);
                case "GET /benchmark" -> benchmark(ex);
                case "GET /hnsw-info" -> hnswInfo(ex);
                case "GET /stats" -> stats(ex);
                case "GET /status" -> status(ex);
                case "POST /doc/insert" -> docInsert(ex);
                case "GET /doc/list" -> docList(ex);
                case "POST /doc/search" -> docSearch(ex);
                case "POST /doc/ask" -> docAsk(ex);
                default -> notFound(ex);
            }
        } catch (Exception e) {
            e.printStackTrace();
            try {
                json(ex, 500, "{\"error\":" + Json.quote(String.valueOf(e.getMessage())) + "}");
            } catch (IOException ignored) {
                // client already gone
            }
        } finally {
            ex.close();
        }
    }

    // =====================================================================
    //  DEMO VECTOR ENDPOINTS
    // =====================================================================

    private void search(HttpExchange ex) throws IOException {
        Map<String, String> q = query(ex.getRequestURI());
        float[] v = parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) {
            json(ex, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = parseK(q.get("k"), 5);
        String metric = q.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";
        String algo = q.getOrDefault("algo", "hnsw");
        if (algo.isEmpty()) algo = "hnsw";

        VectorDB.SearchOut out = db.search(v, k, metric, algo);
        StringBuilder sb = new StringBuilder("{\"results\":[");
        for (int i = 0; i < out.hits().size(); i++) {
            if (i > 0) sb.append(',');
            VectorDB.Hit h = out.hits().get(i);
            sb.append("{\"id\":").append(h.id())
              .append(",\"metadata\":").append(Json.quote(h.meta()))
              .append(",\"category\":").append(Json.quote(h.cat()))
              .append(",\"distance\":").append(fmt(h.dist(), 6))
              .append(",\"embedding\":").append(jVec(h.emb())).append('}');
        }
        sb.append("],\"latencyUs\":").append(out.us())
          .append(",\"algo\":").append(Json.quote(out.algo()))
          .append(",\"metric\":").append(Json.quote(out.metric())).append('}');
        json(ex, sb.toString());
    }

    private void insert(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.parseObject(readBody(ex));
        String meta = Json.getString(body, "metadata");
        String cat = Json.getString(body, "category");
        float[] emb = Json.getFloats(body, "embedding");
        if (meta.isEmpty() || emb.length != DIMS) {
            json(ex, "{\"error\":\"invalid body\"}");
            return;
        }
        int id = db.insert(meta, cat, emb, Metrics.COSINE);
        json(ex, "{\"id\":" + id + "}");
    }

    private void items(HttpExchange ex) throws IOException {
        List<VectorItem> items = db.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            VectorItem v = items.get(i);
            sb.append("{\"id\":").append(v.id())
              .append(",\"metadata\":").append(Json.quote(v.metadata()))
              .append(",\"category\":").append(Json.quote(v.category()))
              .append(",\"embedding\":").append(jVec(v.emb())).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    private void benchmark(HttpExchange ex) throws IOException {
        Map<String, String> q = query(ex.getRequestURI());
        float[] v = parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) {
            json(ex, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = parseK(q.get("k"), 5);
        String metric = q.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";

        VectorDB.BenchOut b = db.benchmark(v, k, metric);
        json(ex, "{\"bruteforceUs\":" + b.bfUs() + ",\"kdtreeUs\":" + b.kdUs()
                + ",\"hnswUs\":" + b.hnswUs() + ",\"itemCount\":" + b.n() + "}");
    }

    private void hnswInfo(HttpExchange ex) throws IOException {
        HNSW.GraphInfo gi = db.hnswInfo();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"topLayer\":").append(gi.topLayer())
          .append(",\"nodeCount\":").append(gi.nodeCount())
          .append(",\"nodesPerLayer\":").append(intArray(gi.nodesPerLayer()))
          .append(",\"edgesPerLayer\":").append(intArray(gi.edgesPerLayer()))
          .append(",\"nodes\":[");
        for (int i = 0; i < gi.nodes().size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.NodeView n = gi.nodes().get(i);
            sb.append("{\"id\":").append(n.id())
              .append(",\"metadata\":").append(Json.quote(n.metadata()))
              .append(",\"category\":").append(Json.quote(n.category()))
              .append(",\"maxLyr\":").append(n.maxLyr()).append('}');
        }
        sb.append("],\"edges\":[");
        for (int i = 0; i < gi.edges().size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.EdgeView e = gi.edges().get(i);
            sb.append("{\"src\":").append(e.src())
              .append(",\"dst\":").append(e.dst())
              .append(",\"lyr\":").append(e.lyr()).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    private void stats(HttpExchange ex) throws IOException {
        json(ex, "{\"count\":" + db.size() + ",\"dims\":" + DIMS
                + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
    }

    // =====================================================================
    //  DOCUMENT + RAG ENDPOINTS
    // =====================================================================

    /** POST /doc/insert {"title":"...","text":"..."}: chunk, embed each chunk, store. */
    private void docInsert(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.parseObject(readBody(ex));
        String title = Json.getString(body, "title");
        String text = Json.getString(body, "text");
        if (title.isEmpty() || text.isEmpty()) {
            json(ex, "{\"error\":\"need title and text\"}");
            return;
        }

        List<String> chunks = TextChunker.chunk(text, 250, 30);
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                json(ex, "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: "
                        + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                return;
            }
            String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        StringBuilder sb = new StringBuilder("{\"ids\":[");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(ids.get(i));
        }
        sb.append("],\"chunks\":").append(chunks.size())
          .append(",\"dims\":").append(docDB.getDims()).append('}');
        json(ex, sb.toString());
    }

    private void docList(HttpExchange ex) throws IOException {
        List<DocumentDB.DocItem> docs = docDB.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocItem d = docs.get(i);
            String preview = d.text().length() > 120 ? d.text().substring(0, 120) + "…" : d.text();
            String stripped = d.text().strip();
            int words = stripped.isEmpty() ? 0 : stripped.split("\\s+").length;
            sb.append("{\"id\":").append(d.id())
              .append(",\"title\":").append(Json.quote(d.title()))
              .append(",\"preview\":").append(Json.quote(preview))
              .append(",\"words\":").append(words).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    /** POST /doc/search: retrieval only (used by the UI visualiser). */
    private void docSearch(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.parseObject(readBody(ex));
        String question = Json.getString(body, "question");
        int k = Json.getInt(body, "k", 3);
        if (question.isEmpty()) {
            json(ex, "{\"error\":\"need question\"}");
            return;
        }
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            json(ex, "{\"error\":\"Ollama unavailable\"}");
            return;
        }
        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);
        StringBuilder sb = new StringBuilder("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.doc().id())
              .append(",\"title\":").append(Json.quote(h.doc().title()))
              .append(",\"distance\":").append(fmt(h.dist(), 4)).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    /** POST /doc/ask: full RAG pipeline (embed, retrieve, generate). */
    private void docAsk(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.parseObject(readBody(ex));
        String question = Json.getString(body, "question");
        int k = Json.getInt(body, "k", 3);
        if (question.isEmpty()) {
            json(ex, "{\"error\":\"need question\"}");
            return;
        }

        // 1. embed the question
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            json(ex, "{\"error\":\"Ollama unavailable\"}");
            return;
        }
        // 2. retrieve top-k chunks
        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);

        // 3. build the prompt
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append('[').append(i + 1).append("] ").append(hits.get(i).doc().title()).append(":\n")
               .append(hits.get(i).doc().text()).append("\n\n");
        }
        String prompt =
                "You are a helpful assistant. Answer the user's question directly. "
              + "Use the provided context if it contains relevant information. "
              + "If it doesn't, just use your own general knowledge. "
              + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
              + "'the context doesn't mention'. Just answer the question naturally.\n\n"
              + "Context:\n" + ctx + "Question: " + question + "\n\nAnswer:";

        // 4. generate
        String answer = ollama.generate(prompt);

        // 5. respond
        StringBuilder sb = new StringBuilder();
        sb.append("{\"answer\":").append(Json.quote(answer))
          .append(",\"model\":").append(Json.quote(ollama.genModel))
          .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.doc().id())
              .append(",\"title\":").append(Json.quote(h.doc().title()))
              .append(",\"text\":").append(Json.quote(h.doc().text()))
              .append(",\"distance\":").append(fmt(h.dist(), 4)).append('}');
        }
        sb.append("],\"docCount\":").append(docDB.size()).append('}');
        json(ex, sb.toString());
    }

    private void status(HttpExchange ex) throws IOException {
        boolean up = ollama.isAvailable();
        json(ex, "{\"ollamaAvailable\":" + up
                + ",\"embedModel\":" + Json.quote(ollama.embedModel)
                + ",\"genModel\":" + Json.quote(ollama.genModel)
                + ",\"docCount\":" + docDB.size()
                + ",\"docDims\":" + docDB.getDims()
                + ",\"demoDims\":" + DIMS
                + ",\"demoCount\":" + db.size() + "}");
    }

    private void serveIndex(HttpExchange ex) throws IOException {
        Path index = Path.of("index.html");
        if (!Files.isRegularFile(index)) {
            notFound(ex);
            return;
        }
        send(ex, 200, "text/html; charset=utf-8", Files.readAllBytes(index));
    }

    // =====================================================================
    //  HTTP / JSON HELPERS
    // =====================================================================

    private static void addCors(Headers h) {
        h.set("Access-Control-Allow-Origin", "*");
        h.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        h.set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void json(HttpExchange ex, String body) throws IOException { json(ex, 200, body); }

    private static void json(HttpExchange ex, int status, String body) throws IOException {
        send(ex, status, "application/json; charset=utf-8", body.getBytes(StandardCharsets.UTF_8));
    }

    private static void notFound(HttpExchange ex) throws IOException {
        send(ex, 404, "text/plain; charset=utf-8", "Not found".getBytes(StandardCharsets.UTF_8));
    }

    private static void send(HttpExchange ex, int status, String type, byte[] bytes) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    private static Map<String, String> query(URI uri) {
        Map<String, String> out = new HashMap<>();
        String raw = uri.getRawQuery();
        if (raw == null || raw.isEmpty()) return out;
        for (String pair : raw.split("&")) {
            int eq = pair.indexOf('=');
            String k = eq < 0 ? pair : pair.substring(0, eq);
            String v = eq < 0 ? "" : pair.substring(eq + 1);
            out.put(URLDecoder.decode(k, StandardCharsets.UTF_8),
                    URLDecoder.decode(v, StandardCharsets.UTF_8));
        }
        return out;
    }

    /** Comma-separated floats; unparseable tokens are skipped (same as the C++ parseVec). */
    private static float[] parseVec(String s) {
        List<Float> vals = new ArrayList<>();
        for (String t : s.split(",")) {
            try {
                vals.add(Float.parseFloat(t.trim()));
            } catch (NumberFormatException ignored) {
                // skip
            }
        }
        float[] out = new float[vals.size()];
        for (int i = 0; i < out.length; i++) out[i] = vals.get(i);
        return out;
    }

    private static int parseK(String s, int def) {
        try {
            return Math.max(1, Integer.parseInt(s));
        } catch (RuntimeException e) { // null or malformed
            return def;
        }
    }

    // Locale.ROOT matters: some locales would print "0,9" and break the JSON.
    private static String fmt(float v, int decimals) {
        return String.format(Locale.ROOT, "%." + decimals + "f", v);
    }

    private static String jVec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(fmt(v[i], 4));
        }
        return sb.append(']').toString();
    }

    private static String intArray(int[] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(a[i]);
        }
        return sb.append(']').toString();
    }
}
