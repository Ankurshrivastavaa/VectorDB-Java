package vectordb;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** Thin client for the local Ollama REST API (/api/embeddings, /api/generate). */
public final class OllamaClient {
    private final String baseUrl;
    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();

    public final String embedModel;
    public final String genModel;

    public OllamaClient(String host, int port, String embedModel, String genModel) {
        this.baseUrl = "http://" + host + ":" + port;
        this.embedModel = embedModel;
        this.genModel = genModel;
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2)).GET().build();
            return http.send(req, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /** Returns an empty array if Ollama is down or the model is missing. */
    public float[] embed(String text) {
        String body = "{\"model\":" + Json.quote(embedModel) + ",\"prompt\":" + Json.quote(text) + "}";
        String resp = post("/api/embeddings", body, 30);
        if (resp == null) return new float[0];
        return Json.getFloats(Json.parseObject(resp), "embedding");
    }

    /** Returns an "ERROR: ..." string if Ollama is unavailable. */
    public String generate(String prompt) {
        String body = "{\"model\":" + Json.quote(genModel) + ",\"prompt\":" + Json.quote(prompt)
                + ",\"stream\":false}";
        String resp = post("/api/generate", body, 180); // LLMs on laptop CPUs are slow
        if (resp == null) return "ERROR: Ollama unavailable. Run: ollama serve";
        return Json.getString(Json.parseObject(resp), "response");
    }

    private String post(String path, String jsonBody, int timeoutSeconds) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                    .timeout(Duration.ofSeconds(timeoutSeconds))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
            HttpResponse<String> r = http.send(req, HttpResponse.BodyHandlers.ofString());
            return r.statusCode() == 200 ? r.body() : null;
        } catch (IOException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }
}
