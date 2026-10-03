package vectordb;

/** A stored vector plus its metadata (mirrors the C++ VectorItem struct). */
public record VectorItem(int id, String metadata, String category, float[] emb) {}
