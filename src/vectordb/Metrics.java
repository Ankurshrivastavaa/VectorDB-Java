package vectordb;

/** The three distance metrics. */
public final class Metrics {
    private Metrics() {}

    public static final DistFn EUCLIDEAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    };

    /** Cosine DISTANCE = 1 - cosine similarity. */
    public static final DistFn COSINE = (a, b) -> {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    };

    public static final DistFn MANHATTAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    };

    /** Unknown names fall back to Euclidean, same as the C++ getDistFn(). */
    public static DistFn byName(String name) {
        if ("cosine".equals(name)) return COSINE;
        if ("manhattan".equals(name)) return MANHATTAN;
        return EUCLIDEAN;
    }
}
