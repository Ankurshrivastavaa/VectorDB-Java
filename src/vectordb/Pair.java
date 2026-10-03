package vectordb;

/**
 * (distance, id) pair. Orders by distance, then id, which matches how
 * std::pair<float,int> compares in the C++ version.
 */
public record Pair(float dist, int id) implements Comparable<Pair> {
    @Override
    public int compareTo(Pair o) {
        int c = Float.compare(dist, o.dist);
        return c != 0 ? c : Integer.compare(id, o.id);
    }
}
