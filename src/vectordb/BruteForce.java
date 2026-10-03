package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** O(N*d) exact scan. The baseline everything else is compared against. */
public final class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) { items.add(v); }

    public void remove(int id) { items.removeIf(v -> v.id() == id); }

    public List<Pair> knn(float[] q, int k, DistFn dist) {
        List<Pair> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new Pair(dist.apply(q, v.emb()), v.id()));
        Collections.sort(r);
        return r.size() > k ? new ArrayList<>(r.subList(0, k)) : r;
    }
}
