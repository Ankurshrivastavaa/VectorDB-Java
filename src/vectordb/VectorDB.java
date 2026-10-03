package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;

/** Demo database: one store, three indexes (brute force, KD-tree, HNSW). */
public final class VectorDB {
    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw = new HNSW(16, 200);
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private int nextId = 1;
    public final int dims;

    public VectorDB(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
    }

    public int insert(String meta, String cat, float[] emb, DistFn dist) {
        lock.writeLock().lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id(), v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public boolean remove(int id) {
        lock.writeLock().lock();
        try {
            if (store.remove(id) == null) return false;
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values())); // KD-tree has no cheap delete
            return true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public record Hit(int id, String meta, String cat, float[] emb, float dist) {}
    public record SearchOut(List<Hit> hits, long us, String algo, String metric) {}

    public SearchOut search(float[] q, int k, String metric, String algo) {
        return read(() -> {
            DistFn dfn = Metrics.byName(metric);
            long t0 = System.nanoTime();
            List<Pair> raw;
            if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
            else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
            else raw = hnsw.knn(q, k, 50, dfn);
            long us = (System.nanoTime() - t0) / 1000;

            List<Hit> hits = new ArrayList<>();
            for (Pair p : raw) {
                VectorItem v = store.get(p.id());
                if (v != null) hits.add(new Hit(v.id(), v.metadata(), v.category(), v.emb(), p.dist()));
            }
            return new SearchOut(hits, us, algo, metric);
        });
    }

    public record BenchOut(long bfUs, long kdUs, long hnswUs, int n) {}

    public BenchOut benchmark(float[] q, int k, String metric) {
        return read(() -> {
            DistFn dfn = Metrics.byName(metric);
            long bfUs = timeUs(() -> bf.knn(q, k, dfn));
            long kdUs = timeUs(() -> kdt.knn(q, k, dfn));
            long hnswUs = timeUs(() -> hnsw.knn(q, k, 50, dfn));
            return new BenchOut(bfUs, kdUs, hnswUs, store.size());
        });
    }

    private static long timeUs(Runnable r) {
        long t = System.nanoTime();
        r.run();
        return (System.nanoTime() - t) / 1000;
    }

    public List<VectorItem> all() { return read(() -> new ArrayList<>(store.values())); }

    public HNSW.GraphInfo hnswInfo() { return read(hnsw::getInfo); }

    public int size() { return read(store::size); }

    private <T> T read(Supplier<T> s) {
        lock.readLock().lock();
        try {
            return s.get();
        } finally {
            lock.readLock().unlock();
        }
    }
}
