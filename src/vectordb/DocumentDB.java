package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** Document-chunk store backed by HNSW over real (e.g. 768D) Ollama embeddings. */
public final class DocumentDB {
    public record DocItem(int id, String title, String text, float[] emb) {}
    public record DocHit(float dist, DocItem doc) {}

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce(); // exact search for tiny collections
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private int nextId = 1;
    private int dims = 0; // set by the first inserted embedding

    public int insert(String title, String text, float[] emb) {
        lock.writeLock().lock();
        try {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id(), item);
            VectorItem vi = new VectorItem(item.id(), title, "doc", emb);
            hnsw.insert(vi, Metrics.COSINE);
            bf.insert(vi);
            return item.id();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Top-k chunks by cosine distance, dropping anything farther than maxDist. */
    public List<DocHit> search(float[] q, int k, float maxDist) {
        lock.readLock().lock();
        try {
            if (store.isEmpty()) return new ArrayList<>();
            List<Pair> raw = store.size() < 10
                    ? bf.knn(q, k, Metrics.COSINE)
                    : hnsw.knn(q, k, 50, Metrics.COSINE);
            List<DocHit> out = new ArrayList<>();
            for (Pair p : raw) {
                DocItem d = store.get(p.id());
                if (d != null && p.dist() <= maxDist) out.add(new DocHit(p.dist(), d));
            }
            return out;
        } finally {
            lock.readLock().unlock();
        }
    }

    public List<DocHit> search(float[] q, int k) { return search(q, k, 0.7f); }

    public boolean remove(int id) {
        lock.writeLock().lock();
        try {
            if (store.remove(id) == null) return false;
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public List<DocItem> all() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.readLock().unlock();
        }
    }

    public int size() {
        lock.readLock().lock();
        try {
            return store.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    public int getDims() {
        lock.readLock().lock();
        try {
            return dims;
        } finally {
            lock.readLock().unlock();
        }
    }
}
