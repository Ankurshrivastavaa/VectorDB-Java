package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/**
 * Hierarchical Navigable Small World graph (Malkov and Yashunin, 2016).
 *
 * Every node gets a random max layer. Layer 0 holds all nodes with up to 2*M
 * links each; higher layers are exponentially sparser with up to M links.
 * Search greedily descends from the top layer, then runs a beam search
 * (width ef) on layer 0.
 *
 * Not thread-safe by itself. The owning database provides the locking.
 */
public final class HNSW {
    private static final class Node {
        final VectorItem item;
        final int maxLyr;
        final List<List<Integer>> nbrs; // nbrs.get(layer) = neighbour ids
        Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = new ArrayList<>(maxLyr + 1);
            for (int i = 0; i <= maxLyr; i++) nbrs.add(new ArrayList<>());
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int m, m0, efBuild;
    private final double mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.m = m;
        this.m0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1.0 / Math.log(m);
    }

    private int randLevel() {
        double u = 1.0 - rng.nextDouble(); // (0, 1], avoids log(0)
        return (int) Math.floor(-Math.log(u) * mL);
    }

    /** Beam search on one layer. Returns up to ef results sorted nearest first. */
    private List<Pair> searchLayer(float[] q, int ep, int ef, int layer, DistFn dist) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Pair> cands = new PriorityQueue<>();                       // min-heap
        PriorityQueue<Pair> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

        float d0 = dist.apply(q, graph.get(ep).item.emb());
        visited.add(ep);
        cands.add(new Pair(d0, ep));
        found.add(new Pair(d0, ep));

        while (!cands.isEmpty()) {
            Pair c = cands.poll();
            if (found.size() >= ef && c.dist() > found.peek().dist()) break;
            Node cn = graph.get(c.id());
            if (layer >= cn.nbrs.size()) continue;
            for (int nid : cn.nbrs.get(layer)) {
                if (!visited.add(nid)) continue;
                Node nn = graph.get(nid);
                if (nn == null) continue;
                float nd = dist.apply(q, nn.item.emb());
                if (found.size() < ef || nd < found.peek().dist()) {
                    cands.add(new Pair(nd, nid));
                    found.add(new Pair(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }
        List<Pair> res = new ArrayList<>(found);
        Collections.sort(res);
        return res;
    }

    public void insert(VectorItem item, DistFn dist) {
        int id = item.id();
        int lvl = randLevel();
        graph.put(id, new Node(item, lvl));

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        // Phase 1: greedy descent through layers above the new node's level
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Pair> w = searchLayer(item.emb(), ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id();
            }
        }
        // Phase 2: from min(topLayer, lvl) down to 0, find neighbours and link
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<Pair> w = searchLayer(item.emb(), ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? m0 : m;

            List<Integer> sel = new ArrayList<>();
            for (int i = 0; i < Math.min(w.size(), maxM); i++) sel.add(w.get(i).id());
            graph.get(id).nbrs.set(lc, sel);

            for (int nid : sel) {
                Node nb = graph.get(nid);
                if (nb == null) continue;
                while (nb.nbrs.size() <= lc) nb.nbrs.add(new ArrayList<>());
                List<Integer> conn = nb.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) { // prune back to the maxM closest
                    List<Pair> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cn = graph.get(c);
                        if (cn != null) ds.add(new Pair(dist.apply(nb.item.emb(), cn.item.emb()), c));
                    }
                    Collections.sort(ds);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id());
                }
            }
            if (!w.isEmpty()) ep = w.get(0).id();
        }
        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public List<Pair> knn(float[] q, int k, int ef, DistFn dist) {
        if (entryPt == -1 || k <= 0) return new ArrayList<>();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Pair> w = searchLayer(q, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id();
            }
        }
        List<Pair> w = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return w.size() > k ? new ArrayList<>(w.subList(0, k)) : w;
    }

    /**
     * Hard delete: strip the id from every neighbour list, then drop the node.
     * O(N * layers). Fine at this scale; production systems use tombstones.
     * Entry point becomes the remaining node with the highest layer.
     */
    public void remove(int id) {
        if (graph.remove(id) == null) return;
        for (Node nd : graph.values())
            for (List<Integer> layer : nd.nbrs) layer.remove(Integer.valueOf(id));

        if (entryPt == id) {
            entryPt = -1;
            topLayer = -1;
            for (Node nd : graph.values()) {
                if (nd.maxLyr > topLayer) {
                    topLayer = nd.maxLyr;
                    entryPt = nd.item.id();
                }
            }
        }
    }

    public int size() { return graph.size(); }

    // ---- graph introspection for the /hnsw-info endpoint ----

    public record NodeView(int id, String metadata, String category, int maxLyr) {}
    public record EdgeView(int src, int dst, int lyr) {}
    public record GraphInfo(int topLayer, int nodeCount,
                            int[] nodesPerLayer, int[] edgesPerLayer,
                            List<NodeView> nodes, List<EdgeView> edges) {}

    public GraphInfo getInfo() {
        int maxL = Math.max(topLayer + 1, 1);
        int[] nodesPerLayer = new int[maxL];
        int[] edgesPerLayer = new int[maxL];
        List<NodeView> nodes = new ArrayList<>();
        List<EdgeView> edges = new ArrayList<>();

        for (Map.Entry<Integer, Node> e : graph.entrySet()) {
            int id = e.getKey();
            Node nd = e.getValue();
            nodes.add(new NodeView(id, nd.item.metadata(), nd.item.category(), nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                nodesPerLayer[lc]++;
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) { // count each undirected edge once
                            edgesPerLayer[lc]++;
                            edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }
        return new GraphInfo(topLayer, graph.size(), nodesPerLayer, edgesPerLayer, nodes, edges);
    }
}
