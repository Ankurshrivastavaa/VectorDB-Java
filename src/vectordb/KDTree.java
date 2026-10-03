package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Axis-aligned binary space partitioning tree.
 *
 * Note: the pruning test (|axis diff| < current worst distance) is a true
 * bound only for Euclidean distance. With cosine/Manhattan it can prune too
 * aggressively, so KD-Tree results may differ slightly from brute force.
 * That matches the original C++ behaviour.
 */
public final class KDTree {
    private static final class Node {
        final VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) { this.dims = dims; }

    public void insert(VectorItem v) { root = ins(root, v, 0); }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }

    private Node ins(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int ax = depth % dims;
        if (v.emb()[ax] < n.item.emb()[ax]) n.left = ins(n.left, v, depth + 1);
        else n.right = ins(n.right, v, depth + 1);
        return n;
    }

    public List<Pair> knn(float[] q, int k, DistFn dist) {
        if (k <= 0) return new ArrayList<>();
        // max-heap: worst of the current best-k sits on top
        PriorityQueue<Pair> heap = new PriorityQueue<>(Collections.reverseOrder());
        search(root, q, k, 0, dist, heap);
        List<Pair> r = new ArrayList<>(heap);
        Collections.sort(r);
        return r;
    }

    private void search(Node n, float[] q, int k, int depth, DistFn dist, PriorityQueue<Pair> heap) {
        if (n == null) return;
        float dn = dist.apply(q, n.item.emb());
        if (heap.size() < k || dn < heap.peek().dist()) {
            heap.add(new Pair(dn, n.item.id()));
            if (heap.size() > k) heap.poll();
        }
        int ax = depth % dims;
        float diff = q[ax] - n.item.emb()[ax];
        Node closer = diff < 0 ? n.left : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        search(closer, q, k, depth + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist())
            search(farther, q, k, depth + 1, dist, heap);
    }
}
