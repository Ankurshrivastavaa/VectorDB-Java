package vectordb;

/** A distance function between two equal-length vectors. Smaller = closer. */
@FunctionalInterface
public interface DistFn {
    float apply(float[] a, float[] b);
}
