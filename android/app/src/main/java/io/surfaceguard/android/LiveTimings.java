package io.surfaceguard.android;

import java.util.ArrayDeque;
import java.util.Arrays;

final class LiveTimings {
    private static final int LIMIT = 600;
    private static final long SECOND = 1_000_000_000L;
    private final ArrayDeque<Sample> samples = new ArrayDeque<>();

    record Sample(long receivedNs, long decisionNs, long sensorNs) {}
    record Summary(int frames, double spanSeconds, double decisionsPerSecond,
                   double analyzerP50Ms, double analyzerP95Ms, int sensorFrames,
                   Double sensorP50Ms, Double sensorP95Ms) {}

    synchronized void clear() { samples.clear(); }

    synchronized void record(long received, long decision, long sensor, boolean comparable) {
        if (decision < received) return;
        if (samples.size() == LIMIT) samples.removeFirst();
        // Avoid mixing different runs after the operator pauses or leaves the activity.
        if (!samples.isEmpty() && received - samples.getLast().decisionNs() > 2 * SECOND) samples.clear();
        long capture = comparable && sensor > 0 && sensor <= received && received - sensor < SECOND ? sensor : 0;
        samples.addLast(new Sample(received, decision, capture));
    }

    synchronized Summary summary() {
        int size = samples.size();
        if (size == 0) return new Summary(0,0,0,0,0,0,null,null);
        double[] analyzer = new double[size], camera = new double[size];
        int i = 0, count = 0;
        for (Sample sample : samples) {
            analyzer[i++] = (sample.decisionNs() - sample.receivedNs()) / 1_000_000.0;
            if (sample.sensorNs() != 0) camera[count++] = (sample.decisionNs() - sample.sensorNs()) / 1_000_000.0;
        }
        double span = (samples.getLast().decisionNs() - samples.getFirst().decisionNs()) / (double) SECOND;
        Arrays.sort(analyzer);
        Arrays.sort(camera,0,count);
        return new Summary(size,span,span > 0 ? (size - 1) / span : 0,
                percentile(analyzer,size,.5),percentile(analyzer,size,.95),count,
                count > 0 ? percentile(camera,count,.5) : null,count > 0 ? percentile(camera,count,.95) : null);
    }

    private static double percentile(double[] sorted, int count, double quantile) {
        return sorted[(int)Math.ceil(count * quantile) - 1];
    }
}
