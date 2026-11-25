package io.surfaceguard.android;

import java.io.BufferedInputStream;
import java.io.DataInputStream;

public final class PairAudit {
    public static void main(String[] args) throws Exception {
        try (var input = new DataInputStream(new BufferedInputStream(System.in))) {
            float threshold = input.readFloat();
            int count = input.readInt();
            for (int row = 0; row < count; row++) {
                var ambient = new Inspection.Frame(input.readNBytes(Inspection.N));
                var relit = new Inspection.Frame(input.readNBytes(Inspection.N));
                var pair = Lighting.measure(ambient, relit);
                var a = readObservation(input, ambient, threshold);
                var b = readObservation(input, relit, threshold);
                var agreement = Lighting.agree(a.learned(), b.learned(), pair);
                System.out.printf(java.util.Locale.US, "%.6f,%.6f,%b,%b,%s,%d,%d,%.5f,%d,%.5f,%.5f,%.5f,%b,%.5f%n",
                        a.score(), b.score(), a.learned() != null, b.learned() != null,
                        pair.status(), pair.dx(), pair.dy(), pair.correlation(), pair.lightChange(), agreement.boxOverlap(),
                        agreement.ambientSupport(), agreement.relitSupport(), agreement.confirmed(),
                        a.learned() == null ? 0 : a.learned().elongation());
            }
        }
    }

    private static Inspection.Observation readObservation(DataInputStream input, Inspection.Frame frame,
                                                           float threshold) throws Exception {
        float[] logits = new float[Inspection.N];
        for (int i = 0; i < logits.length; i++) logits[i] = input.readFloat();
        return Inspection.observe(frame, logits, threshold, null);
    }
}
