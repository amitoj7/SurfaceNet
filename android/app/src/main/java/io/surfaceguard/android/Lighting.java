package io.surfaceguard.android;

import java.util.Arrays;
import java.util.Locale;

/** Register the two exposures before deciding whether they provide independent evidence. */
final class Lighting {
    static final int TRAVEL = 4;
    static final double MIN_CORRELATION = .58;
    static final int MIN_LIGHT_CHANGE = 7;
    static final double MIN_OVERLAP = .25;
    static final double MIN_MASK_SUPPORT = .60;

    record Agreement(double boxOverlap, double ambientSupport, double relitSupport) {
        boolean confirmed() {
            return boxOverlap >= MIN_OVERLAP && Math.min(ambientSupport, relitSupport) >= MIN_MASK_SUPPORT;
        }

        String evidence() {
            return String.format(Locale.US, "mask support %.0f%% / %.0f%%", ambientSupport * 100, relitSupport * 100);
        }
    }

    static Agreement agree(Inspection.Region base, Inspection.Region relit, Pair pair) {
        if (base == null || relit == null) return new Agreement(0, 0, 0);
        return new Agreement(overlap(base, relit, pair),
                support(base.pixels(), relit.pixels(), pair.dx(), pair.dy()),
                support(relit.pixels(), base.pixels(), -pair.dx(), -pair.dy()));
    }

    private static double support(int[] source, int[] target, int dx, int dy) {
        boolean[] dilated = new boolean[Inspection.N];
        int width = Inspection.W, height = Inspection.H;
        for (int index : target) {
            int x = index % width, y = index / width;
            for (int yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
                for (int xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
                    dilated[yy * width + xx] = true;
                }
            }
        }
        int matched = 0;
        for (int index : source) {
            int x = index % width + dx, y = index / width + dy;
            if (x >= 0 && x < width && y >= 0 && y < height && dilated[y * width + x]) matched++;
        }
        return (double) matched / source.length;
    }

    enum Status { USABLE, CAPTURE_QUALITY, ALIGNMENT, INSUFFICIENT_LIGHT }

    record Pair(Status status, int dx, int dy, double correlation, int lightChange) {
        String evidence() {
            return String.format(Locale.US, "registration %.2f · offset %+d,%+d px · light Δ%d",
                    correlation, dx, dy, lightChange);
        }
    }

    static Pair measure(Inspection.Frame base, Inspection.Frame relit) {
        if (!base.quality().problem().isEmpty() || !relit.quality().problem().isEmpty())
            return new Pair(Status.CAPTURE_QUALITY, 0, 0, 0, 0);
        int width = Inspection.W, height = Inspection.H;
        int[] a = detail(base), b = detail(relit);
        double best = -2, correlation = 0;
        int bestX = 0, bestY = 0;
        for (int dy = -TRAVEL; dy <= TRAVEL; dy++) {
            for (int dx = -TRAVEL; dx <= TRAVEL; dx++) {
                long ab = 0, aa = 0, bb = 0;
                int count = 0;
                for (int y = 8; y < height - 8; y += 4) {
                    for (int x = 8; x < width - 8; x += 3) {
                        int i = y * width + x, j = (y + dy) * width + x + dx;
                        int av = a[i], bv = b[j];
                        ab += av * bv;
                        aa += av * av;
                        bb += bv * bv;
                        count++;
                    }
                }
                double value = Math.min(aa, bb) < 36L * count ? 0 : ab / Math.sqrt((double) aa * bb);
                double score = value - .002 * (Math.abs(dx) + Math.abs(dy));
                if (score > best) {
                    best = score;
                    correlation = value;
                    bestX = dx;
                    bestY = dy;
                }
            }
        }
        int[] changes = new int[Inspection.N / 10];
        int count = 0;
        for (int y = 8; y < height - 8; y += 4) {
            for (int x = 8; x < width - 8; x += 3) {
                int i = y * width + x, j = (y + bestY) * width + x + bestX;
                changes[count++] = Math.abs(relit.at(j) - base.at(i));
            }
        }
        Arrays.sort(changes, 0, count);
        int lightChange = changes[(count - 1) * 3 / 4];
        Status status = Status.USABLE;
        if (correlation < MIN_CORRELATION || Math.abs(bestX) == TRAVEL || Math.abs(bestY) == TRAVEL) {
            status = Status.ALIGNMENT;
        } else if (lightChange < MIN_LIGHT_CHANGE) {
            status = Status.INSUFFICIENT_LIGHT;
        }
        return new Pair(status, bestX, bestY, correlation, lightChange);
    }

    private static int[] detail(Inspection.Frame frame) {
        int w = Inspection.W;
        int[] response = new int[Inspection.N];
        for (int y = 2; y < Inspection.H - 2; y++) {
            for (int x = 2; x < w - 2; x++) {
                int i = y * w + x;
                int contrast = 4 * frame.at(i) - frame.at(i - 2) - frame.at(i + 2)
                        - frame.at(i - 2 * w) - frame.at(i + 2 * w);
                response[i] = Math.max(-72, Math.min(72, contrast));
            }
        }
        return response;
    }

    static double overlap(Inspection.Region base, Inspection.Region relit, Pair pair) {
        if (base == null || relit == null) return 0;
        int shiftedX = base.minX() + pair.dx(), shiftedY = base.minY() + pair.dy();
        int width = Math.max(0, Math.min(base.maxX() + pair.dx(), relit.maxX())
                - Math.max(shiftedX, relit.minX()) + 1);
        int height = Math.max(0, Math.min(base.maxY() + pair.dy(), relit.maxY())
                - Math.max(shiftedY, relit.minY()) + 1);
        int baseArea = (base.maxX() - base.minX() + 1) * (base.maxY() - base.minY() + 1);
        int relitArea = (relit.maxX() - relit.minX() + 1) * (relit.maxY() - relit.minY() + 1);
        return (double) (width * height) / (baseArea + relitArea - width * height);
    }
}
