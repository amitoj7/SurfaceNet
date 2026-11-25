package io.surfaceguard.android;

import java.util.ArrayDeque;
import java.util.Arrays;

/** Decision logic is independent of the Android camera and the advisory model. */
final class Inspection {
    static final int W = 96, H = 256, N = W * H;
    static final int ENROLL_COUNT = 8;
    static final int CLEAR_FRAMES = 4, ALERT_FRAMES = 3, WINDOW = 4;

    record Frame(byte[] gray) {
        int at(int index) { return gray[index] & 255; }
        Quality quality() {
            double total = 0, edges = 0;
            int clipped = 0;
            for (int y = 1; y < H; y++) {
                for (int x = 1; x < W; x++) {
                    int index = y * W + x, value = at(index);
                    total += value;
                    edges += Math.abs(value - at(index - 1)) + Math.abs(value - at(index - W));
                    if (value <= 9 || value >= 248) clipped++;
                }
            }
            int count = (W - 1) * (H - 1);
            return new Quality(total / count, (double) clipped / count, edges / (2 * count));
        }
    }

    record Quality(double luma, double clipped, double detail) {
        String problem() {
            if (luma < 24 || luma > 234 || clipped > .24) return "Adjust light and remove glare.";
            if (detail < .65) return "Focus on the surface and stabilize the phone.";
            return "";
        }
    }

    record Region(int minX, int minY, int maxX, int maxY, int[] pixels, double elongation) {
        double overlap(Region other) {
            int iw = Math.max(0, Math.min(maxX, other.maxX) - Math.max(minX, other.minX) + 1);
            int ih = Math.max(0, Math.min(maxY, other.maxY) - Math.max(minY, other.minY) + 1);
            double area = (maxX - minX + 1) * (maxY - minY + 1);
            double otherArea = (other.maxX - other.minX + 1) * (other.maxY - other.minY + 1);
            return iw * ih / (area + otherArea - iw * ih);
        }
    }

    record Reference(float[] mean, float[] variance) {
        static Reference from(Frame[] frames) {
            float[] mean = new float[N], variance = new float[N];
            for (int n = 0; n < frames.length; n++) {
                for (int i = 0; i < N; i++) {
                    float delta = frames[n].at(i) - mean[i];
                    mean[i] += delta / (n + 1);
                    variance[i] += delta * (frames[n].at(i) - mean[i]);
                }
            }
            for (int i = 0; i < N; i++) variance[i] /= frames.length - 1;
            return new Reference(mean, variance);
        }

        Comparison compare(Frame frame) {
            double best = Double.POSITIVE_INFINITY;
            int bestX = 0, bestY = 0;
            double bestBias = 0;
            for (int dy = -4; dy <= 4; dy += 2) {
                for (int dx = -4; dx <= 4; dx += 2) {
                    double bias = 0, error = 0;
                    int count = 0;
                    for (int y = 8; y < H - 8; y += 6) {
                        for (int x = 8; x < W - 8; x += 4) {
                            double offset = frame.at(y * W + x) - mean[(y + dy) * W + x + dx];
                            bias += Math.max(-45, Math.min(45, offset));
                            count++;
                        }
                    }
                    bias /= count;
                    for (int y = 8; y < H - 8; y += 6) {
                        for (int x = 8; x < W - 8; x += 4) {
                            double offset = frame.at(y * W + x) - mean[(y + dy) * W + x + dx];
                            error += Math.min(35, Math.abs(offset - bias));
                        }
                    }
                    double cost = error / count + .03 * (Math.abs(dx) + Math.abs(dy));
                    if (cost < best) {
                        best = cost;
                        bestX = dx;
                        bestY = dy;
                        bestBias = bias;
                    }
                }
            }
            if (best > 17 || Math.abs(bestBias) > 30) return new Comparison(false, null);
            boolean[] mask = new boolean[N];
            for (int y = 6; y < H - 6; y++) {
                for (int x = 6; x < W - 6; x++) {
                    int i = y * W + x, j = (y + bestY) * W + x + bestX;
                    double residual = Math.abs(frame.at(i) - mean[j] - bestBias);
                    double gradient = Math.max(Math.abs(mean[j + 1] - mean[j - 1]),
                                               Math.abs(mean[j + W] - mean[j - W]));
                    double tolerance = 13 + 3 * Math.sqrt(variance[j]) + .45 * gradient;
                    mask[i] = residual > tolerance;
                }
            }
            return new Comparison(true, component(mask, 12, -1));
        }
    }

    record Comparison(boolean aligned, Region anomaly) {}

    record Observation(double score, Quality quality, boolean aligned, Region learned, Region novelty) {
        Region region() {
            if (learned != null && novelty != null && learned.overlap(novelty) < .12) return novelty;
            return learned == null ? novelty : learned;
        }
        boolean novel() { return region() != null && region() == novelty; }
        String kind() {
            Region region = region();
            if (region == null) return "No localized finding";
            if (novel()) return "Unknown surface change";
            if (region.elongation >= 3.2) return "Crack / scratch-like line";
            double boxArea = (region.maxX - region.minX + 1) * (region.maxY - region.minY + 1);
            return region.pixels.length / boxArea > .48 ? "Compact mark / pit-like spot" : "Diffuse surface mark";
        }
        String severity() {
            Region region = region();
            if (region == null) return "—";
            if (novel()) return "Unrated";
            double fraction = (double) region.pixels.length / N;
            return fraction >= .012 || region.elongation >= 4.5 ? "High" : fraction >= .002 ? "Moderate" : "Low";
        }
    }

    static Observation observe(Frame frame, float[] logits, float threshold, Reference reference) {
        Quality quality = frame.quality();
        float[] probabilities = new float[N];
        for (int i = 0; i < N; i++) probabilities[i] = (float) (1 / (1 + Math.exp(-logits[i])));
        float score = 0;
        int peak = -1;
        for (int y = 0; y < H - 1; y++) {
            for (int x = 0; x < W - 1; x++) {
                int i = y * W + x;
                float value = (probabilities[i] + probabilities[i + 1]
                        + probabilities[i + W] + probabilities[i + W + 1]) * .25f;
                if (value > score) {
                    score = value;
                    peak = i;
                }
            }
        }
        Region learned = null;
        if (score > threshold) {
            boolean[] mask = new boolean[N];
            for (int i = 0; i < N; i++) mask[i] = probabilities[i] > .85f * threshold;
            learned = component(mask, 1, peak);
        }
        Comparison comparison = reference == null ? new Comparison(true, null) : reference.compare(frame);
        return new Observation(score, quality, comparison.aligned, learned, comparison.anomaly);
    }

    private static Region component(boolean[] active, int minimum, int peak) {
        int[] queue = new int[N];
        Region largest = null;
        for (int first = 0; first < N; first++) {
            if (!active[first]) continue;
            active[first] = false;
            int size = 1, minX = W, minY = H, maxX = 0, maxY = 0;
            boolean coversPeak = false;
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            queue[0] = first;
            for (int head = 0; head < size; head++) {
                int i = queue[head], x = i % W, y = i / W;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
                sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                coversPeak |= peak >= 0 && x >= peak % W && x <= peak % W + 1
                        && y >= peak / W && y <= peak / W + 1;
                for (int yy = Math.max(0, y - 1); yy <= Math.min(H - 1, y + 1); yy++) {
                    for (int xx = Math.max(0, x - 1); xx <= Math.min(W - 1, x + 1); xx++) {
                        int neighbor = yy * W + xx;
                        if (active[neighbor]) {
                            active[neighbor] = false;
                            queue[size++] = neighbor;
                        }
                    }
                }
            }
            if (size < minimum || (peak >= 0 && !coversPeak)
                    || (largest != null && size <= largest.pixels.length)) continue;
            double vx = sxx / size - Math.pow(sx / size, 2);
            double vy = syy / size - Math.pow(sy / size, 2);
            double covariance = sxy / size - (sx / size) * (sy / size);
            double radius = Math.hypot(vx - vy, 2 * covariance);
            double elongation = Math.sqrt((vx + vy + radius + .5) / (vx + vy - radius + .5));
            largest = new Region(minX, minY, maxX, maxY, Arrays.copyOf(queue, size), elongation);
        }
        return largest;
    }

    record Decision(String state, String reason, int agreement) {
        boolean recordable() { return state.equals("PASS") || state.equals("HOLD") || state.equals("REVIEW"); }
    }

    static final class Gate {
        private final ArrayDeque<Observation> history = new ArrayDeque<>();

        void reset() { history.clear(); }

        Decision accept(Observation current, boolean enrolled, boolean live, double clearBelow) {
            String problem = current.quality.problem();
            if (!problem.isEmpty() || !current.aligned) {
                reset();
                return new Decision("REACQUIRE", problem.isEmpty()
                        ? "Restore fixture alignment or the enrolled illumination." : problem, 0);
            }
            if (!live) {
                return new Decision("REVIEW", current.region() == null
                        ? "No localized finding in this still. Confirm on the fixture."
                        : "Inspect the outlined region and confirm on the fixture.", 1);
            }
            if (history.size() == WINDOW) history.removeFirst();
            history.addLast(current);
            Region region = current.region();
            if (region != null) {
                int matches = 0;
                for (Observation prior : history) {
                    if (prior.region() != null && prior.novel() == current.novel()
                            && region.overlap(prior.region()) >= .25) matches++;
                }
                if (matches >= ALERT_FRAMES)
                    return new Decision("HOLD", current.novel()
                            ? "Segregate the part and ask quality to identify the change."
                            : "Segregate the part and verify the outlined defect.", matches);
                return new Decision("VERIFYING", "Keep the finding in view for repeat confirmation.", matches);
            }
            int clear = 0;
            for (Observation prior : history) if (prior.region() == null) clear++;
            if (!enrolled) return new Decision("SETUP", "Enroll a verified clean part in the fixture.", clear);
            if (clear >= CLEAR_FRAMES) {
                for (Observation prior : history) {
                    if (prior.score() >= clearBelow)
                        return new Decision("REVIEW", "Surface score is in the review range. Check the part under a second light.", clear);
                }
                return new Decision("PASS", "All frames scored below the clear threshold.", clear);
            }
            return new Decision("VERIFYING", "Keep the part still while the station gathers clear frames.", clear);
        }
    }
}
