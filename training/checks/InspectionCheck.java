package io.surfaceguard.android;

import java.util.Arrays;

public final class InspectionCheck {
    private static final int W = Inspection.W, H = Inspection.H, N = Inspection.N;

    private static void check(boolean condition, String name) {
        if (!condition) throw new AssertionError(name);
    }

    private static Inspection.Frame texture(int shift, int bias, boolean changed) {
        byte[] data = new byte[N];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                double value = 110 + 28 * Math.sin((x + shift) * .18) + 14 * Math.cos(y * .12);
                int mark = changed && x >= 40 && x < 44 && y >= 101 && y < 119 ? 55 : 0;
                data[y * W + x] = (byte) (value + bias + mark);
            }
        }
        return new Inspection.Frame(data);
    }

    private static void checkMaskAgreement() {
        int[] down = new int[21], up = new int[21], shifted = new int[21];
        for (int i = 0; i < down.length; i++) {
            down[i] = (60 + i) * W + 30 + i;
            up[i] = (80 - i) * W + 30 + i;
            shifted[i] = (61 + i) * W + 32 + i;
        }
        var first = new Inspection.Region(30, 60, 50, 80, down, 10);
        var crossing = new Inspection.Region(30, 60, 50, 80, up, 10);
        var relocated = new Inspection.Region(32, 61, 52, 81, shifted, 10);
        var aligned = new Lighting.Pair(Lighting.Status.USABLE, 0, 0, .95, 22);
        var translation = new Lighting.Pair(Lighting.Status.USABLE, 2, 1, .95, 22);
        var disjoint = Lighting.agree(first, crossing, aligned);
        check(disjoint.boxOverlap() == 1 && !disjoint.confirmed(),
                "same bounding box cannot confirm crossing defects");
        check(Lighting.agree(first, relocated, translation).confirmed(),
                "registered thin defects preserve pixel support");
    }

    public static void main(String[] args) {
        checkMaskAgreement();
        var clean = texture(0, 0, false);
        Inspection.Frame[] samples = {clean, clean, clean, clean, clean, clean, clean, clean};
        var reference = Inspection.Reference.from(samples);
        float[] low = new float[N], high = new float[N];
        Arrays.fill(low, -10);
        Arrays.fill(high, -10);
        for (int y = 105; y < 123; y++) {
            for (int x = 40; x < 42; x++) high[y * W + x] = 10;
        }
        var cleanObservation = Inspection.observe(clean, low, .5f, reference);
        var mark = Inspection.observe(clean, high, .5f, reference);
        var novel = Inspection.observe(texture(2, 12, true), low, .5f, reference);
        var moved = Inspection.observe(texture(2, 12, false), low, .5f, reference);
        var black = Inspection.observe(new Inspection.Frame(new byte[N]), low, .5f, reference);

        check(cleanObservation.region() == null, "clean remains clean");
        check(moved.aligned() && moved.region() == null, "registration and exposure compensation");
        check(mark.kind().contains("Crack / scratch") && mark.severity().equals("High"),
                "learned mask geometry");
        check(novel.aligned() && novel.novel(), "unseen local change after alignment");
        var gate = new Inspection.Gate();
        for (int i = 0; i < 4; i++) check(gate.accept(cleanObservation, false, true, .30).state().equals("SETUP"),
                "no PASS without clean enrollment");
        gate.reset();
        for (int i = 0; i < 3; i++) check(gate.accept(cleanObservation, true, true, .30).state().equals("VERIFYING"),
                "wait for clear evidence");
        check(gate.accept(cleanObservation, true, true, .30).state().equals("PASS"), "four clear frames");
        check(gate.accept(mark, true, true, .30).state().equals("VERIFYING"), "one-frame anomaly cannot HOLD");
        check(gate.accept(cleanObservation, true, true, .30).state().equals("VERIFYING"), "no PASS after a conflicting hit");
        check(gate.accept(mark, true, true, .30).state().equals("VERIFYING"), "two alerts in window");
        check(gate.accept(mark, true, true, .30).state().equals("HOLD"), "three spatially consistent alerts");
        check(gate.accept(black, true, true, .30).state().equals("REACQUIRE"), "bad exposure invalidates window");
        check(gate.accept(mark, true, true, .30).state().equals("VERIFYING"), "reacquisition clears evidence");
        check(gate.accept(mark, false, false, .30).state().equals("REVIEW"), "still input remains operator review");
        gate.reset();
        var uncertain = new Inspection.Observation(.31, cleanObservation.quality(), true, null, null);
        check(gate.accept(uncertain, true, true, .30).state().equals("VERIFYING"), "wait for all four frames");
        for (int i = 0; i < 2; i++) gate.accept(cleanObservation, true, true, .30);
        check(gate.accept(cleanObservation, true, true, .30).state().equals("REVIEW"), "a high score blocks a clear window");
        check(gate.accept(cleanObservation, true, true, .30).state().equals("PASS"), "expired score no longer blocks a clear window");
        System.out.println("PASS: enrollment, registration, novelty, morphology, temporal evidence, review band, reacquisition, stills, aligned mask agreement");
    }
}
