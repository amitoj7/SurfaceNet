package io.surfaceguard.android;

import android.os.Build;
import android.os.PowerManager;
import android.os.SystemClock;
import java.util.Arrays;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;

final class EdgeBenchmark {
    private static final int WARMUPS = 16, RUNS = 120;
    private static final String SCOPE = "Warm single-ROI Module.forward on two bundled specimens";

    record Timing(double median, double p95, double[] samples) {
        JSONObject json() throws JSONException {
            JSONArray values = new JSONArray();
            for (double sample : samples) values.put(sample);
            return new JSONObject().put("p50_ms", median).put("p95_ms", p95).put("runs", RUNS)
                    .put("samples_ms", values);
        }
    }

    record Report(Timing single, Timing reflected, int thermalStart, int thermalEnd) {
        JSONObject json() throws JSONException {
            return new JSONObject()
                    .put("manufacturer", Build.MANUFACTURER).put("device", Build.MODEL)
                    .put("soc", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : JSONObject.NULL)
                    .put("sdk", Build.VERSION.SDK_INT).put("runtime", "ExecuTorch 1.5.0 / XNNPACK")
                    .put("scope", SCOPE).put("warmups", WARMUPS)
                    .put("thermal_status_start", thermalStart < 0 ? JSONObject.NULL : thermalStart)
                    .put("thermal_status_end", thermalEnd < 0 ? JSONObject.NULL : thermalEnd)
                    .put("single_view_int8", single.json()).put("four_view_int8", reflected.json());
        }
    }

    static Report run(Module single, Module reflected, EValue[] inputs, PowerManager power) {
        int start = temperatureStatus(power);
        for (int i = 0; i < WARMUPS; i++) {
            EValue input = inputs[i % inputs.length];
            single.forward(input);
            reflected.forward(input);
        }
        double[] a = new double[RUNS], b = new double[RUNS];
        for (int i = 0; i < RUNS; i++) {
            EValue input = inputs[(i / 2) % inputs.length];
            if (i % 2 == 0) {
                a[i] = time(single, input);
                b[i] = time(reflected, input);
            } else {
                b[i] = time(reflected, input);
                a[i] = time(single, input);
            }
        }
        return new Report(summarize(a), summarize(b), start, temperatureStatus(power));
    }

    private static int temperatureStatus(PowerManager power) {
        return power != null && Build.VERSION.SDK_INT >= 29 ? power.getCurrentThermalStatus() : -1;
    }

    private static double time(Module module, EValue input) {
        long start = SystemClock.elapsedRealtimeNanos();
        module.forward(input);
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0;
    }

    private static Timing summarize(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double median = (sorted[RUNS / 2 - 1] + sorted[RUNS / 2]) / 2;
        return new Timing(median, sorted[(int) Math.ceil(RUNS * .95) - 1], values);
    }
}
