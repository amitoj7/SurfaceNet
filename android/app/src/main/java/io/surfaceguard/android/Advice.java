package io.surfaceguard.android;

import android.content.Context;
import android.net.Uri;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Locale;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmModule;
import org.pytorch.executorch.extension.llm.LlmModuleConfig;

final class Advice {
    private static final String MODEL_SHA = "8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b";
    private static final String TOKENIZER_SHA = "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55";
    private static final int MAX_SEQUENCE_LENGTH = 256, MAX_RESPONSE_TOKENS = 12;
    private final Context context;
    private LlmModule runner;

    record Result(String text, String source, long elapsedMs) {}

    Advice(Context context) { this.context = context; }

    boolean configured() { return asset(false).isFile() && asset(true).isFile(); }

    private File asset(boolean tokenizer) {
        return new File(context.getFilesDir(), tokenizer ? "tokenizer.model" : "llama.pte");
    }

    void importFile(Uri source, boolean tokenizer) throws Exception {
        File destination = asset(tokenizer);
        File temporary = new File(context.getCacheDir(), destination.getName() + ".import");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try {
            try (InputStream input = context.getContentResolver().openInputStream(source);
                 FileOutputStream output = new FileOutputStream(temporary)) {
                if (input == null) throw new java.io.IOException("Selected file could not be opened.");
                Files.copy(new DigestInputStream(input, digest), output);
            }
            StringBuilder actual = new StringBuilder(64);
            for (byte value : digest.digest()) actual.append(String.format(Locale.US, "%02x", value & 255));
            if (!actual.toString().equals(tokenizer ? TOKENIZER_SHA : MODEL_SHA)) {
                throw new java.io.IOException("This file does not match the pinned Llama 3.2 export.");
            }
            if (!temporary.renameTo(destination)) throw new java.io.IOException("Could not store model file.");
            close();
        } finally {
            if (temporary.exists()) temporary.delete();
        }
    }

    static Result procedure(Inspection.Observation observation, Inspection.Decision decision, String challenge) {
        return new Result(Procedure.choices(observation, decision, challenge).a(), "Station procedure", 0);
    }

    Result generate(Inspection.Observation observation, Inspection.Decision decision, String challenge) {
        if (!configured()) return procedure(observation, decision, challenge);
        long start = android.os.SystemClock.elapsedRealtime();
        Procedure.Choices choices = Procedure.choices(observation, decision, challenge);
        String prompt = Procedure.prompt(observation, decision, challenge, choices);
        try {
            if (runner == null) {
                LlmModule candidate = new LlmModule(LlmModuleConfig.create()
                        .modulePath(asset(false).getAbsolutePath())
                        .tokenizerPath(asset(true).getAbsolutePath())
                        .temperature(0f)
                        .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
                        .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
                        .build());
                runner = candidate;
                runner.load();
            }
            LlmModule active = runner;
            ProcedureResponse response = new ProcedureResponse(active);
            active.resetContext();
            // This Android AAR forwards seqLen to JNI. Stop decoding as soon as the choice is complete.
            active.generate(prompt, LlmGenerationConfig.create().seqLen(MAX_SEQUENCE_LENGTH)
                    .echo(false).temperature(0f).numBos(1).build(), response);
            if (!response.failed && !response.choice.isEmpty()) {
                return new Result(response.choice.equals("A") ? choices.a() : choices.b(),
                        "Llama 3.2 1B · ExecuTorch / XNNPACK",
                        android.os.SystemClock.elapsedRealtime() - start);
            }
        } catch (RuntimeException | LinkageError failure) {
            android.util.Log.w("SurfaceNet", "Local Llama inference failed", failure);
            close();
        }
        return failed(observation, decision, challenge, start);
    }

    private static final class ProcedureResponse implements LlmCallback {
        private final LlmModule runner;
        private final StringBuilder output = new StringBuilder();
        private String choice = "";
        private int tokens;
        private boolean failed;

        ProcedureResponse(LlmModule runner) { this.runner = runner; }

        @Override public void onResult(String token) {
            if (!choice.isEmpty() || tokens >= MAX_RESPONSE_TOKENS) return;
            output.append(token);
            tokens++;
            choice = Procedure.selection(output.toString());
            if (!choice.isEmpty() || tokens >= MAX_RESPONSE_TOKENS) runner.stop();
        }

        @Override public void onError(int errorCode, String message) { failed = true; }
    }

    private static Result failed(Inspection.Observation observation, Inspection.Decision decision, String challenge, long start) {
        return new Result(procedure(observation, decision, challenge).text,
                "Station procedure · Llama unavailable", android.os.SystemClock.elapsedRealtime() - start);
    }

    void close() {
        if (runner != null) {
            runner.close();
            runner = null;
        }
    }
}
