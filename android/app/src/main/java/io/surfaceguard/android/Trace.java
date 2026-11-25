package io.surfaceguard.android;

import android.content.Context;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.Bitmap;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.UUID;

final class Trace extends SQLiteOpenHelper {
    static final String[] COLUMNS = {
        "id", "time", "part", "source", "decision", "kind", "severity", "score", "agreement",
        "vision_ms", "challenge", "advisory", "advisory_source", "advisory_ms", "vision_model"
    };

    Trace(Context context) { super(context, "inspections.db", null, 2); }

    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE inspections (" + String.join(" TEXT,", COLUMNS)
                + " TEXT, frame BLOB, mask BLOB, relit_frame BLOB, relit_mask BLOB, PRIMARY KEY(id))");
    }

    @Override public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        if (oldVersion == 1 && newVersion == 2) {
            db.execSQL("ALTER TABLE inspections ADD COLUMN vision_model TEXT NOT NULL DEFAULT 'SurfaceNet v1'");
            return;
        }
        throw new IllegalStateException("Unexpected trace schema upgrade");
    }

    void save(String part, String source, Inspection.Observation observation,
              Inspection.Decision decision, double ms, String challenge, Advice.Result advice, String visionModel,
              Bitmap frame, Bitmap mask, Bitmap relitFrame, Bitmap relitMask) {
        String[] values = {
            UUID.randomUUID().toString(), java.time.Instant.now().toString(), part, source,
            decision.state(), observation.kind(), observation.severity(),
            Double.toString(observation.score()), Integer.toString(decision.agreement()),
            Double.toString(ms), challenge, advice.text(), advice.source(), Long.toString(advice.elapsedMs()), visionModel
        };
        ContentValues row = new ContentValues();
        for (int i = 0; i < COLUMNS.length; i++) row.put(COLUMNS[i], values[i]);
        row.put("frame", png(frame));
        row.put("mask", png(mask));
        row.put("relit_frame", png(relitFrame));
        row.put("relit_mask", png(relitMask));
        getWritableDatabase().insertOrThrow("inspections", null, row);
    }

    Cursor latest() {
        return getReadableDatabase().query("inspections", COLUMNS, null, null, null, null, "time DESC", "50");
    }

    byte[][] evidence(String id) {
        try (Cursor cursor = getReadableDatabase().query("inspections", new String[]{"frame", "mask", "relit_frame", "relit_mask"},
                "id = ?", new String[]{id}, null, null, null)) {
            if (!cursor.moveToFirst()) return new byte[4][];
            return new byte[][] {cursor.getBlob(0), cursor.getBlob(1), cursor.getBlob(2), cursor.getBlob(3)};
        }
    }

    void export(OutputStream output) throws java.io.IOException {
        writeRow(output, COLUMNS);
        try (Cursor cursor = getReadableDatabase().query("inspections", COLUMNS, null, null, null, null, "time ASC")) {
            String[] values = new String[COLUMNS.length];
            while (cursor.moveToNext()) {
                for (int i = 0; i < values.length; i++) values[i] = cursor.getString(i);
                writeRow(output, values);
            }
        }
    }

    private static void writeRow(OutputStream output, String[] values) throws java.io.IOException {
        String[] escaped = Arrays.stream(values).map(value -> {
            String text = value == null ? "" : value;
            if (!text.isEmpty() && "=+-@\t\r\n".indexOf(text.charAt(0)) >= 0) text = "'" + text;
            return "\"" + text.replace("\"", "\"\"") + "\"";
        }).toArray(String[]::new);
        output.write((String.join(",", escaped) + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    private static byte[] png(Bitmap image) {
        if (image == null) return null;
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.PNG, 100, output);
        return output.toByteArray();
    }
}
