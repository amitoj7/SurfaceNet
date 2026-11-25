package io.surfaceguard.android;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.TypefaceSpan;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.activity.ComponentActivity;
import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.camera.core.Camera;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONObject;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import static io.surfaceguard.android.StationStyle.*;

public final class MainActivity extends ComponentActivity {
    private final ExecutorService visionWorker = Executors.newSingleThreadExecutor();
    private final ExecutorService adviceWorker = Executors.newSingleThreadExecutor();
    private final ExecutorService traceWorker = Executors.newSingleThreadExecutor();
    private final Inspection.Gate gate = new Inspection.Gate();
    private final AtomicBoolean uiPosted = new AtomicBoolean();
    private final Inspection.Frame[] enrollment = new Inspection.Frame[Inspection.ENROLL_COUNT];
    private final double[] timings = new double[120];

    private Module model;
    private float threshold;
    private double clearBelow;
    private String visionName;
    private boolean loadedReflection;
    private volatile boolean useReflection;
    private Inspection.Reference reference;
    private int enrolledCount, timingIndex, timingCount;
    private long lastAnalyzed;
    private final LiveTimings liveTimings = new LiveTimings();
    private volatile boolean sensorRealtime;
    private String liveTimingProfile;
    private volatile int generation;
    private volatile boolean cameraLive, frozen, closed, benchmarking;
    private volatile Challenge challenge;
    private ProcessCameraProvider provider;
    private Camera camera;
    private Advice advisor;
    private Trace trace;

    private LinearLayout root, inspectScreen, traceScreen, traceList, traceDetail, deviceScreen, entries;
    private LinearLayout inspectHeading, sources, controls, inspectionResult, workstation, pairControls;
    private final Button[] navigation = new Button[3];
    private final int[] scrollPositions = new int[3];
    private final OnBackPressedCallback sectionBack = new OnBackPressedCallback(false) {
        @Override public void handleOnBackPressed() {
            if (activeSection == 1 && traceDetail.getVisibility() == View.VISIBLE) closeTraceDetail();
            else showSection(0, true);
        }
    };
    private boolean traceLoading;
    private int traceListScroll;
    private ScrollView scroll;
    private int activeSection;
    private Typeface display, editorial;
    private Bitmap specimenArt;
    private boolean maskVisible = true;
    private EvidenceView preview;
    private Button liveButton, enrollButton, freezeButton, challengeButton, adviseButton, saveButton, profileButton;
    private Button importButton, sampleButton, replayButton, maskToggle, benchmarkButton, benchmarkExport;
    private TextView state, finding, evidence, reason, timing, advisory, note, inputLabel;
    private TextView inspectTitle, compareLabel, lightingReadout, deviceProfile, deviceModel, benchmarkResult;
    private LinearLayout benchmarkReadings;
    private TextView singleReading, reflectedReading;
    private EditText part;
    private Bitmap latestFrame, latestMask, relitFrame, relitMask;
    private Lighting.Pair relitAlignment;
    private Snapshot latest, locked;
    private String challengeResult = "Not run";
    private Advice.Result recordedAdvice;
    private boolean advising, importing, saving, saved;
    private String benchmarkJson;
    private String liveTimingsJson;
    private TextView liveTimingStatus;

    record Snapshot(Inspection.Observation observation, Inspection.Decision decision,
                    Inspection.Frame frame, String source, double latency, String performance, long time) {}
    record ChallengeCapture(Snapshot snapshot, Lighting.Pair measurement) {}
    private static final class Challenge {
        final Snapshot base;
        final long started = SystemClock.elapsedRealtime();
        int eligible, overlap;
        Snapshot example;
        Inspection.Region region;
        volatile ChallengeCapture last;
        Lighting.Pair examplePair;
        Lighting.Agreement exampleAgreement;
        Challenge(Snapshot base) { this.base = base; }
    }
    private static final String LIGHTING_REPLAY = "KSDD2 · LIGHTING REPLAY";

    private final ActivityResultLauncher<String> photoPicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), this::inspectImage);
    private final ActivityResultLauncher<String[]> modelPicker =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> importAsset(uri, false));
    private final ActivityResultLauncher<String[]> tokenizerPicker =
            registerForActivityResult(new ActivityResultContracts.OpenDocument(), uri -> importAsset(uri, true));
    private final ActivityResultLauncher<String> csvPicker =
            registerForActivityResult(new ActivityResultContracts.CreateDocument("text/csv"), this::exportTrace);
    private final ActivityResultLauncher<String> benchmarkPicker =
            registerForActivityResult(new ActivityResultContracts.CreateDocument("application/json"), this::exportBenchmark);
    private final ActivityResultLauncher<String> liveTimingPicker =
            registerForActivityResult(new ActivityResultContracts.CreateDocument("application/json"),
                    uri -> exportTimingJson(uri,liveTimingsJson,"Live timings"));
    private final ActivityResultLauncher<String> permission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) openCamera();
                else note("Camera access is required for live inspection.");
            });

    @Override public void onCreate(Bundle stateBundle) {
        super.onCreate(stateBundle);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        advisor = new Advice(getApplicationContext());
        trace = new Trace(getApplicationContext());
        buildUi();
        getOnBackPressedDispatcher().addCallback(this, sectionBack);
    }

    private void buildUi() {
        display = Typeface.createFromAsset(getAssets(), "type/barlow-condensed-semibold.ttf");
        editorial = Typeface.createFromAsset(getAssets(), "type/instrument-serif-italic.ttf");
        specimenArt = fixtureImage("replay-ambient.png");
        root = column();
        root.setBackgroundColor(BG);
        setContentView(root);

        LinearLayout header = row();
        header.setPadding(dp(14), 0, dp(18), 0);
        header.setMinimumHeight(dp(55));
        ImageView mark = new ImageView(this);
        mark.setImageResource(R.drawable.surfacenet_logo);
        mark.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        mark.setScaleType(ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams markLayout = new LinearLayout.LayoutParams(dp(34), dp(34));
        markLayout.setMarginEnd(dp(8));
        header.addView(mark, markLayout);
        TextView wordmark = label("SurfaceNet", 18, INK, true);
        wordmark.setLetterSpacing(-.045f);
        header.addView(wordmark, new LinearLayout.LayoutParams(0, -2, 1));
        header.addView(label("●  ON DEVICE", 10, SIGNAL, true));
        root.addView(header);
        line(root);

        note = label("", 12, INK, false);
        note.setBackgroundColor(0xff2a2b21);
        note.setPadding(dp(18), dp(5), dp(18), dp(5));
        note.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        note.setVisibility(View.GONE);
        root.addView(note);

        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setVerticalScrollBarEnabled(false);
        scroll.setClipToPadding(false);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        LinearLayout content = column();
        content.setPadding(dp(17), dp(12), dp(17), dp(20));
        scroll.addView(content);
        inspectScreen = column();
        traceScreen = column();
        deviceScreen = column();
        content.addView(inspectScreen);
        content.addView(traceScreen);
        content.addView(deviceScreen);
        buildInspect();
        buildTrace();
        buildDevice();

        line(root);
        LinearLayout dock = row();
        dock.setPadding(dp(10), dp(3), dp(10), dp(5));
        String[] pages = {"Inspect", "Trace log", "Device"};
        for (int i = 0; i < pages.length; i++) {
            int section = i;
            navigation[i] = button(pages[i], () -> showSection(section, true));
            navigation[i].setBackground(StationStyle.navigation(dp(5)));
            navigation[i].setTextSize(12);
            navigation[i].setCompoundDrawablePadding(dp(3));
            addHalf(dock, navigation[i]);
        }
        root.addView(dock);
        showSection(0, false);
        showSpecimen(true);
        buttons();
    }

    private void buildInspect() {
        inspectHeading = row();
        inspectHeading.addView(kicker("INSPECTION STATION"), new LinearLayout.LayoutParams(0, -2, 1));
        inspectScreen.addView(inspectHeading);
        inspectTitle = title("Inspect ", "surface.", 58);
        inspectScreen.addView(inspectTitle);

        preview = new EvidenceView(this);
        int height = Math.max(205, Math.min(320, (int) (getResources().getConfiguration().screenHeightDp * .38f)));
        LinearLayout.LayoutParams previewLayout = new LinearLayout.LayoutParams(-1, dp(height));
        previewLayout.topMargin = dp(10);
        inspectScreen.addView(preview, previewLayout);

        LinearLayout previewFooter = row();
        inputLabel = kicker("KSDD2  /  DEMONSTRATION");
        inputLabel.setTextColor(DIM);
        previewFooter.addView(inputLabel, new LinearLayout.LayoutParams(0, -2, 1));
        maskToggle = button("Hide mask", () -> toggleMask(preview, maskToggle));
        previewFooter.addView(maskToggle);
        inspectScreen.addView(previewFooter);
        pairControls = compareControls(preview, "A  Original", "B  Rear light");
        compareLabel = pairControls.findViewWithTag("alternate-caption");
        inspectScreen.addView(pairControls);
        pairControls.setVisibility(View.GONE);

        sources = column();
        LinearLayout captureActions = row();
        liveButton = button("Rear camera", this::requestCamera, true);
        importButton = button("Open image", () -> photoPicker.launch("image/*"));
        sampleButton = button("Specimens", this::specimens);
        replayButton = button("Lighting replay ↗", this::runLightingReplay);
        addHalf(captureActions, liveButton);
        addHalf(captureActions, importButton);
        sources.addView(captureActions);
        LinearLayout demoActions = row();
        addHalf(demoActions, sampleButton);
        addHalf(demoActions, replayButton);
        sources.addView(demoActions);
        inspectScreen.addView(sources);

        inspectionResult = card();
        inspectionResult.addView(kicker("DECISION"));
        state = label("", 34, SIGNAL, true);
        state.setTypeface(display);
        state.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        finding = label("", 16, INK, true);
        evidence = label("", 13, DIM, false);
        reason = label("", 13, DIM, false);
        lightingReadout = label("", 13, INK, false);
        timing = label("", 11, DIM, true);
        inspectionResult.addView(state);
        inspectionResult.addView(finding);
        inspectionResult.addView(evidence);
        inspectionResult.addView(reason);
        inspectionResult.addView(lightingReadout);
        inspectionResult.addView(timing);
        inspectScreen.addView(inspectionResult);

        controls = row();
        enrollButton = button("Enroll clean", this::enroll);
        freezeButton = button("Freeze result", this::freezeOrResume, true);
        addHalf(controls, enrollButton);
        addHalf(controls, freezeButton);
        inspectScreen.addView(controls);
        challengeButton = button("Verify with rear light", this::startChallenge, true);
        inspectScreen.addView(challengeButton);

        workstation = card();
        workstation.addView(kicker("OPERATOR ADVISORY"));
        advisory = label("", 16, INK, false);
        workstation.addView(advisory);
        adviseButton = button("Ask local Llama", this::generateAdvice);
        workstation.addView(adviseButton);
        part = new EditText(this);
        part.setTextColor(INK);
        part.setHintTextColor(DIM);
        part.setHint("Part ID or lot");
        part.setSingleLine(true);
        part.setTextSize(15);
        part.setMinHeight(dp(48));
        part.setContentDescription("Part ID or lot");
        part.setFilters(new android.text.InputFilter[] {new android.text.InputFilter.LengthFilter(80)});
        workstation.addView(part);
        saveButton = button("Save inspection", this::saveInspection, true);
        workstation.addView(saveButton);
        inspectScreen.addView(workstation);
    }

    private void buildTrace() {
        traceList = column();
        traceList.addView(kicker("STORED ON DEVICE"));
        traceList.addView(title("Trace ", "log.", 58));
        traceList.addView(button("Export CSV", () -> csvPicker.launch("surfacenet-inspections.csv")));
        entries = column();
        traceList.addView(entries);
        traceScreen.addView(traceList);
        traceDetail = column();
        traceDetail.setVisibility(View.GONE);
        traceScreen.addView(traceDetail);
    }

    private void buildDevice() {
        deviceScreen.addView(kicker("EXECUTORCH  /  ARM CPU"));
        deviceScreen.addView(title("Device", ".", 62));
        LinearLayout detector = card();
        detector.addView(kicker("VISION"));
        deviceProfile = label("", 23, INK, true);
        detector.addView(deviceProfile);
        detector.addView(label("INT8 · SurfaceNet v2", 13, DIM, false));
        profileButton = button("Switch profile", this::profileMenu);
        detector.addView(profileButton);
        deviceScreen.addView(detector);

        LinearLayout benchmark = card();
        benchmark.addView(kicker("DEVICE BENCHMARK"));
        benchmark.addView(label("Single view vs four views", 21, INK, true));
        benchmarkResult = label("Measure warmed forwards on this phone.", 14, DIM, false);
        benchmark.addView(benchmarkResult);
        benchmarkReadings = row();
        singleReading = timingColumn(benchmarkReadings, "SINGLE VIEW");
        reflectedReading = timingColumn(benchmarkReadings, "FOUR VIEWS");
        benchmarkReadings.setVisibility(View.GONE);
        benchmark.addView(benchmarkReadings);
        benchmarkButton = button("Run benchmark", this::runBenchmark, true);
        benchmark.addView(benchmarkButton);
        benchmarkExport = button("Export JSON", () -> benchmarkPicker.launch("surfacenet-device-benchmark.json"));
        benchmarkExport.setVisibility(View.GONE);
        benchmark.addView(benchmarkExport);
        deviceScreen.addView(benchmark);

        LinearLayout cameraTiming = card();
        cameraTiming.addView(kicker("LIVE CAMERA TIMING"));
        liveTimingStatus = label("Keep the rear camera running, then export the timing sample.",14,DIM,false);
        cameraTiming.addView(liveTimingStatus);
        cameraTiming.addView(button("Export live timings",this::prepareLiveTimings));
        deviceScreen.addView(cameraTiming);

        LinearLayout modelFiles = card();
        modelFiles.addView(kicker("LOCAL ADVISORY"));
        modelFiles.addView(label("Llama 3.2 1B", 21, INK, true));
        deviceModel = label("", 14, DIM, false);
        modelFiles.addView(deviceModel);
        refreshModelStatus();
        modelFiles.addView(button("Manage model files", this::modelMenu));
        deviceScreen.addView(modelFiles);
    }

    private TextView timingColumn(LinearLayout readings, String title) {
        LinearLayout stat = column();
        stat.setPadding(0, dp(7), dp(6), dp(7));
        stat.addView(kicker(title));
        TextView value = label("", 28, INK, false);
        value.setTypeface(display);
        stat.addView(value);
        readings.addView(stat, new LinearLayout.LayoutParams(0, -2, 1));
        return value;
    }

    private void showSection(int section, boolean animate) {
        if (animate && activeSection == section) return;
        if (animate) scrollPositions[activeSection] = scroll.getScrollY();
        activeSection = section;
        sectionBack.setEnabled(section != 0);
        LinearLayout[] screens = {inspectScreen, traceScreen, deviceScreen};
        for (int i = 0; i < screens.length; i++) {
            screens[i].setVisibility(i == section ? View.VISIBLE : View.GONE);
            navigation[i].setSelected(i == section);
            navigation[i].setCompoundDrawablesWithIntrinsicBounds(null,
                    new StationStyle.DockIcon(i, dp(21), i == section), null, null);
        }
        if (section == 1) refreshTrace();
        if (section == 2) showLiveTimings(liveTimings.summary());
        scroll.post(() -> {
            if (activeSection != section) return;
            scroll.scrollTo(0, scrollPositions[section]);
            if (animate) StationStyle.enter(screens[section], dp(7));
        });
    }

    private void showSpecimen(boolean animate) {
        inputLabel.setText("KSDD2  /  DEMONSTRATION");
        preview.specimen(specimenArt, animate);
        maskToggle.setVisibility(View.GONE);
        pairControls.setVisibility(View.GONE);
    }

    private LinearLayout compareControls(EvidenceView image, String original, String alternate) {
        LinearLayout controls = column();
        LinearLayout captions = row();
        TextView first = label(original, 12, DIM, false);
        captions.addView(first, new LinearLayout.LayoutParams(0, -2, 1));
        TextView second = label(alternate, 12, SIGNAL, false);
        second.setTag("alternate-caption");
        captions.addView(second);
        controls.addView(captions);
        SeekBar split = new SeekBar(this);
        split.setMax(100);
        split.setProgress(50);
        split.setContentDescription("Exposure split. Higher values show more of the original capture.");
        split.setMinHeight(dp(48));
        split.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar bar, int value, boolean byTouch) { image.setSplit(value); }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        image.setSplitListener(split::setProgress);
        controls.addView(split);
        return controls;
    }

    private void requestCamera() {
        if (cameraLive) {
            resetInput();
            note("Rear camera stopped.");
            return;
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permission.launch(Manifest.permission.CAMERA);
        } else openCamera();
    }

    @androidx.annotation.OptIn(markerClass = ExperimentalCamera2Interop.class)
    private void openCamera() {
        resetInput();
        liveTimings.clear();
        liveTimingProfile = useReflection ? "Four views INT8" : "Single view INT8";
        final int session = generation;
        note("Hold the phone fixed. Fill the outlined region with the part.");
        preview.render(null, null, null, SIGNAL);
        inputLabel.setText("REAR CAMERA  /  PREPARING");
        var providerFuture = ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            if (generation != session || closed) return;
            try {
                provider = providerFuture.get();
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setTargetRotation(getWindowManager().getDefaultDisplay().getRotation())
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
                analysis.setAnalyzer(visionWorker, image -> analyzeCamera(image,session));
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis);
                sensorRealtime = Integer.valueOf(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
                        .equals(Camera2CameraInfo.from(camera.getCameraInfo())
                                .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE));
                cameraLive = true;
                inputLabel.setText("REAR CAMERA  /  LIVE");
                buttons();
            } catch (Exception failure) {
                resetInput();
                note("Rear camera unavailable. " + failure.getClass().getSimpleName());
            }
        }, getMainExecutor());
        preview.postDelayed(() -> {
            if (session == generation && !closed && latest == null) {
                resetInput();
                note("No camera frames arrived. Close other apps using the camera, then reopen Rear camera.");
            }
        },8000);
    }

    private void analyzeCamera(ImageProxy image, int session) {
        long receivedNs = SystemClock.elapsedRealtimeNanos();
        try {
            long now = receivedNs / 1_000_000L;
            if (session != generation || frozen || !cameraLive || benchmarking || now - lastAnalyzed < 100) return;
            if (now - lastAnalyzed > 1_100) {
                gate.reset();
                if (enrolling) enrolledCount = 0;
            }
            lastAnalyzed = now;
            Inspection.Frame frame = Frames.camera(image);
            Challenge active = challenge;
            Snapshot result = analyze(frame, "REAR CAMERA", true);
            if (active == null && session == generation && !frozen) {
                liveTimings.record(receivedNs,SystemClock.elapsedRealtimeNanos(),image.getImageInfo().getTimestamp(),sensorRealtime);
            }
            if (active != null && active == challenge) {
                analyzeChallenge(active, result, session);
            } else if (uiPosted.compareAndSet(false, true)) {
                runOnUiThread(() -> {
                    uiPosted.set(false);
                    if (session == generation && !closed && !frozen && challenge == null) showSnapshot(result);
                });
            }
        } catch (Exception | LinkageError failure) {
            showVisionError(failure, session);
        } finally {
            image.close();
        }
    }

    private void ensureModel() throws Exception {
        boolean reflected = useReflection;
        if (model != null) {
            if (loadedReflection == reflected) return;
            model.destroy();
            model = null;
        }
        try (InputStream metadata = getAssets().open("vision.json")) {
            JSONObject spec = new JSONObject(Files.text(metadata));
            try (InputStream edge = getAssets().open(reflected ? "edge-reflection.json" : "edge-local.json")) {
                JSONObject calibration = new JSONObject(Files.text(edge));
                threshold = (float) calibration.getDouble("threshold");
                clearBelow = calibration.getDouble("clear_below");
                visionName = "SurfaceNet " + spec.optString("architecture", "v2") + " · "
                        + (reflected ? "Four views" : "Single view") + " · " + calibration.getString("precision");
            }
            if (spec.getInt("width") != Inspection.W || spec.getInt("height") != Inspection.H
                    || !Float.isFinite(threshold) || threshold <= 0 || threshold >= 1
                    || !Double.isFinite(clearBelow) || clearBelow < 0 || clearBelow > threshold) {
                throw new IllegalStateException("Invalid vision metadata");
            }
        }
        model = openVision(reflected ? "surface-reflection-int8.pte" : "surface-int8.pte");
        loadedReflection = reflected;
    }

    private Module openVision(String name) throws Exception {
        File cached = new File(getCacheDir(), "vision-04-" + name);
        if (!cached.isFile()) {
            File staging = File.createTempFile("vision", ".pte", getCacheDir());
            try {
                try (InputStream asset = getAssets().open(name);
                     OutputStream output = new FileOutputStream(staging)) {
                    Files.copy(asset, output);
                }
                if (!staging.renameTo(cached) && !cached.isFile())
                    throw new IllegalStateException("Could not cache model");
            } finally {
                staging.delete();
            }
        }
        // The pinned Android JNI treats zero as auto and skips resizing for a negative count.
        return Module.load(cached.getAbsolutePath(), Module.LOAD_MODE_MMAP, -1);
    }

    private static EValue tensor(Inspection.Frame frame) {
        float[] input = new float[Inspection.N];
        for (int i = 0; i < input.length; i++) input[i] = frame.at(i) / 127.5f - 1f;
        return EValue.from(Tensor.fromBlob(input, new long[]{1, 1, Inspection.H, Inspection.W}));
    }

    private Snapshot analyze(Inspection.Frame frame, String source, boolean live) throws Exception {
        ensureModel();
        long start = SystemClock.elapsedRealtimeNanos();
        EValue tensor = tensor(frame);
        float[] logits = model.forward(tensor)[0].toTensor().getDataAsFloatArray();
        if (logits.length != Inspection.N || !Float.isFinite(logits[0])) {
            throw new IllegalStateException("Unexpected vision output");
        }
        Inspection.Observation observation = Inspection.observe(frame, logits, threshold, live ? reference : null);
        Inspection.Decision result = gate.accept(observation, reference != null, live, clearBelow);
        if (live && enrolling && challenge == null) {
            result = enrollFrame(frame, observation);
        }
        double ms = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0;
        timings[timingIndex++ % timings.length] = ms;
        timingCount = Math.min(timings.length, timingCount + 1);
        double[] sorted = java.util.Arrays.copyOf(timings, timingCount);
        java.util.Arrays.sort(sorted);
        String performance = String.format(Locale.US, "p50 %.1f ms  ·  p95 %.1f ms  ·  n %d",
                sorted[(timingCount - 1) / 2], sorted[(int) Math.ceil(timingCount * .95) - 1], timingCount);
        return new Snapshot(observation, result, frame, source, ms, performance, SystemClock.elapsedRealtime());
    }

    private boolean enrolling;
    private Inspection.Reference enrollmentAnchor;

    private Inspection.Decision enrollFrame(Inspection.Frame frame, Inspection.Observation observation) {
        if (!observation.quality().problem().isEmpty() || observation.learned() != null) {
            return new Inspection.Decision("ENROLLING", "Use a clear, verified clean part.", enrolledCount);
        }
        if (enrolledCount == 0) {
            enrollmentAnchor = Inspection.Reference.from(new Inspection.Frame[]{frame, frame});
        } else {
            Inspection.Comparison check = enrollmentAnchor.compare(frame);
            if (!check.aligned() || check.anomaly() != null) {
                enrolledCount = 0;
                return new Inspection.Decision("ENROLLING", "Part shifted. Hold the fixture and phone still.", 0);
            }
        }
        enrollment[enrolledCount++] = frame;
        if (enrolledCount < enrollment.length) {
            return new Inspection.Decision("ENROLLING", "Capturing verified clean reference frames.", enrolledCount);
        }
        reference = Inspection.Reference.from(enrollment);
        enrolling = false;
        gate.reset();
        return new Inspection.Decision("VERIFYING", "Reference enrolled. Check the next part in the same fixture.", 0);
    }

    private void enroll() {
        if (!cameraLive || frozen || challenge != null) return;
        note("Place the verified clean part in the fixture and keep it still.");
        visionWorker.execute(() -> {
            reference = null;
            gate.reset();
            enrolledCount = 0;
            enrolling = true;
        });
    }

    private void inspectImage(Uri uri) {
        if (uri == null) return;
        runStill("IMPORTED IMAGE", () -> ImageDecoder.decodeBitmap(
                ImageDecoder.createSource(getContentResolver(), uri), (decoder, info, source) -> {
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                    int side = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                    if (side > 1024) decoder.setTargetSampleSize((side + 1023) / 1024);
                }));
    }

    private void specimens() {
        String[] labels = {"Annotated defect", "Clean control", "Lighting replay (simulated)"};
        new AlertDialog.Builder(this).setTitle("Held-out production specimens")
                .setItems(labels, (dialog, item) -> {
                    if (item == 2) runLightingReplay();
                    else runStill(item == 0 ? "KSDD2 · DEFECT" : "KSDD2 · CLEAN",
                            () -> fixtureImage(item == 0 ? "defect.png" : "clean.png"));
                }).show();
    }

    private void runLightingReplay() {
        resetInput();
        int session = generation;
        note("Comparing a held-out image with its simulated second exposure.");
        visionWorker.execute(() -> {
            try {
                Snapshot original = analyze(fixtureFrame("replay-ambient.png"), LIGHTING_REPLAY, false);
                Snapshot alternate = analyze(fixtureFrame("replay-relit.png"), LIGHTING_REPLAY, false);
                Lighting.Pair pair = Lighting.measure(original.frame, alternate.frame);
                String agreement = pair.status() == Lighting.Status.USABLE
                        ? Lighting.agree(original.observation.region(), alternate.observation.region(), pair).evidence()
                        : "comparison inconclusive";
                String outcome = "Simulated lighting replay · " + pair.evidence() + " · " + agreement;
                runOnUiThread(() -> {
                    if (session != generation || closed) return;
                    if (!original.decision.recordable()) {
                        showSnapshot(original);
                        note("Replay requires a usable original capture.");
                        return;
                    }
                    relitFrame = Frames.bitmap(alternate.frame);
                    relitMask = Frames.mask(alternate.observation.region(), alternate.observation.novel());
                    relitAlignment = pair;
                    challengeResult = outcome;
                    lock(original);
                    note("Simulated pair. Move the slider to compare exposures.");
                });
            } catch (Exception | LinkageError failure) {
                showVisionError(failure, session);
            }
        });
    }

    @FunctionalInterface private interface ImageLoader { Bitmap load() throws Exception; }

    private void runStill(String source, ImageLoader loader) {
        resetInput();
        note("Analyzing the still locally.");
        int session = generation;
        visionWorker.execute(() -> {
            try {
                Bitmap original = loader.load();
                if (original == null) throw new java.io.IOException("Image could not be decoded");
                Inspection.Frame frame = Frames.still(original);
                original.recycle();
                Snapshot snapshot = analyze(frame, source, false);
                runOnUiThread(() -> {
                    if (session != generation || closed) return;
                    showSnapshot(snapshot);
                    if (snapshot.decision.recordable()) lock(snapshot);
                    note("Still imaging uses the learned defect model.");
                });
            } catch (Exception | LinkageError failure) {
                showVisionError(failure, session);
            }
        });
    }

    private void resetInput() {
        generation++;
        cameraLive = false;
        if (camera != null) camera.getCameraControl().enableTorch(false);
        if (provider != null) provider.unbindAll();
        camera = null;
        challenge = null;
        clearLock();
        latest = null;
        latestFrame = null;
        latestMask = null;
        showSpecimen(false);
        state.setText("");
        finding.setText("");
        evidence.setText("");
        reason.setText("");
        timing.setText("");
        visionWorker.execute(() -> {
            reference = null;
            enrolling = false;
            enrolledCount = 0;
            lastAnalyzed = 0;
            timingIndex = timingCount = 0;
            gate.reset();
        });
        buttons();
    }

    private void showVisionError(Throwable failure, int session) {
        if (session != generation || closed) return;
        android.util.Log.e("SurfaceNet", "Vision inference failed", failure);
        runOnUiThread(() -> {
            if (session != generation || closed) return;
            resetInput();
            state.setText("VISION UNAVAILABLE");
            state.setTextColor(RED);
            inspectionResult.setVisibility(View.VISIBLE);
            note("Vision could not run. Check the bundled ExecuTorch model and app build.");
        });
    }

    private void showSnapshot(Snapshot snapshot) {
        latest = snapshot;
        var obs = snapshot.observation;
        var result = snapshot.decision;
        int accent = decisionColor(result.state());
        updateText(inputLabel, snapshot.source.toUpperCase(Locale.US) + (frozen ? "  /  FROZEN" : "  /  ANALYSIS"));
        if (!result.state().contentEquals(state.getText())) {
            state.setText(result.state());
            if (activeSection == 0) StationStyle.enter(state, dp(3));
        }
        state.setTextColor(accent);
        finding.setText(obs.kind());
        int pixels = obs.region() == null ? 0 : obs.region().pixels().length;
        evidence.setText(String.format(Locale.US, "%s severity  ·  %d mask px  ·  score %.3f",
                obs.severity(), pixels, obs.score()));
        reason.setText(result.reason() + (snapshot.source.equals("REAR CAMERA")
                ? String.format(Locale.US, "  %d/%d", result.agreement(),
                result.state().equals("ENROLLING") ? Inspection.ENROLL_COUNT : Inspection.WINDOW) : ""));
        updateText(lightingReadout, challengeResult);
        timing.setText("EDGE VISION  ·  " + snapshot.performance);
        latestFrame = Frames.bitmap(snapshot.frame);
        latestMask = Frames.mask(obs.region(), obs.novel());
        preview.render(latestFrame, latestMask, obs.region(), accent);
        preview.compare(relitFrame, relitMask, LIGHTING_REPLAY.equals(snapshot.source), relitAlignment);
        if (relitFrame != null) compareLabel.setText(LIGHTING_REPLAY.equals(snapshot.source) ? "B  Simulated light" : "B  Rear light");
        buttons();
    }

    private void freezeOrResume() {
        if (frozen) {
            clearLock();
            latest = null;
            if (cameraLive) {
                visionWorker.execute(gate::reset);
                note("Live inspection resumed.");
            } else {
                resetInput();
                note("Open a new image or start the rear camera.");
            }
            buttons();
        } else if (fresh()) lock(latest);
    }

    private boolean fresh() {
        return latest != null && latest.decision.recordable()
                && SystemClock.elapsedRealtime() - latest.time < 2_000;
    }

    private void lock(Snapshot snapshot) {
        frozen = true;
        locked = snapshot;
        showSnapshot(snapshot);
        recordedAdvice = Advice.procedure(snapshot.observation, snapshot.decision, challengeResult);
        advisory.setText(recordedAdvice.text() + "\n" + recordedAdvice.source());
        note("Evidence frozen. Confirm the finding and record a part or lot ID.");
        if (activeSection == 0) StationStyle.enter(preview, 0);
        buttons();
    }

    private void clearLock() {
        frozen = false;
        locked = null;
        recordedAdvice = null;
        saved = false;
        relitFrame = relitMask = null;
        relitAlignment = null;
        preview.compare(null, null, false, null);
        maskVisible = true;
        preview.setMaskVisible(true);
        maskToggle.setText("Hide mask");
        challengeResult = "Not run";
        advisory.setText("Freeze a finding to review its procedure.");
    }

    private void startChallenge() {
        if (benchmarking || camera == null || !camera.getCameraInfo().hasFlashUnit() || !fresh()
                || latest.observation.region() == null || !latest.decision.state().equals("HOLD")) return;
        Challenge candidate = new Challenge(latest);
        challenge = candidate;
        int session = generation;
        var torch = camera.getCameraControl().enableTorch(true);
        torch.addListener(() -> {
            if (challenge != candidate || session != generation) return;
            try { torch.get(); }
            catch (Exception failure) { finishChallenge(candidate, "Rear light unavailable", session, null, null, null); }
        }, getMainExecutor());
        note("Rear light verification. Hold the phone and part still.");
        buttons();
        preview.postDelayed(() -> {
            if (challenge != candidate) return;
            ChallengeCapture last = candidate.last;
            Snapshot capture = last == null ? null : last.snapshot();
            Lighting.Pair pair = last == null ? null : last.measurement();
            String reason = pair == null ? "No settled frame" : switch (pair.status()) {
                case CAPTURE_QUALITY -> "Exposure or focus";
                case ALIGNMENT -> "Frame moved or lacked texture";
                case INSUFFICIENT_LIGHT -> "Light change too small";
                case USABLE -> "Too few usable frames";
            };
            String outcome = "Inconclusive capture · " + reason + (pair == null ? "" : " · " + pair.evidence());
            finishChallenge(candidate, outcome, session, capture, capture == null ? null : capture.observation.region(), pair);
        }, 8500);
    }

    private void analyzeChallenge(Challenge candidate, Snapshot snapshot, int session) {
        if (snapshot.time - candidate.started < 950) return;
        Lighting.Pair pair = Lighting.measure(candidate.base.frame, snapshot.frame);
        candidate.last = new ChallengeCapture(snapshot, pair);
        if (pair.status() != Lighting.Status.USABLE) return;
        candidate.eligible++;
        Inspection.Observation original = candidate.base.observation;
        Inspection.Region sameBranch = original.novel() ? snapshot.observation.novelty() : snapshot.observation.learned();
        Lighting.Agreement agreement = Lighting.agree(original.region(), sameBranch, pair);
        if (agreement.confirmed()) {
            candidate.overlap++;
            candidate.example = snapshot;
            candidate.region = sameBranch;
            candidate.examplePair = pair;
            candidate.exampleAgreement = agreement;
        }
        if (candidate.eligible == 5) {
            boolean persistent = candidate.overlap >= 3;
            String outcome = persistent ? "Persistent under rear light" : "Light-sensitive or shifted";
            Lighting.Pair selectedPair = persistent ? candidate.examplePair : pair;
            Lighting.Agreement selectedAgreement = persistent ? candidate.exampleAgreement : agreement;
            String summary = outcome + " (" + candidate.overlap + "/5) · " + selectedPair.evidence()
                    + " · " + selectedAgreement.evidence();
            Snapshot example = persistent ? candidate.example : snapshot;
            Inspection.Region region = persistent ? candidate.region : snapshot.observation.region();
            runOnUiThread(() -> finishChallenge(candidate, summary, session, example, region, selectedPair));
        }
    }

    private void finishChallenge(Challenge candidate, String outcome, int session,
                                 Snapshot relit, Inspection.Region region, Lighting.Pair pair) {
        if (challenge != candidate || generation != session || closed) return;
        challenge = null;
        camera.getCameraControl().enableTorch(false);
        challengeResult = outcome;
        if (relit != null) {
            relitFrame = Frames.bitmap(relit.frame);
            relitMask = Frames.mask(region, region != null && region == relit.observation.novelty());
            relitAlignment = pair;
        }
        lock(candidate.base);
        note("Rear-light check complete. Original HOLD retained for review.");
    }

    private void generateAdvice() {
        if (locked == null || advising || importing || benchmarking || !advisor.configured()) return;
        Snapshot snapshot = locked;
        String lighting = challengeResult;
        advising = true;
        advisory.setText("Running Llama 3.2 locally via ExecuTorch…");
        buttons();
        // Llama initialization can resize the native thread pool retained by existing vision delegates.
        var unloadVision = visionWorker.submit(this::releaseVision);
        adviceWorker.execute(() -> {
            Advice.Result recommendation;
            try {
                unloadVision.get();
                recommendation = advisor.generate(snapshot.observation, snapshot.decision, lighting);
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                recommendation = Advice.procedure(snapshot.observation, snapshot.decision, lighting);
            } catch (java.util.concurrent.ExecutionException failure) {
                android.util.Log.w("SurfaceNet", "Could not prepare the runtime for local advice", failure);
                recommendation = Advice.procedure(snapshot.observation, snapshot.decision, lighting);
            }
            Advice.Result result = recommendation;
            runOnUiThread(() -> {
                if (closed) return;
                advising = false;
                if (locked == snapshot) {
                    recordedAdvice = result;
                    advisory.setText(result.text() + "\n" + result.source()
                            + (result.elapsedMs() == 0 ? "" : "  ·  " + result.elapsedMs() + " ms"));
                }
                buttons();
            });
        });
    }

    private void releaseVision() {
        if (model != null) {
            model.destroy();
            model = null;
        }
    }

    private void runBenchmark() {
        if (benchmarking || challenge != null || advising || importing || saving) {
            note("Finish the current check before running the benchmark.");
            return;
        }
        benchmarking = true;
        benchmarkResult.setText("Measuring single-view and four-view INT8…");
        benchmarkReadings.setVisibility(View.GONE);
        note("Camera analysis is paused during the benchmark.");
        buttons();
        visionWorker.execute(() -> {
            Module single = null, reflected = null;
            try {
                single = openVision("surface-int8.pte");
                reflected = openVision("surface-reflection-int8.pte");
                EValue[] inputs = {fixtureTensor("clean.png"), fixtureTensor("defect.png")};
                EdgeBenchmark.Report report = EdgeBenchmark.run(single, reflected, inputs,
                        (android.os.PowerManager) getSystemService(POWER_SERVICE));
                String json = report.json().toString(2);
                runOnUiThread(() -> {
                    if (closed) return;
                    benchmarking = false;
                    benchmarkJson = json;
                    benchmarkResult.setText("P95 · 120 warmed forwards per profile");
                    singleReading.setText(String.format(Locale.US, "%.2f ms", report.single().p95()));
                    reflectedReading.setText(String.format(Locale.US, "%.2f ms", report.reflected().p95()));
                    benchmarkReadings.setVisibility(View.VISIBLE);
                    buttons();
                    note("Device benchmark complete.");
                });
            } catch (Exception | LinkageError error) {
                android.util.Log.e("SurfaceNet", "Device benchmark failed", error);
                runOnUiThread(() -> {
                    if (closed) return;
                    benchmarking = false;
                    benchmarkResult.setText("Could not complete the device benchmark.");
                    buttons();
                    note("Could not complete the device benchmark.");
                });
            } finally {
                if (single != null) single.destroy();
                if (reflected != null) reflected.destroy();
                gate.reset();
            }
        });
    }

    private EValue fixtureTensor(String asset) {
        return tensor(fixtureFrame(asset));
    }

    private Bitmap fixtureImage(String asset) {
        try (InputStream stream = getAssets().open(asset)) {
            return BitmapFactory.decodeStream(stream);
        } catch (java.io.IOException failure) {
            android.util.Log.w("SurfaceNet", "Could not open specimen", failure);
            return null;
        }
    }

    private Inspection.Frame fixtureFrame(String asset) {
        Bitmap image = fixtureImage(asset);
        if (image == null) throw new IllegalStateException("Could not read " + asset);
        try { return Frames.still(image); }
        finally { image.recycle(); }
    }

    private void exportBenchmark(Uri uri) {
        exportTimingJson(uri,benchmarkJson,"Benchmark JSON");
    }

    private void showLiveTimings(LiveTimings.Summary stats) {
        if (stats.frames() == 0) {
            liveTimingStatus.setText("Start the rear camera and wait for live decisions before exporting timings.");
            return;
        }
        String text = String.format(Locale.US,"%d frames · %.1f decisions/s · analyzer p95 %.1f ms",
                stats.frames(),stats.decisionsPerSecond(),stats.analyzerP95Ms());
        if (stats.sensorP95Ms() != null) text += String.format(Locale.US,"\nSensor to decision p95 %.1f ms",stats.sensorP95Ms());
        liveTimingStatus.setText(text);
    }

    private void prepareLiveTimings() {
        LiveTimings.Summary stats = liveTimings.summary();
        if (stats.frames() < 2) {
            note("Run the rear camera until live decisions appear before exporting timings.");
            return;
        }
        showLiveTimings(stats);
        try {
            var power = (android.os.PowerManager)getSystemService(POWER_SERVICE);
            liveTimingsJson = new JSONObject().put("recorded_at",java.time.Instant.now().toString())
                    .put("device",android.os.Build.MODEL).put("sdk",android.os.Build.VERSION.SDK_INT)
                    .put("profile",liveTimingProfile)
                    .put("scope","Rolling CameraX analyzer samples to computed decision; excludes screen rendering and torch checks")
                    .put("frames",stats.frames()).put("duration_seconds",stats.spanSeconds())
                    .put("decisions_per_second",stats.decisionsPerSecond())
                    .put("analyzer_p50_ms",stats.analyzerP50Ms()).put("analyzer_p95_ms",stats.analyzerP95Ms())
                    .put("sensor_frames",stats.sensorFrames())
                    .put("sensor_p50_ms",java.util.Objects.requireNonNullElse(stats.sensorP50Ms(),JSONObject.NULL))
                    .put("sensor_p95_ms",java.util.Objects.requireNonNullElse(stats.sensorP95Ms(),JSONObject.NULL))
                    .put("thermal_status",android.os.Build.VERSION.SDK_INT >= 29 && power != null
                            ? power.getCurrentThermalStatus() : JSONObject.NULL).toString(2);
            liveTimingPicker.launch("surfacenet-live-timings.json");
        } catch (org.json.JSONException failure) {
            note("Could not prepare the live timing report.");
        }
    }

    private void exportTimingJson(Uri uri, String json, String label) {
        if (uri == null || json == null) return;
        traceWorker.execute(() -> {
            boolean saved = false;
            try (OutputStream stream = getContentResolver().openOutputStream(uri)) {
                if (stream != null) {
                    stream.write(json.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                    saved = true;
                }
            } catch (Exception error) {
                android.util.Log.e("SurfaceNet", "Benchmark export failed", error);
            }
            boolean success = saved;
            runOnUiThread(() -> { if (!closed) note(success ? label + " exported." : "Could not export timings."); });
        });
    }

    private void profileMenu() {
        String[] labels = {"Single view · lower latency", "Four views · higher sensitivity"};
        new AlertDialog.Builder(this).setTitle("Vision profile")
                .setSingleChoiceItems(labels, useReflection ? 1 : 0, (dialog, item) -> {
                    dialog.dismiss();
                    boolean next = item == 1;
                    if (next == useReflection) return;
                    resetInput();
                    useReflection = next;
                    buttons();
                    note(labels[item] + ". Start camera or select a specimen.");
                }).setNegativeButton("Close", null).show();
    }

    private void modelMenu() {
        String status = advisor.configured() ? "Llama files installed. Generate advice to verify runtime."
                : "Load the Llama 3.2 1B ExecuTorch model and matching tokenizer.";
        new AlertDialog.Builder(this).setTitle("Offline models")
                .setMessage("Vision · SurfaceNet v2 INT8 on ExecuTorch / XNNPACK\n\n" + status)
                .setPositiveButton("Import model", (dialog, which) -> modelPicker.launch(new String[]{"*/*"}))
                .setNeutralButton("Import tokenizer", (dialog, which) -> tokenizerPicker.launch(new String[]{"*/*"}))
                .setNegativeButton("Close", null).show();
    }

    private void importAsset(Uri uri, boolean tokenizer) {
        if (uri == null || importing || benchmarking || advising) return;
        importing = true;
        note(tokenizer ? "Importing tokenizer…" : "Copying model to private storage. Keep the app open.");
        buttons();
        adviceWorker.execute(() -> {
            String feedback;
            try {
                advisor.importFile(uri, tokenizer);
                feedback = advisor.configured() ? "Llama files installed. Verify on a frozen result."
                        : "File installed. Import the matching Llama file.";
            } catch (Exception failure) {
                feedback = "Import failed. " + failure.getMessage();
            }
            String result = feedback;
            runOnUiThread(() -> {
                if (closed) return;
                importing = false;
                    refreshModelStatus();
                note(result);
                buttons();
            });
        });
    }

    private void saveInspection() {
        if (locked == null || saving || saved || advising || recordedAdvice == null) return;
        String partId = part.getText().toString().trim();
        if (partId.isEmpty()) { part.setError("Enter a part ID or lot"); return; }
        Snapshot snapshot = locked;
        Bitmap frame = latestFrame, mask = latestMask, secondFrame = relitFrame, secondMask = relitMask;
        Advice.Result result = recordedAdvice;
        String lighting = challengeResult;
        saving = true;
        buttons();
        traceWorker.execute(() -> {
            boolean success;
            try {
                trace.save(partId, snapshot.source, snapshot.observation, snapshot.decision,
                        snapshot.latency, lighting, result, visionName, frame, mask, secondFrame, secondMask);
                success = true;
            } catch (RuntimeException failure) {
                android.util.Log.e("SurfaceNet", "Could not record inspection", failure);
                success = false;
            }
            boolean stored = success;
            runOnUiThread(() -> {
                if (closed) return;
                saving = false;
                if (locked == snapshot) saved = stored;
                note(stored ? "Inspection saved in the local trace log." : "Inspection could not be saved.");
                buttons();
                if (traceScreen.getVisibility() == View.VISIBLE) refreshTrace();
            });
        });
    }

    private void refreshTrace() {
        traceWorker.execute(() -> {
            var results = new java.util.ArrayList<String[]>();
            try (Cursor cursor = trace.latest()) {
                while (cursor.moveToNext()) {
                    String[] values = new String[Trace.COLUMNS.length];
                    for (int i = 0; i < values.length; i++) values[i] = cursor.getString(i);
                    results.add(values);
                }
            } catch (RuntimeException failure) {
                runOnUiThread(() -> note("Local records could not be read."));
                return;
            }
            runOnUiThread(() -> {
                if (closed) return;
                entries.removeAllViews();
                if (results.isEmpty()) entries.addView(label("No inspections recorded yet.", 14, DIM, false));
                for (String[] record : results) {
                    LinearLayout card = card();
                    card.addView(label(record[4], 11, decisionColor(record[4]), true));
                    card.addView(label(record[2], 22, INK, true));
                    card.addView(label(record[5] + "  ·  " + record[3], 13, DIM, false));
                    card.addView(label(record[1], 11, DIM, false));
                    card.addView(button("Review evidence", () -> openTraceDetail(record)));
                    entries.addView(card);
                }
            });
        });
    }

    private void openTraceDetail(String[] record) {
        if (traceLoading) return;
        traceLoading = true;
        traceListScroll = scroll.getScrollY();
        traceWorker.execute(() -> {
            byte[][] png;
            try {
                png = trace.evidence(record[0]);
            } catch (RuntimeException failure) {
                runOnUiThread(() -> {
                    traceLoading = false;
                    if (!closed) note("Saved evidence could not be opened.");
                });
                return;
            }
            Bitmap[] images = new Bitmap[png.length];
            for (int i = 0; i < png.length; i++) {
                if (png[i] != null) images[i] = BitmapFactory.decodeByteArray(png[i], 0, png[i].length);
            }
            Lighting.Pair alignment = images[0] != null && images[2] != null
                    ? Lighting.measure(Frames.still(images[0]), Frames.still(images[2])) : null;
            runOnUiThread(() -> {
                traceLoading = false;
                if (closed || activeSection != 1) return;
                traceDetail.removeAllViews();
                LinearLayout header = row();
                header.addView(title("Saved ", "evidence.", 42), new LinearLayout.LayoutParams(0, -2, 1));
                header.addView(button("Back to records", this::closeTraceDetail));
                traceDetail.addView(header);
                int accent = decisionColor(record[4]);
                traceDetail.addView(label(record[4] + "  /  " + record[2], 14, accent, true));
                EvidenceView image = new EvidenceView(this);
                image.render(images[0], images[1], null, accent);
                boolean simulated = LIGHTING_REPLAY.equals(record[3]);
                image.compare(images[2], images[3], simulated, alignment);
                traceDetail.addView(image, new LinearLayout.LayoutParams(-1, dp(278)));
                if (images[2] != null) {
                    traceDetail.addView(compareControls(image, "A  Original", simulated ? "B  Simulated light" : "B  Rear light"));
                }
                if (images[1] != null || images[3] != null) {
                    boolean[] showingMask = {true};
                    Button toggle = button("Hide mask", () -> {});
                    toggle.setOnClickListener(view -> {
                        showingMask[0] = !showingMask[0];
                        image.setMaskVisible(showingMask[0]);
                        toggle.setText(showingMask[0] ? "Hide mask" : "Show mask");
                    });
                    traceDetail.addView(toggle);
                }
                LinearLayout details = card();
                details.addView(label(record[5] + " · " + record[6], 14, INK, true));
                details.addView(label(record[14], 13, DIM, false));
                details.addView(label(record[10], 13, DIM, false));
                details.addView(label(record[11] + "\n" + record[12], 13, DIM, false));
                traceDetail.addView(details);
                traceList.setVisibility(View.GONE);
                traceDetail.setVisibility(View.VISIBLE);
                scroll.post(() -> {
                    if (activeSection != 1 || traceDetail.getVisibility() != View.VISIBLE) return;
                    scroll.scrollTo(0, 0);
                    StationStyle.enter(traceDetail, dp(7));
                });
            });
        });
    }

    private void closeTraceDetail() {
        traceDetail.setVisibility(View.GONE);
        traceDetail.removeAllViews();
        traceList.setVisibility(View.VISIBLE);
        scroll.post(() -> { if (activeSection == 1) scroll.scrollTo(0, traceListScroll); });
    }

    private void exportTrace(Uri uri) {
        if (uri == null) return;
        traceWorker.execute(() -> {
            String result;
            try (OutputStream output = getContentResolver().openOutputStream(uri, "wt")) {
                if (output == null) throw new java.io.IOException("Export destination unavailable");
                trace.export(output);
                result = "Inspection CSV exported.";
            } catch (Exception failure) { result = "Could not export inspection CSV."; }
            String feedback = result;
            runOnUiThread(() -> { if (!closed) note(feedback); });
        });
    }

    private void buttons() {
        boolean idle = challenge == null && !saving && !benchmarking;
        String profile = useReflection ? "Four views" : "Single view";
        updateText(deviceProfile, profile);
        int headingVisibility = frozen ? View.GONE : View.VISIBLE;
        if (inspectHeading.getVisibility() != headingVisibility) {
            inspectHeading.setVisibility(headingVisibility);
            inspectTitle.setTextSize(frozen ? 41 : 58);
            inspectTitle.setText(frozen ? titleText("Review ", "evidence.") : titleText("Inspect ", "surface."));
        }
        boolean canChangeProfile = idle && !frozen && !advising && !importing;
        profileButton.setEnabled(canChangeProfile);
        benchmarkButton.setEnabled(idle && !advising && !importing);
        benchmarkExport.setVisibility(benchmarkJson == null ? View.GONE : View.VISIBLE);
        benchmarkExport.setEnabled(!benchmarking);
        liveButton.setEnabled(idle);
        updateText(liveButton,cameraLive ? "Stop camera" : "Rear camera");
        importButton.setEnabled(idle);
        sampleButton.setEnabled(idle);
        replayButton.setEnabled(idle);
        inspectionResult.setVisibility(latest != null || cameraLive ? View.VISIBLE : View.GONE);
        workstation.setVisibility(frozen ? View.VISIBLE : View.GONE);
        controls.setVisibility(cameraLive || frozen ? View.VISIBLE : View.GONE);
        enrollButton.setVisibility(cameraLive && !frozen ? View.VISIBLE : View.GONE);
        enrollButton.setEnabled(idle && cameraLive && !frozen);
        updateText(freezeButton, frozen ? cameraLive ? "Resume live" : "Close evidence" : "Freeze result");
        freezeButton.setEnabled(idle && !advising && (frozen || fresh()));
        boolean canVerify = !frozen && camera != null && camera.getCameraInfo().hasFlashUnit()
                && fresh() && latest.decision.state().equals("HOLD") && latest.observation.region() != null;
        challengeButton.setVisibility(canVerify || challenge != null ? View.VISIBLE : View.GONE);
        challengeButton.setEnabled(idle && canVerify);
        updateText(challengeButton, challenge == null ? "Verify with rear light" : "Checking rear light…");
        adviseButton.setEnabled(frozen && !advising && !importing && idle && !saved && advisor.configured());
        saveButton.setEnabled(frozen && !advising && idle && !saved);
        updateText(saveButton, saved ? "Inspection saved" : "Save inspection");
        maskToggle.setVisibility(frozen && (latestMask != null || relitMask != null) ? View.VISIBLE : View.GONE);
        pairControls.setVisibility(frozen && relitFrame != null ? View.VISIBLE : View.GONE);
        lightingReadout.setVisibility(frozen && !"Not run".equals(challengeResult) ? View.VISIBLE : View.GONE);
        sources.setVisibility(frozen ? View.GONE : View.VISIBLE);
    }

    private LinearLayout column() {
        LinearLayout view = new LinearLayout(this);
        view.setOrientation(LinearLayout.VERTICAL);
        return view;
    }
    private LinearLayout row() {
        LinearLayout view = new LinearLayout(this);
        view.setGravity(Gravity.CENTER_VERTICAL);
        return view;
    }
    private LinearLayout card() {
        LinearLayout view = column();
        view.setBackground(StationStyle.plate(PANEL, LINE, dp(3)));
        view.setPadding(dp(14), dp(11), dp(14), dp(13));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.setMargins(0, dp(8), 0, dp(3));
        view.setLayoutParams(params);
        return view;
    }
    private void addHalf(LinearLayout row, Button button) {
        var params = new LinearLayout.LayoutParams(0, -2, 1);
        params.setMargins(dp(3), dp(2), dp(3), dp(2));
        row.addView(button, params);
    }
    private TextView label(String text, int size, int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(color);
        view.setTextSize(size);
        view.setPadding(0, dp(3), 0, dp(3));
        if (bold) view.setTypeface(size <= 11 ? Typeface.MONOSPACE : Typeface.DEFAULT, Typeface.BOLD);
        return view;
    }
    private Button button(String text, Runnable listener) {
        return button(text, listener, false);
    }
    private Button button(String text, Runnable listener, boolean strong) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextSize(13);
        button.setAllCaps(false);
        button.setMinHeight(dp(48));
        button.setMinimumWidth(dp(48));
        button.setMinWidth(dp(48));
        button.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        button.setPadding(dp(9), dp(5), dp(9), dp(5));
        button.setTextColor(StationStyle.buttonInk(strong));
        button.setBackground(StationStyle.button(strong, dp(3)));
        button.setOnClickListener(view -> listener.run());
        return button;
    }
    private TextView kicker(String text) {
        TextView view = label(text, 10, SIGNAL, true);
        view.setLetterSpacing(.075f);
        return view;
    }
    private TextView title(String main, String accent, int size) {
        TextView view = label("", size, INK, false);
        view.setTypeface(display);
        view.setLineSpacing(0, .9f);
        view.setAccessibilityHeading(true);
        view.setText(titleText(main, accent));
        return view;
    }
    private SpannableString titleText(String main, String accent) {
        SpannableString title = new SpannableString(main + accent);
        title.setSpan(new TypefaceSpan(editorial), main.length(), title.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        title.setSpan(new android.text.style.ForegroundColorSpan(SIGNAL), main.length(), title.length(),
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        return title;
    }
    private void line(LinearLayout container) {
        View view = new View(this);
        view.setBackgroundColor(LINE);
        container.addView(view, new LinearLayout.LayoutParams(-1, dp(1)));
    }
    private void toggleMask(EvidenceView image, Button toggle) {
        maskVisible = !maskVisible;
        image.setMaskVisible(maskVisible);
        toggle.setText(maskVisible ? "Hide mask" : "Show mask");
    }
    private static void updateText(TextView view, String value) {
        if (!value.contentEquals(view.getText())) view.setText(value);
    }
    private void refreshModelStatus() {
        updateText(deviceModel, advisor.configured() ? "Model and tokenizer installed."
                : "Import the model and tokenizer to use offline advice.");
    }
    private int dp(float value) { return (int) (value * getResources().getDisplayMetrics().density + .5f); }
    private void note(String content) {
        updateText(note, content);
        note.setVisibility(content.isEmpty() ? View.GONE : View.VISIBLE);
    }

    @Override protected void onDestroy() {
        closed = true;
        generation++;
        if (camera != null) camera.getCameraControl().enableTorch(false);
        if (provider != null) provider.unbindAll();
        visionWorker.execute(this::releaseVision);
        adviceWorker.execute(advisor::close);
        traceWorker.execute(trace::close);
        visionWorker.shutdown();
        adviceWorker.shutdown();
        traceWorker.shutdown();
        super.onDestroy();
    }
}
