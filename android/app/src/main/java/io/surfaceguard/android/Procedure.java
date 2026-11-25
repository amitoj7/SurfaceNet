package io.surfaceguard.android;

import java.util.Locale;

final class Procedure {
    private static final java.util.regex.Pattern CHOICE = java.util.regex.Pattern.compile(
            "(?:The correct answer is:\\s*)?([AB])[:.]?(?:\\s*<\\|eot_id\\|>)?");

    record Choices(String a, String b) {}

    static String selection(String output) {
        var match = CHOICE.matcher(output.trim());
        return match.matches() ? match.group(1) : "";
    }

    static Choices choices(Inspection.Observation observation, Inspection.Decision decision, String challenge) {
        if (challenge.startsWith("Light-sensitive") || challenge.startsWith("Inconclusive")
                || challenge.startsWith("Rear light unavailable")) {
            return new Choices(
                    "Keep the part segregated. Reseat it in the fixture and repeat the lighting check.",
                    "Keep the part segregated. Inspect the outlined area under calibrated side lighting.");
        }
        if (decision.state().equals("PASS")) return new Choices(
                "Record the clean result and follow the station release procedure.",
                "Compare with the approved sample before the operator releases the part.");
        if (observation.region() == null) return new Choices(
                "Keep the part in review. Compare it with a verified clean reference.",
                "Keep the part in review. Reinspect it under the approved fixture lighting before release.");
        if (observation.novel()) return new Choices(
                "Keep the part in review. Compare it with a verified clean reference.",
                "Keep the part segregated and ask quality engineering to identify the surface change.");
        if (observation.kind().startsWith("Crack")) return new Choices(
                "Keep the part segregated. Confirm the line under magnification.",
                "Keep the part segregated. Ask quality engineering to distinguish a crack from a handling scratch.");
        if (observation.kind().startsWith("Compact")) return new Choices(
                "Keep the part segregated. Check the spot under calibrated side lighting.",
                "Keep the part segregated. Ask quality engineering to verify pit depth.");
        return new Choices("Keep the part segregated and verify the mark under side lighting.",
                "Keep the part segregated and ask quality engineering to compare the lot and handling surfaces.");
    }

    static String prompt(Inspection.Observation observation, Inspection.Decision decision,
                         String challenge, Choices choices) {
        String evidence = String.format(Locale.US,
                "Disposition: %s. Finding: %s. Severity proxy: %s. Agreement: %d. Altered lighting: %s.",
                decision.state(), observation.kind(), observation.severity(), decision.agreement(), challenge);
        String answerPrefix = decision.state().equals("PASS") ? "" : "Answer:";
        return "<|start_header_id|>system<|end_header_id|>\n\n"
                + "Choose one factory quality-control procedure using the recorded evidence. "
                + "If a finding moved under rear light or the capture was inconclusive, choose A for confirmation. "
                + "Choose B for high severity with persistent evidence or for a repeatable unknown change. Choose A for a clean part. "
                + "Return only the letter A or B.<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\n" + evidence
                + "\nA: " + choices.a + "\nB: " + choices.b + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n" + answerPrefix;
    }
}
