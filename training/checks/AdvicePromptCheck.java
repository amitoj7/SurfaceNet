package io.surfaceguard.android;

public final class AdvicePromptCheck {
    public static void main(String[] args) {
        if (args.length == 2 && args[0].equals("parse")) {
            System.out.print(Procedure.selection(args[1]));
            return;
        }
        var quality = new Inspection.Quality(120, .02, 4);
        String fixture = args.length == 0 ? "line" : args[0];
        boolean pass = fixture.equals("clean");
        boolean clean = pass || fixture.equals("still-clean");
        boolean compact = fixture.equals("compact");
        var region = compact ? new Inspection.Region(20, 30, 43, 57, new int[663], 1.6)
                : new Inspection.Region(40, 40, 45, 55, new int[]{0, 1, 2, 3}, 8);
        String lighting = switch (fixture) {
            case "clean", "still-clean", "compact" -> "Not run";
            case "shifted" -> "Light-sensitive or shifted (0/5)";
            default -> "Persistent under rear light (4/5)";
        };
        var observation = new Inspection.Observation(clean ? .075 : .873, quality, true,
                clean ? null : region, null);
        var decision = new Inspection.Decision(pass ? "PASS" : clean || compact ? "REVIEW" : "HOLD",
                "smoke test", pass ? 4 : clean || compact ? 1 : 3);
        var choices = Procedure.choices(observation, decision, lighting);
        if (fixture.equals("shifted") && (choices.a().contains("engineering") || choices.b().contains("engineering"))) {
            throw new AssertionError("A shifted finding requires visual confirmation");
        }
        System.out.print(Procedure.prompt(observation, decision,
                lighting, choices));
    }
}
