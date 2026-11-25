package io.surfaceguard.android;

import android.animation.ValueAnimator;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.view.View;
import android.view.animation.PathInterpolator;

final class StationStyle {
    static final int BG = 0xff151712, PANEL = 0xff1e211a, INK = 0xfff6f1df;
    static final int DIM = 0xffb5b5a3, LINE = 0xff45483b, SIGNAL = 0xfff2ab78;
    static final int GREEN = 0xffc8dc9a, AMBER = 0xfff4cf88, RED = 0xffff9c89;
    private static final PathInterpolator EASE_OUT = new PathInterpolator(.2f, .7f, .25f, 1f);

    private StationStyle() {}

    static GradientDrawable plate(int fill, int line, int radius) {
        var background = new GradientDrawable();
        background.setColor(fill);
        if (line != 0) background.setStroke(1, line);
        background.setCornerRadius(radius);
        return background;
    }

    static RippleDrawable button(boolean strong, int radius) {
        int fill = strong ? SIGNAL : PANEL;
        var disabled = plate(0xff25271f, 0xff34372b, radius);
        var idle = plate(fill, strong ? SIGNAL : LINE, radius);
        var states = new android.graphics.drawable.StateListDrawable();
        states.addState(new int[]{-android.R.attr.state_enabled}, disabled);
        states.addState(new int[]{android.R.attr.state_focused}, plate(fill, INK, radius));
        states.addState(new int[]{android.R.attr.state_selected}, plate(0xff303026, SIGNAL, radius));
        states.addState(new int[0], idle);
        return new RippleDrawable(ColorStateList.valueOf(strong ? 0x33151712 : 0x33f2ab78),
                states, null);
    }

    static ColorStateList buttonInk(boolean strong) {
        return new ColorStateList(new int[][]{{-android.R.attr.state_enabled}, {android.R.attr.state_selected}, {}},
                new int[]{0xff939685, SIGNAL, strong ? BG : INK});
    }

    static RippleDrawable navigation(int radius) {
        var states = new android.graphics.drawable.StateListDrawable();
        states.addState(new int[]{android.R.attr.state_focused}, plate(PANEL, SIGNAL, radius));
        states.addState(new int[]{android.R.attr.state_selected}, plate(0xff2c2a20, 0, radius));
        states.addState(new int[0], plate(BG, 0, radius));
        return new RippleDrawable(ColorStateList.valueOf(0x33f2ab78), states, null);
    }

    static int decisionColor(String state) {
        return switch (state) {
            case "PASS" -> GREEN;
            case "HOLD", "VISION UNAVAILABLE" -> RED;
            case "REVIEW", "REACQUIRE" -> AMBER;
            default -> SIGNAL;
        };
    }

    static void enter(View view, float distance) {
        view.animate().cancel();
        view.setAlpha(1);
        view.setTranslationY(0);
        if (!ValueAnimator.areAnimatorsEnabled() || !view.isLaidOut()) return;
        view.setAlpha(.55f);
        view.setTranslationY(distance);
        view.animate().alpha(1).translationY(0).setDuration(210).setInterpolator(EASE_OUT).start();
    }

    static final class DockIcon extends Drawable {
        private final Paint stroke = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final int section;
        private final int size;

        DockIcon(int section, int size, boolean selected) {
            this.section = section;
            this.size = size;
            stroke.setStrokeWidth(1.35f);
            stroke.setStrokeCap(Paint.Cap.ROUND);
            stroke.setStyle(Paint.Style.STROKE);
            stroke.setColor(selected ? SIGNAL : DIM);
        }

        @Override public void draw(Canvas canvas) {
            Rect bounds = getBounds();
            canvas.save();
            canvas.translate(bounds.left, bounds.top);
            canvas.scale(bounds.width() / 22f, bounds.height() / 22f);
            switch (section) {
                case 0 -> {
                    corner(canvas, 3, 3, 1, 1); corner(canvas, 19, 3, -1, 1);
                    corner(canvas, 3, 19, 1, -1); corner(canvas, 19, 19, -1, -1);
                    canvas.drawCircle(11, 11, 3.15f, stroke);
                }
                case 1 -> {
                    canvas.drawRect(5, 3, 17, 19, stroke);
                    canvas.drawLine(8, 8, 14, 8, stroke);
                    canvas.drawLine(8, 12, 14, 12, stroke);
                    canvas.drawLine(8, 16, 12, 16, stroke);
                }
                case 2 -> {
                    canvas.drawRect(5, 5, 17, 17, stroke);
                    canvas.drawRect(8, 8, 14, 14, stroke);
                    for (int pin = 8; pin <= 14; pin += 3) {
                        canvas.drawLine(pin, 2, pin, 5, stroke);
                        canvas.drawLine(pin, 17, pin, 20, stroke);
                        canvas.drawLine(2, pin, 5, pin, stroke);
                        canvas.drawLine(17, pin, 20, pin, stroke);
                    }
                }
            }
            canvas.restore();
        }

        private void corner(Canvas c, int x, int y, int horizontal, int vertical) {
            c.drawLine(x, y, x + horizontal * 5, y, stroke);
            c.drawLine(x, y, x, y + vertical * 5, stroke);
        }

        @Override public int getIntrinsicWidth() { return size; }
        @Override public int getIntrinsicHeight() { return size; }
        @Override public void setAlpha(int alpha) { stroke.setAlpha(alpha); }
        @Override public void setColorFilter(ColorFilter filter) { stroke.setColorFilter(filter); }
        @Override public int getOpacity() { return PixelFormat.TRANSLUCENT; }
    }
}
