package io.surfaceguard.android;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.animation.PathInterpolator;
import java.util.function.IntConsumer;

final class EvidenceView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final RectF roi = new RectF();
    private final RectF alignedAlternate = new RectF();
    private final float density;
    private final Typeface display;
    private Bitmap frame, mask, alternate, alternateMask;
    private Inspection.Region region;
    private int accent = StationStyle.SIGNAL;
    private boolean specimen, maskVisible = true, simulated;
    private Lighting.Pair alignment;
    private float split = .5f, sweep = -1;
    private Shader lamps;
    private ValueAnimator reveal;
    private IntConsumer splitListener;
    private float touchX, touchY;
    private boolean scrubbing;

    EvidenceView(Context context) {
        super(context);
        density = getResources().getDisplayMetrics().density;
        display = Typeface.createFromAsset(context.getAssets(), "type/barlow-condensed-semibold.ttf");
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_YES);
    }

    void specimen(Bitmap example, boolean animate) {
        render(example, null, null, StationStyle.SIGNAL);
        specimen = true;
        describe();
        if (animate && ValueAnimator.areAnimatorsEnabled()) {
            reveal = ValueAnimator.ofFloat(0f, 1f);
            reveal.setDuration(640);
            reveal.setInterpolator(new PathInterpolator(.25f, .15f, .3f, 1f));
            reveal.addUpdateListener(animation -> {
                float progress = (float) animation.getAnimatedValue();
                sweep = progress >= 1f ? -1 : progress;
                invalidate();
            });
            reveal.start();
        }
        invalidate();
    }

    void render(Bitmap frame, Bitmap mask, Inspection.Region region, int accent) {
        if (specimen && reveal != null) {
            reveal.cancel();
            reveal = null;
            sweep = -1;
        }
        this.frame = frame;
        this.mask = mask;
        this.region = region;
        this.accent = accent;
        specimen = false;
        describe();
        invalidate();
    }

    void compare(Bitmap alternate, Bitmap alternateMask, boolean simulated, Lighting.Pair pair) {
        this.alternate = alternate;
        this.alternateMask = alternateMask;
        this.simulated = simulated;
        alignment = pair != null && pair.status() == Lighting.Status.USABLE ? pair : null;
        describe();
        invalidate();
    }

    void setSplit(int percent) {
        split = percent / 100f;
        invalidate();
    }

    void setSplitListener(IntConsumer listener) { splitListener = listener; }

    @Override public boolean onTouchEvent(MotionEvent event) {
        if (alternate == null || splitListener == null) return super.onTouchEvent(event);
        float x = event.getX(), y = event.getY();
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN -> {
                if (!roi.contains(x, y)) return false;
                touchX = x;
                touchY = y;
                scrubbing = false;
                return true;
            }
            case MotionEvent.ACTION_MOVE -> {
                float dx = Math.abs(x - touchX), dy = Math.abs(y - touchY);
                if (!scrubbing && dx >= ViewConfiguration.get(getContext()).getScaledTouchSlop() && dx > dy) {
                    scrubbing = true;
                    getParent().requestDisallowInterceptTouchEvent(true);
                }
                if (scrubbing) scrub(x);
                return true;
            }
            case MotionEvent.ACTION_UP -> {
                if (scrubbing || roi.contains(x, y)) scrub(x);
                if (!scrubbing) performClick();
                scrubbing = false;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
            }
            case MotionEvent.ACTION_CANCEL -> {
                scrubbing = false;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
            }
            default -> { return true; }
        }
    }

    @Override public boolean performClick() {
        super.performClick();
        return true;
    }

    private void scrub(float x) {
        int percent = Math.round((x - roi.left) * 100f / roi.width());
        splitListener.accept(Math.max(0, Math.min(100, percent)));
    }

    void setMaskVisible(boolean visible) {
        maskVisible = visible;
        describe();
        invalidate();
    }

    private void describe() {
        String description;
        if (specimen) description = "Annotated demonstration specimen from KolektorSDD2";
        else if (frame == null) description = "Inspection viewport waiting for capture";
        else if (alternate != null) description = simulated
                ? "Original and simulated alternate exposure. Adjust split to compare."
                : "Original and rear-light exposure. Adjust split to compare.";
        else description = maskVisible && mask != null ? "Inspection region with defect overlay" : "Inspection region";
        if (!description.contentEquals(getContentDescription() == null ? "" : getContentDescription())) {
            setContentDescription(description);
        }
    }

    @Override protected void onSizeChanged(int width, int height, int previousWidth, int previousHeight) {
        lamps = new LinearGradient(0, 0, width, height, 0xff303328, 0xff141612, Shader.TileMode.CLAMP);
    }

    @Override protected void onDraw(Canvas canvas) {
        float w = getWidth(), h = getHeight();
        paint.setShader(lamps);
        canvas.drawRect(0, 0, w, h, paint);
        paint.setShader(null);
        paint.setColor(0xff373a2d);
        paint.setStrokeWidth(density * .6f);
        for (float x = w / 2 % (22 * density); x < w; x += 22 * density) canvas.drawLine(x, 0, x, h, paint);
        for (float y = h / 2 % (22 * density); y < h; y += 22 * density) canvas.drawLine(0, y, w, y, paint);

        float halfHeight = Math.min(h * .425f, w * .34f);
        float halfWidth = halfHeight * Inspection.W / Inspection.H;
        float mid = specimen ? w * .70f : w * .5f;
        roi.set(mid - halfWidth, h / 2 - halfHeight, mid + halfWidth, h / 2 + halfHeight);
        paint.setColor(0xff12140f);
        canvas.drawRect(roi, paint);
        if (frame != null) canvas.drawBitmap(frame, null, roi, paint);
        if (maskVisible && mask != null) canvas.drawBitmap(mask, null, roi, paint);

        if (alternate != null) {
            float boundary = roi.left + split * roi.width();
            canvas.save();
            canvas.clipRect(boundary, roi.top, roi.right, roi.bottom);
            alignedAlternate.set(roi);
            if (alignment != null) {
                alignedAlternate.offset(-alignment.dx() * roi.width() / Inspection.W,
                        -alignment.dy() * roi.height() / Inspection.H);
                paint.setColor(0xff12140f);
                canvas.drawRect(roi, paint);
            }
            canvas.drawBitmap(alternate, null, alignedAlternate, paint);
            if (maskVisible && alternateMask != null) canvas.drawBitmap(alternateMask, null, alignedAlternate, paint);
            canvas.restore();
            paint.setColor(StationStyle.SIGNAL);
            paint.setStrokeWidth(density);
            canvas.drawLine(boundary, roi.top, boundary, roi.bottom, paint);
            mono(canvas, "A", roi.left - density * 17, roi.top + density * 11, StationStyle.INK);
            mono(canvas, "B", roi.right + density * 9, roi.top + density * 11, StationStyle.SIGNAL);
        } else if (maskVisible && region != null) {
            float sx = roi.width() / Inspection.W, sy = roi.height() / Inspection.H;
            paint.setColor(accent);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(density);
            canvas.drawRect(roi.left + region.minX() * sx, roi.top + region.minY() * sy,
                    roi.left + (region.maxX() + 1) * sx, roi.top + (region.maxY() + 1) * sy, paint);
            paint.setStyle(Paint.Style.FILL);
        }
        if (sweep >= 0) {
            float y = roi.top + sweep * roi.height();
            paint.setColor(0x80f2ab78);
            canvas.drawRect(roi.left, y, roi.right, y + density, paint);
        }

        markers(canvas, roi.left, roi.top, roi.right, roi.bottom, accent, 9 * density);
        markers(canvas, 1, 1, w - 1, h - 1, 0xff74745e, 12 * density);
        if (specimen) {
            float left = density * 17;
            mono(canvas, "KOLEKTORSDD2", left, h * .37f, StationStyle.SIGNAL);
            paint.setTypeface(display);
            paint.setColor(StationStyle.INK);
            paint.setTextSize(Math.min(w * .093f, density * 35));
            canvas.drawText("Defect", left, h * .53f, paint);
            canvas.drawText("specimen", left, h * .65f, paint);
        } else if (frame == null) {
            mono(canvas, "96 × 256 px", density * 15, h - density * 14, StationStyle.DIM);
        }
    }

    private void markers(Canvas canvas, float left, float top, float right, float bottom, int color, float length) {
        paint.setColor(color);
        paint.setStrokeWidth(density * 1.15f);
        corner(canvas, left, top, length, length);
        corner(canvas, right, top, -length, length);
        corner(canvas, left, bottom, length, -length);
        corner(canvas, right, bottom, -length, -length);
    }

    private void corner(Canvas canvas, float x, float y, float dx, float dy) {
        canvas.drawLine(x, y, x + dx, y, paint);
        canvas.drawLine(x, y, x, y + dy, paint);
    }

    private void mono(Canvas canvas, String text, float x, float y, int color) {
        paint.setTypeface(Typeface.MONOSPACE);
        paint.setTextSize(density * 10);
        paint.setColor(color);
        canvas.drawText(text, x, y, paint);
    }

    @Override protected void onDetachedFromWindow() {
        if (reveal != null) reveal.cancel();
        reveal = null;
        sweep = -1;
        super.onDetachedFromWindow();
    }
}
