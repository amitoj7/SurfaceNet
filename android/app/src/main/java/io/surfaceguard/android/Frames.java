package io.surfaceguard.android;

import android.graphics.Bitmap;
import android.graphics.Color;
import androidx.camera.core.ImageProxy;
import java.nio.ByteBuffer;

final class Frames {
    static Inspection.Frame camera(ImageProxy image) {
        int rotation = image.getImageInfo().getRotationDegrees();
        int width = image.getWidth(), height = image.getHeight();
        boolean sideways = rotation == 90 || rotation == 270;
        int uprightWidth = sideways ? height : width, uprightHeight = sideways ? width : height;
        int cropHeight = (int) Math.min(uprightHeight * .82, uprightWidth * .86 * Inspection.H / Inspection.W);
        int cropWidth = cropHeight * Inspection.W / Inspection.H;
        int left = (uprightWidth - cropWidth) / 2, top = (uprightHeight - cropHeight) / 2;
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer pixels = plane.getBuffer();
        int offset = pixels.position(), rowStride = plane.getRowStride(), pixelStride = plane.getPixelStride();
        byte[] gray = new byte[Inspection.N];
        for (int y = 0; y < Inspection.H; y++) {
            int py = top + (2 * y + 1) * cropHeight / (2 * Inspection.H);
            for (int x = 0; x < Inspection.W; x++) {
                int px = left + (2 * x + 1) * cropWidth / (2 * Inspection.W);
                int rawX = switch (rotation) {
                    case 90 -> py;
                    case 180 -> width - 1 - px;
                    case 270 -> width - 1 - py;
                    default -> px;
                };
                int rawY = switch (rotation) {
                    case 90 -> height - 1 - px;
                    case 180 -> height - 1 - py;
                    case 270 -> px;
                    default -> py;
                };
                gray[y * Inspection.W + x] = pixels.get(offset + rawY * rowStride + rawX * pixelStride);
            }
        }
        return new Inspection.Frame(gray);
    }

    static Inspection.Frame still(Bitmap image) {
        Bitmap resized = Bitmap.createScaledBitmap(image, Inspection.W, Inspection.H, true);
        int[] pixels = new int[Inspection.N];
        resized.getPixels(pixels, 0, Inspection.W, 0, 0, Inspection.W, Inspection.H);
        if (resized != image) resized.recycle();
        byte[] gray = new byte[Inspection.N];
        for (int i = 0; i < gray.length; i++) {
            int rgb = pixels[i];
            gray[i] = (byte) ((77 * Color.red(rgb) + 150 * Color.green(rgb) + 29 * Color.blue(rgb) + 128) >> 8);
        }
        return new Inspection.Frame(gray);
    }

    static Bitmap bitmap(Inspection.Frame frame) {
        int[] pixels = new int[Inspection.N];
        for (int i = 0; i < pixels.length; i++) {
            int gray = frame.at(i);
            pixels[i] = Color.rgb(gray, gray, gray);
        }
        return Bitmap.createBitmap(pixels, Inspection.W, Inspection.H, Bitmap.Config.ARGB_8888);
    }

    static Bitmap mask(Inspection.Region region, boolean novelty) {
        if (region == null) return null;
        int[] pixels = new int[Inspection.N];
        int color = novelty ? 0x99F7C36B : 0xB3F07865;
        for (int index : region.pixels()) pixels[index] = color;
        return Bitmap.createBitmap(pixels, Inspection.W, Inspection.H, Bitmap.Config.ARGB_8888);
    }
}
