package ai.echogesture.app.engine

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.camera.core.ImageProxy
import ai.echogesture.signcore.Hand
import ai.echogesture.signcore.Layout
import ai.echogesture.signcore.RawFrame
import ai.echogesture.signcore.Slots
import com.google.mediapipe.tasks.components.containers.Landmark
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult
import java.nio.ByteBuffer

private val FACE_MESH_MAX = Layout.FACE_MESH.max()

private fun copyNormalizedHand(img: List<NormalizedLandmark>): FloatArray {
    val out = FloatArray(63)
    val n = minOf(21, img.size)
    for (k in 0 until n) {
        val p = img[k]
        val base = k * 3
        out[base] = p.x()
        out[base + 1] = p.y()
        out[base + 2] = p.z()
    }
    return out
}

private fun copyWorldHand(world: List<Landmark>): FloatArray? {
    if (world.isEmpty()) return null
    val out = FloatArray(63)
    val n = minOf(21, world.size)
    for (k in 0 until n) {
        val p = world[k]
        val base = k * 3
        out[base] = p.x()
        out[base + 1] = p.y()
        out[base + 2] = p.z()
    }
    return out
}

/** MediaPipe Holistic output -> the signcore frame the model was trained on (same as src/holistic_kp.py). */
fun HolisticLandmarkerResult.toRawFrame(): RawFrame {
    val poseList = poseLandmarks()
    val pose = if (poseList.isEmpty()) null else FloatArray(33 * 4).also { a ->
        val n = minOf(33, poseList.size)
        for (k in 0 until n) {
            val p = poseList[k]
            val base = k * 4
            a[base] = p.x()
            a[base + 1] = p.y()
            a[base + 2] = p.z()
            a[base + 3] = p.visibility().orElse(1f)
        }
    }
    val found = ArrayList<Hand>(2)
    val lh = leftHandLandmarks()
    if (lh.isNotEmpty()) found.add(Hand(copyNormalizedHand(lh), copyWorldHand(leftHandWorldLandmarks()), preferredSlot = 0))
    val rh = rightHandLandmarks()
    if (rh.isNotEmpty()) found.add(Hand(copyNormalizedHand(rh), copyWorldHand(rightHandWorldLandmarks()), preferredSlot = 1))
    val slots = Slots.assign(found, pose)   // by distance to the pose wrists, not MediaPipe's handedness label
    val faceList = faceLandmarks()
    val face = if (faceList.size <= FACE_MESH_MAX) null else FloatArray(Layout.NFACE * 3).also { a ->
        val mesh = Layout.FACE_MESH
        for (k in mesh.indices) {
            val p = faceList[mesh[k]]
            val base = k * 3
            a[base] = p.x()
            a[base + 1] = p.y()
            a[base + 2] = p.z()
        }
    }
    return RawFrame(pose, arrayOf(slots[0]?.image, slots[1]?.image), arrayOf(slots[0]?.world, slots[1]?.world), face)
}

/** Upright, ARGB_8888, longest side at most [maxSide] px: the frame size the model's training videos had. */
fun Bitmap.prepared(rotationDegrees: Int = 0, maxSide: Int = 480): Bitmap {
    val copied = config != Bitmap.Config.ARGB_8888
    val src = if (copied) copy(Bitmap.Config.ARGB_8888, false) else this
    val scale = minOf(1f, maxSide.toFloat() / maxOf(src.width, src.height))
    if (rotationDegrees != 0 || scale < 1f) {
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()); postScale(scale, scale) }
        val out = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        if (copied && src !== out) src.recycle()
        return out
    }
    return src
}

/**
 * Reusable RGBA frame converter for live CameraX ImageAnalysis: avoids allocating two Bitmaps (~1.8 MB) per frame
 * (54 MB/s of GC churn at 30 fps). Must be called from CameraX's single-threaded analyzer executor.
 */
internal class CameraFrameConverter(private val maxSide: Int = 480, slots: Int = 4) {
    private var srcBitmap: Bitmap? = null
    private var packedBuf: ByteBuffer? = null
    private val ring = arrayOfNulls<Bitmap>(slots)
    private val canvases = arrayOfNulls<Canvas>(slots)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var nextSlot = 0

    fun convert(proxy: ImageProxy): Bitmap {
        val rot = ((proxy.imageInfo.rotationDegrees % 360) + 360) % 360
        val srcW = proxy.width
        val srcH = proxy.height
        val swap = rot == 90 || rot == 270
        val rotW = if (swap) srcH else srcW
        val rotH = if (swap) srcW else srcH
        val scale = minOf(1f, maxSide.toFloat() / maxOf(rotW, rotH))
        val outW = maxOf(1, Math.round(rotW * scale))
        val outH = maxOf(1, Math.round(rotH * scale))

        val slot = nextSlot
        nextSlot = (nextSlot + 1) % ring.size
        var dst = ring[slot]
        if (dst == null || dst.isRecycled || dst.width != outW || dst.height != outH) {
            dst = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            ring[slot] = dst
            canvases[slot] = Canvas(dst)
        }

        val planes = proxy.planes
        if (planes.size == 1 && planes[0].pixelStride == 4) {
            val plane = planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val rowBytes = srcW * 4
            if (rot == 0 && outW == srcW && outH == srcH) {
                copyRgbaToBitmap(buf, rowStride, rowBytes, srcH, dst)
                return dst
            }
            var src = srcBitmap
            if (src == null || src.isRecycled || src.width != srcW || src.height != srcH) {
                src?.recycle()
                src = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)
                srcBitmap = src
            }
            copyRgbaToBitmap(buf, rowStride, rowBytes, srcH, src)
            drawTransformed(src, srcW, srcH, rot, rotW, rotH, outW, outH, canvases[slot]!!)
            return dst
        }

        // Unexpected format fallback.
        val fallback = proxy.toBitmap()
        return try {
            drawTransformed(fallback, fallback.width, fallback.height, rot, rotW, rotH, outW, outH, canvases[slot]!!)
            dst
        } finally {
            fallback.recycle()
        }
    }

    private fun copyRgbaToBitmap(buf: ByteBuffer, rowStride: Int, rowBytes: Int, height: Int, target: Bitmap) {
        buf.rewind()
        if (rowStride == rowBytes) {
            target.copyPixelsFromBuffer(buf)
            return
        }
        val need = rowBytes * height
        var packed = packedBuf
        if (packed == null || packed.capacity() < need) {
            packed = ByteBuffer.allocateDirect(need)
            packedBuf = packed
        }
        packed.clear()
        val dup = buf.duplicate()
        for (y in 0 until height) {
            val start = y * rowStride
            dup.limit(start + rowBytes)
            dup.position(start)
            packed.put(dup)
        }
        packed.flip()
        target.copyPixelsFromBuffer(packed)
    }

    private fun drawTransformed(
        src: Bitmap, srcW: Int, srcH: Int, rot: Int, rotW: Int, rotH: Int, outW: Int, outH: Int, canvas: Canvas,
    ) {
        matrix.reset()
        matrix.postTranslate(-srcW * 0.5f, -srcH * 0.5f)
        if (rot != 0) matrix.postRotate(rot.toFloat())
        matrix.postTranslate(rotW * 0.5f, rotH * 0.5f)
        if (outW != rotW || outH != rotH) matrix.postScale(outW.toFloat() / rotW, outH.toFloat() / rotH)
        canvas.drawBitmap(src, matrix, paint)
    }
}

/** The 21-point hand skeleton, for drawing. */
val HAND_EDGES = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
)
