package ai.echogesture.app.engine

import android.graphics.Bitmap
import android.graphics.Matrix
import ai.echogesture.signcore.Hand
import ai.echogesture.signcore.Layout
import ai.echogesture.signcore.RawFrame
import ai.echogesture.signcore.Slots
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

/** MediaPipe Holistic output -> the signcore frame the model was trained on (same as src/holistic_kp.py). */
fun HolisticLandmarkerResult.toRawFrame(): RawFrame {
    val poseList = poseLandmarks()
    val pose = if (poseList.isEmpty()) null else FloatArray(33 * 4).also { a ->
        for (k in 0 until minOf(33, poseList.size)) {
            val p = poseList[k]
            a[k * 4] = p.x(); a[k * 4 + 1] = p.y(); a[k * 4 + 2] = p.z()
            a[k * 4 + 3] = p.visibility().orElse(1f)
        }
    }
    val found = ArrayList<Hand>(2)
    val sides = listOf(leftHandLandmarks() to leftHandWorldLandmarks(), rightHandLandmarks() to rightHandWorldLandmarks())
    for ((img, world) in sides) {
        if (img.isEmpty()) continue
        val image = FloatArray(63) { k -> img[k / 3].let { p -> when (k % 3) { 0 -> p.x(); 1 -> p.y(); else -> p.z() } } }
        val metric = if (world.isEmpty()) null else
            FloatArray(63) { k -> world[k / 3].let { p -> when (k % 3) { 0 -> p.x(); 1 -> p.y(); else -> p.z() } } }
        found.add(Hand(image, metric))
    }
    val slots = Slots.assign(found, pose)   // by distance to the pose wrists, not MediaPipe's handedness label
    val faceList = faceLandmarks()
    val face = if (faceList.size <= Layout.FACE_MESH.max()) null else FloatArray(Layout.NFACE * 3).also { a ->
        for ((k, id) in Layout.FACE_MESH.withIndex()) {
            val p = faceList[id]
            a[k * 3] = p.x(); a[k * 3 + 1] = p.y(); a[k * 3 + 2] = p.z()
        }
    }
    return RawFrame(pose, arrayOf(slots[0]?.image, slots[1]?.image), arrayOf(slots[0]?.world, slots[1]?.world), face)
}

/** Upright, ARGB_8888, longest side at most [maxSide] px: the frame size the model's training videos had. */
fun Bitmap.prepared(rotationDegrees: Int = 0, maxSide: Int = 480): Bitmap {
    var b = if (config == Bitmap.Config.ARGB_8888) this else copy(Bitmap.Config.ARGB_8888, false)
    val scale = minOf(1f, maxSide.toFloat() / maxOf(b.width, b.height))
    if (rotationDegrees != 0 || scale < 1f) {
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()); postScale(scale, scale) }
        b = Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true)
    }
    return b
}

/** The 21-point hand skeleton, for drawing. */
val HAND_EDGES = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
)
