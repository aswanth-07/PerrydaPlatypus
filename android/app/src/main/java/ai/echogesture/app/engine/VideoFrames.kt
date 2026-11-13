package ai.echogesture.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri

/** One frame picked from a video: its position among the picks, its time, and the prepared bitmap. */
class PickedFrame(val k: Int, val timeMs: Long, val bitmap: Bitmap)

/**
 * Reads the frames a clip is recognised from in one sequential pass of the hardware decoder.
 *
 * The earlier path asked MediaMetadataRetriever for each of the 24 frames separately; every call seeks back to a
 * keyframe and decodes forward again, which cost about 70 ms a frame on a flagship phone. Here the decoder runs
 * through the video once, only the picked frames are converted (YUV straight to an upright ARGB bitmap whose
 * longest side is at most 480 px, as for the training videos, with 2x2 averaging when shrinking), and each one is
 * handed over as soon as it is ready, so hand tracking runs while decoding continues.
 */
object VideoFrames {
    /**
     * Calls [onFrame] for up to [maxFrames] evenly spaced frames, in order. Returns false when this video cannot
     * be decoded this way (no flexible YUV output, unusual codec); the caller then falls back to the retriever.
     */
    fun decode(context: Context, uri: Uri, maxFrames: Int, maxSide: Int = 480, onFrame: (PickedFrame) -> Unit): Boolean {
        val meta = MediaMetadataRetriever()
        val count: Int
        val rotation: Int
        try {
            meta.setDataSource(context, uri)
            count = meta.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
            rotation = meta.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        } finally {
            meta.release()
        }
        if (count <= 0) return false
        val n = minOf(count, maxFrames)
        val picks = IntArray(n) { if (n == 1) 0 else Math.round(it * (count - 1).toDouble() / (n - 1)).toInt() }

        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: return false
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: return false
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            val fullRange = format.containsKey(MediaFormat.KEY_COLOR_RANGE) &&
                format.getInteger(MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL
            val c = MediaCodec.createDecoderByType(mime)
            codec = c
            c.configure(format, null, null, 0)
            c.start()

            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var index = 0          // decoded frames so far, in presentation order
            var next = 0           // next pick to deliver
            var firstUs = -1L
            while (next < n) {
                if (!inputDone) {
                    val inIdx = c.dequeueInputBuffer(5_000)
                    if (inIdx >= 0) {
                        val buf = c.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            c.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            c.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = c.dequeueOutputBuffer(info, 5_000)
                if (outIdx < 0) continue
                val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                if (info.size > 0) {
                    if (firstUs < 0) firstUs = info.presentationTimeUs
                    // Several picks can land on one frame when the clip is shorter than maxFrames' spacing.
                    if (picks[next] == index) {
                        val image = c.getOutputImage(outIdx)
                        if (image == null) { c.releaseOutputBuffer(outIdx, false); return next > 0 }
                        val bmp = image.use { toBitmap(it, maxSide, rotation, fullRange) }
                        val t = (info.presentationTimeUs - firstUs) / 1000
                        while (next < n && picks[next] == index) {
                            onFrame(PickedFrame(next, t, bmp))
                            next++
                        }
                    }
                    index++
                }
                c.releaseOutputBuffer(outIdx, false)
                if (eos) break
            }
            return next > 0
        } catch (e: Exception) {
            return false
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    /** YUV_420_888 -> ARGB at most [maxSide] px on the longest side, then rotated upright. */
    private fun toBitmap(image: Image, maxSide: Int, rotation: Int, fullRange: Boolean): Bitmap {
        val crop = image.cropRect
        val srcW = crop.width()
        val srcH = crop.height()
        val scale = minOf(1.0, maxSide.toDouble() / maxOf(srcW, srcH))
        val w = maxOf(1, Math.round(srcW * scale).toInt())
        val h = maxOf(1, Math.round(srcH * scale).toInt())
        val step = 1.0 / scale
        val avg = step >= 2.0   // shrinking by 2 or more: average a 2x2 block of luma to limit aliasing

        val (yP, uP, vP) = image.planes
        val yBuf = ByteArray(yP.buffer.remaining()).also { yP.buffer.get(it) }
        val uBuf = ByteArray(uP.buffer.remaining()).also { uP.buffer.get(it) }
        val vBuf = ByteArray(vP.buffer.remaining()).also { vP.buffer.get(it) }
        val yRow = yP.rowStride; val yPix = yP.pixelStride
        val uRow = uP.rowStride; val uPix = uP.pixelStride
        val vRow = vP.rowStride; val vPix = vP.pixelStride
        val out = IntArray(w * h)
        for (y in 0 until h) {
            val sy = crop.top + minOf(srcH - 1, (y * step).toInt())
            val sy2 = minOf(crop.bottom - 1, sy + 1)
            for (x in 0 until w) {
                val sx = crop.left + minOf(srcW - 1, (x * step).toInt())
                var yy = yBuf[sy * yRow + sx * yPix].toInt() and 0xFF
                if (avg) {
                    val sx2 = minOf(crop.right - 1, sx + 1)
                    yy = (yy + (yBuf[sy * yRow + sx2 * yPix].toInt() and 0xFF) +
                        (yBuf[sy2 * yRow + sx * yPix].toInt() and 0xFF) + (yBuf[sy2 * yRow + sx2 * yPix].toInt() and 0xFF) + 2) shr 2
                }
                val cy = sy shr 1
                val cx = sx shr 1
                val u = (uBuf[cy * uRow + cx * uPix].toInt() and 0xFF) - 128
                val v = (vBuf[cy * vRow + cx * vPix].toInt() and 0xFF) - 128
                val r: Int
                val g: Int
                val b: Int
                if (fullRange) {   // BT.601 full range
                    r = yy + ((1436 * v) shr 10)
                    g = yy - ((352 * u + 731 * v) shr 10)
                    b = yy + ((1815 * u) shr 10)
                } else {           // BT.601 limited (video) range
                    val c = 1192 * (yy - 16)
                    r = (c + 1634 * v) shr 10
                    g = (c - 833 * v - 400 * u) shr 10
                    b = (c + 2066 * u) shr 10
                }
                out[y * w + x] = (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
            }
        }
        val bmp = Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
        if (rotation % 360 == 0) return bmp
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, w, h, m, true)
    }
}
