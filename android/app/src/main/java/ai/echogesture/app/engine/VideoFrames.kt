package ai.echogesture.app.engine

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri

/** One frame picked from a video: its position among the picks, its time, the prepared bitmap, and total picks. */
class PickedFrame(val k: Int, val timeMs: Long, val bitmap: Bitmap, val total: Int = k + 1)

/**
 * Reads the frames a clip is recognised from in one sequential pass of the hardware decoder.
 *
 * The earlier path asked MediaMetadataRetriever for each of the 24 frames separately; every call seeks back to a
 * keyframe and decodes forward again, which cost about 70 ms a frame on a flagship phone. Here the decoder runs
 * through the video once, only the picked frames are converted (row-buffered from the codec's YUV ByteBuffers into
 * an upright ARGB bitmap whose longest side is at most 480 px, with LUT-hoisted scaling, 2x2 luma averaging when
 * shrinking >= 2x, and fused 90/180/270 rotation), and each one is handed over as soon as it is ready, so hand
 * tracking runs while decoding continues.
 */
object VideoFrames {
    private class Scratch {
        var pixels = IntArray(0)
        var yOffX = IntArray(0)
        var yOffX2 = IntArray(0)
        var uOffX = IntArray(0)
        var vOffX = IntArray(0)
        var yRow1 = ByteArray(0)
        var yRow2 = ByteArray(0)
        var uRow = ByteArray(0)
        var vRow = ByteArray(0)
    }

    /**
     * Calls [onFrame] for up to [maxFrames] evenly spaced frames, in order. Returns false when this video cannot
     * be decoded this way (no flexible YUV output, unusual codec); the caller then falls back to the retriever.
     */
    fun decode(context: Context, uri: Uri, maxFrames: Int, maxSide: Int = 480, onFrame: (PickedFrame) -> Unit): Boolean {
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
            val rotation = if (format.containsKey(MediaFormat.KEY_ROTATION)) format.getInteger(MediaFormat.KEY_ROTATION) else 0
            // Count frames in-process via MediaExtractor instead of spawning a MediaMetadataRetriever Binder IPC.
            var count = 0
            while (extractor.sampleTime >= 0L) {
                count++
                if (!extractor.advance()) break
            }
            if (count <= 0) return false
            extractor.seekTo(0L, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            if (extractor.sampleTime < 0L) return false
            val n = minOf(count, maxFrames)
            val picks = IntArray(n) { if (n == 1) 0 else Math.round(it * (count - 1).toDouble() / (n - 1)).toInt() }
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            val fullRange = format.containsKey(MediaFormat.KEY_COLOR_RANGE) &&
                format.getInteger(MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL
            val c = MediaCodec.createDecoderByType(mime)
            codec = c
            c.configure(format, null, null, 0)
            c.start()

            val info = MediaCodec.BufferInfo()
            val scratch = Scratch()
            var inputDone = false
            var index = 0          // decoded frames so far, in presentation order
            var next = 0           // next pick to deliver
            var firstUs = -1L
            while (next < n) {
                // Feed all immediately available input buffers so the hardware decoder stays saturated.
                while (!inputDone) {
                    val inIdx = c.dequeueInputBuffer(0)
                    if (inIdx < 0) break
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
                var outIdx = c.dequeueOutputBuffer(info, if (inputDone) 5_000 else 2_000)
                if (outIdx < 0) continue
                var hitEos = false
                while (outIdx >= 0 && next < n) {
                    val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    if (info.size > 0) {
                        if (firstUs < 0) firstUs = info.presentationTimeUs
                        // Several picks can land on one frame when the clip is shorter than maxFrames' spacing.
                        if (picks[next] == index) {
                            val image = c.getOutputImage(outIdx)
                            if (image == null) { c.releaseOutputBuffer(outIdx, false); return next == n }
                            val bmp = image.use { toBitmap(it, maxSide, rotation, fullRange, scratch) }
                            val t = (info.presentationTimeUs - firstUs) / 1000
                            while (next < n && picks[next] == index) {
                                onFrame(PickedFrame(next, t, bmp, n))
                                next++
                            }
                        }
                        index++
                    }
                    c.releaseOutputBuffer(outIdx, false)
                    if (eos) { hitEos = true; break }
                    outIdx = c.dequeueOutputBuffer(info, 0)
                }
                if (hitEos) break
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

    /** Row-buffered YUV_420_888 -> upright ARGB bitmap at most [maxSide] px on the longest side. */
    private fun toBitmap(image: Image, maxSide: Int, rotation: Int, fullRange: Boolean, scratch: Scratch): Bitmap {
        val crop = image.cropRect
        val srcW = crop.width()
        val srcH = crop.height()
        val scale = minOf(1.0, maxSide.toDouble() / maxOf(srcW, srcH))
        val w = maxOf(1, Math.round(srcW * scale).toInt())
        val h = maxOf(1, Math.round(srcH * scale).toInt())
        val step = 1.0 / scale
        val avg = step >= 2.0   // shrinking by 2 or more: average a 2x2 block of luma to limit aliasing

        val planes = image.planes
        val yP = planes[0]; val uP = planes[1]; val vP = planes[2]
        val yBuf = yP.buffer; val uBuf = uP.buffer; val vBuf = vP.buffer
        val yRowStride = yP.rowStride; val yPix = yP.pixelStride
        val uRowStride = uP.rowStride; val uPix = uP.pixelStride
        val vRowStride = vP.rowStride; val vPix = vP.pixelStride
        val yLimit = yBuf.limit()
        val uLimit = uBuf.limit()
        val vLimit = vBuf.limit()

        if (scratch.yOffX.size < w) {
            scratch.yOffX = IntArray(w)
            scratch.yOffX2 = IntArray(w)
            scratch.uOffX = IntArray(w)
            scratch.vOffX = IntArray(w)
        }
        val yOffX = scratch.yOffX
        val yOffX2 = scratch.yOffX2
        val uOffX = scratch.uOffX
        val vOffX = scratch.vOffX
        val cropLeft = crop.left
        val cropRight1 = crop.right - 1
        var maxYOff = 0
        var maxUOff = 0
        var maxVOff = 0
        for (x in 0 until w) {
            val sx = cropLeft + minOf(srcW - 1, (x * step).toInt())
            val sx2 = minOf(cropRight1, sx + 1)
            val cx = sx shr 1
            val yo = sx * yPix
            val yo2 = sx2 * yPix
            val uo = cx * uPix
            val vo = cx * vPix
            yOffX[x] = yo
            yOffX2[x] = yo2
            uOffX[x] = uo
            vOffX[x] = vo
            if (yo2 > maxYOff) maxYOff = yo2
            if (uo > maxUOff) maxUOff = uo
            if (vo > maxVOff) maxVOff = vo
        }
        val yNeed = maxYOff + 1
        val uNeed = maxUOff + 1
        val vNeed = maxVOff + 1
        if (scratch.yRow1.size < yNeed) scratch.yRow1 = ByteArray(yNeed)
        if (avg && scratch.yRow2.size < yNeed) scratch.yRow2 = ByteArray(yNeed)
        if (scratch.uRow.size < uNeed) scratch.uRow = ByteArray(uNeed)
        if (scratch.vRow.size < vNeed) scratch.vRow = ByteArray(vNeed)
        val yArr1 = scratch.yRow1
        val yArr2 = scratch.yRow2
        val uArr = scratch.uRow
        val vArr = scratch.vRow

        val rot = ((rotation % 360) + 360) % 360
        val outW = if (rot == 90 || rot == 270) h else w
        val outH = if (rot == 90 || rot == 270) w else h
        val need = outW * outH
        if (scratch.pixels.size < need) scratch.pixels = IntArray(need)
        val out = scratch.pixels

        val dstStep = when (rot) {
            90 -> h
            180 -> -1
            270 -> -h
            else -> 1
        }
        val cropTop = crop.top
        val cropBottom1 = crop.bottom - 1
        var lastCy = -1

        fun copyRow(src: java.nio.ByteBuffer, base: Int, limit: Int, dst: ByteArray, needLen: Int) {
            val avail = minOf(needLen, limit - base)
            if (avail > 0) {
                src.position(base)
                src.get(dst, 0, avail)
                if (avail < needLen) {
                    val pad = dst[avail - 1]
                    for (i in avail until needLen) dst[i] = pad
                }
            }
        }

        for (y in 0 until h) {
            val sy = cropTop + minOf(srcH - 1, (y * step).toInt())
            copyRow(yBuf, sy * yRowStride, yLimit, yArr1, yNeed)
            if (avg) {
                val sy2 = minOf(cropBottom1, sy + 1)
                copyRow(yBuf, sy2 * yRowStride, yLimit, yArr2, yNeed)
            }
            val cy = sy shr 1
            if (cy != lastCy) {
                copyRow(uBuf, cy * uRowStride, uLimit, uArr, uNeed)
                copyRow(vBuf, cy * vRowStride, vLimit, vArr, vNeed)
                lastCy = cy
            }
            var dst = when (rot) {
                90 -> h - 1 - y
                180 -> (h - 1 - y) * w + (w - 1)
                270 -> (w - 1) * h + y
                else -> y * w
            }
            if (fullRange) {
                if (avg) {
                    for (x in 0 until w) {
                        val ox = yOffX[x]; val ox2 = yOffX2[x]
                        val yy = ((yArr1[ox].toInt() and 0xFF) + (yArr1[ox2].toInt() and 0xFF) +
                            (yArr2[ox].toInt() and 0xFF) + (yArr2[ox2].toInt() and 0xFF) + 2) shr 2
                        val u = (uArr[uOffX[x]].toInt() and 0xFF) - 128
                        val v = (vArr[vOffX[x]].toInt() and 0xFF) - 128
                        var r = yy + ((1436 * v) shr 10)
                        var g = yy - ((352 * u + 731 * v) shr 10)
                        var b = yy + ((1815 * u) shr 10)
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255
                        out[dst] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        dst += dstStep
                    }
                } else {
                    for (x in 0 until w) {
                        val yy = yArr1[yOffX[x]].toInt() and 0xFF
                        val u = (uArr[uOffX[x]].toInt() and 0xFF) - 128
                        val v = (vArr[vOffX[x]].toInt() and 0xFF) - 128
                        var r = yy + ((1436 * v) shr 10)
                        var g = yy - ((352 * u + 731 * v) shr 10)
                        var b = yy + ((1815 * u) shr 10)
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255
                        out[dst] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        dst += dstStep
                    }
                }
            } else {
                if (avg) {
                    for (x in 0 until w) {
                        val ox = yOffX[x]; val ox2 = yOffX2[x]
                        val yy = ((yArr1[ox].toInt() and 0xFF) + (yArr1[ox2].toInt() and 0xFF) +
                            (yArr2[ox].toInt() and 0xFF) + (yArr2[ox2].toInt() and 0xFF) + 2) shr 2
                        val u = (uArr[uOffX[x]].toInt() and 0xFF) - 128
                        val v = (vArr[vOffX[x]].toInt() and 0xFF) - 128
                        val c = 1192 * (yy - 16)
                        var r = (c + 1634 * v) shr 10
                        var g = (c - 833 * v - 400 * u) shr 10
                        var b = (c + 2066 * u) shr 10
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255
                        out[dst] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        dst += dstStep
                    }
                } else {
                    for (x in 0 until w) {
                        val yy = yArr1[yOffX[x]].toInt() and 0xFF
                        val u = (uArr[uOffX[x]].toInt() and 0xFF) - 128
                        val v = (vArr[vOffX[x]].toInt() and 0xFF) - 128
                        val c = 1192 * (yy - 16)
                        var r = (c + 1634 * v) shr 10
                        var g = (c - 833 * v - 400 * u) shr 10
                        var b = (c + 2066 * u) shr 10
                        if (r < 0) r = 0 else if (r > 255) r = 255
                        if (g < 0) g = 0 else if (g > 255) g = 255
                        if (b < 0) b = 0 else if (b > 255) b = 255
                        out[dst] = -0x1000000 or (r shl 16) or (g shl 8) or b
                        dst += dstStep
                    }
                }
            }
        }
        val bmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, outW, 0, 0, outW, outH)
        return bmp
    }
}
