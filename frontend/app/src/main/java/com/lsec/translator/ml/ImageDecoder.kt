package com.lsec.translator.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * Decodificador YUV_420_888 → Bitmap.
 *
 * IMPORTANTE: cada [Image] solo puede decodificarse UNA vez porque sus
 * ByteBuffers internos son de un solo uso. Si los consumes dos veces, la
 * segunda llamada corrompe memoria y crashea con SIGSEGV en
 * `Yuv420SpToJpegEncoder::deinterleave`.
 *
 * Por eso este helper existe: lo llamas una vez por frame y reusas el
 * Bitmap resultante en todos los detectores.
 */
object ImageDecoder {

    /**
     * Convierte [image] (YUV_420_888) a un Bitmap RGB.
     *
     * @throws IllegalArgumentException si el formato no es YUV_420_888.
     */
    fun imageToBitmap(image: Image): Bitmap {
        require(image.format == ImageFormat.YUV_420_888) {
            "Formato de imagen no soportado: ${image.format} (esperado YUV_420_888)"
        }

        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // NV21 layout: YYYYY... VU VU VU ...
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null,
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            100,
            out,
        )
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Rota un bitmap [degrees]° en sentido HORARIO (convención de CameraX).
     *
     * Si el grado es 0 retorna el mismo bitmap (sin copia). En otro caso
     * crea un bitmap nuevo con las dimensiones intercambiadas según
     * corresponda. El bitmap original se libera.
     */
    fun rotateBitmap(source: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return source

        val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }
}