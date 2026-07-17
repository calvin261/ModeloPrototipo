package com.lsec.translator.ui.screens.camera

import androidx.compose.ui.geometry.Offset

/**
 * Conversor de coordenadas normalizadas MediaPipe → pixeles del PreviewView.
 *
 * Premisas:
 *   - El bitmap entregado a MediaPipe puede estar en orientación de SENSOR
 *     (no display) si `ImageAnalysis.targetRotation` no se fija.
 *   - Los landmarks vienen en [0,1] × [0,1] sobre ese bitmap.
 *   - PreviewView usa `FILL_CENTER` por defecto: la imagen se escala hasta
 *     llenar la vista (recortando si hay overflow), centrada.
 *   - Para cámara frontal el usuario ve la imagen espejada (como un selfie).
 *
 * Pipeline del mapeo:
 *   1. Si hay rotación, aplicar la rotación inversa al landmark para que
 *      coincida con la orientación de display.
 *   2. FILL_CENTER scaling + centrado.
 *   3. Mirror horizontal si cámara frontal.
 *
 * Para cámara frontal se espeja horizontalmente sobre el eje vertical de la vista.
 */
object CoordinateMapper {

    /**
     * Rota un punto normalizado (nx, ny) en [0,1] × [0,1] según [rotation]
     * grados en sentido HORARIO (convención de CameraX).
     *
     * Útil cuando el bitmap viene rotado respecto al display y queremos
     * alinear los landmarks con lo que el usuario ve.
     */
    fun rotateNormalized(nx: Float, ny: Float, rotation: Int, out: FloatArray) {
        when ((rotation % 360 + 360) % 360) {
            0 -> { out[0] = nx; out[1] = ny }
            90 -> { out[0] = ny; out[1] = 1f - nx }
            180 -> { out[0] = 1f - nx; out[1] = 1f - ny }
            270 -> { out[0] = 1f - ny; out[1] = nx }
            else -> { out[0] = nx; out[1] = ny }
        }
    }

    /**
     * Mapea un punto normalizado [0,1] a coordenadas de Compose.
     *
     * @param nx coordenada x normalizada del landmark
     * @param ny coordenada y normalizada del landmark
     * @param imageWidth ancho del bitmap en pixeles
     * @param imageHeight alto del bitmap en pixeles
     * @param viewWidth ancho del PreviewView en pixeles (Compose px)
     * @param viewHeight alto del PreviewView en pixeles
     * @param isFrontCamera true si se debe espejar (selfie)
     * @param imageRotationDegrees rotación del bitmap vs display (0/90/180/270)
     */
    fun map(
        nx: Float,
        ny: Float,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Float,
        viewHeight: Float,
        isFrontCamera: Boolean,
        imageRotationDegrees: Int = 0,
    ): Offset {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0f || viewHeight <= 0f) {
            return Offset(0f, 0f)
        }

        // 1) Aplicar rotación inversa al landmark para alinear con display
        val tmp = FloatArray(2)
        rotateNormalized(nx, ny, imageRotationDegrees, tmp)
        val rx = tmp[0]
        val ry = tmp[1]

        // 2) FILL_CENTER
        val scale = maxOf(viewWidth / imageWidth, viewHeight / imageHeight)
        val drawnW = imageWidth * scale
        val drawnH = imageHeight * scale
        val dx = (viewWidth - drawnW) / 2f
        val dy = (viewHeight - drawnH) / 2f

        val pxRaw = rx * imageWidth * scale + dx
        val pyRaw = ry * imageHeight * scale + dy

        // 3) Mirror para cámara frontal
        return if (isFrontCamera) {
            Offset(viewWidth - pxRaw, pyRaw)
        } else {
            Offset(pxRaw, pyRaw)
        }
    }

    /**
     * Variante batch: mapea un FloatArray [x0,y0,z0,x1,y1,z1,...] a pixeles
     * del view usando un buffer reusable [out] de tamaño `count * 2`.
     */
    fun mapMany(
        landmarks: FloatArray,
        offset: Int,
        count: Int,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Float,
        viewHeight: Float,
        isFrontCamera: Boolean,
        imageRotationDegrees: Int = 0,
        out: FloatArray,
    ) {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0f || viewHeight <= 0f) {
            for (i in 0 until count) {
                out[i * 2] = 0f
                out[i * 2 + 1] = 0f
            }
            return
        }

        val rotation = imageRotationDegrees
        val scale = maxOf(viewWidth / imageWidth, viewHeight / imageHeight)
        val drawnW = imageWidth * scale
        val drawnH = imageHeight * scale
        val dx = (viewWidth - drawnW) / 2f
        val dy = (viewHeight - drawnH) / 2f

        val tmp = FloatArray(2)
        for (i in 0 until count) {
            val base = offset + i * 3
            val nx = landmarks[base]
            val ny = landmarks[base + 1]

            // 1) Rotación
            rotateNormalized(nx, ny, rotation, tmp)
            val rx = tmp[0]
            val ry = tmp[1]

            // 2) FILL_CENTER
            val pxRaw = rx * imageWidth * scale + dx
            val pyRaw = ry * imageHeight * scale + dy

            // 3) Mirror
            if (isFrontCamera) {
                out[i * 2] = viewWidth - pxRaw
                out[i * 2 + 1] = pyRaw
            } else {
                out[i * 2] = pxRaw
                out[i * 2 + 1] = pyRaw
            }
        }
    }
}