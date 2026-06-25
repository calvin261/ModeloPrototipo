package com.lsec.translator.ml

import kotlin.math.sqrt

/**
 * Normaliza 21 landmarks de mano a un vector invariante a escala y posicion.
 *
 * Port exacto de normalize_landmarks() de Python (landmarks.py):
 * 1. Traslada para que wrist (landmark 0) quede en el origen
 * 2. Escala para que la distancia wrist -> MCP9 (landmark 9) sea 1.0
 * 3. Retorna FloatArray(63) aplanado
 */
object LandmarkNormalizer {

    /**
     * Normaliza un array de 63 floats (21 landmarks x 3 coordenadas).
     *
     * @param rawLandmarks FloatArray de 63 elementos [x0,y0,z0, x1,y1,z1, ...]
     * @return FloatArray(63) normalizado
     */
    fun normalize(rawLandmarks: FloatArray): FloatArray {
        require(rawLandmarks.size == 63) {
            "Expected 63 values (21 landmarks x 3 coords), got ${rawLandmarks.size}"
        }

        // Copiar a array 2D (21 x 3)
        val coords = Array(21) { i ->
            floatArrayOf(
                rawLandmarks[i * 3],
                rawLandmarks[i * 3 + 1],
                rawLandmarks[i * 3 + 2]
            )
        }

        // Trasladar wrist (landmark 0) al origen
        val wristX = coords[0][0]
        val wristY = coords[0][1]
        val wristZ = coords[0][2]
        for (i in 0 until 21) {
            coords[i][0] -= wristX
            coords[i][1] -= wristY
            coords[i][2] -= wristZ
        }

        // Escalar por distancia a landmark 9 (middle finger MCP)
        val refDist = sqrt(
            coords[9][0] * coords[9][0] +
            coords[9][1] * coords[9][1] +
            coords[9][2] * coords[9][2]
        )

        if (refDist > 1e-6f) {
            for (i in 0 until 21) {
                coords[i][0] /= refDist
                coords[i][1] /= refDist
                coords[i][2] /= refDist
            }
        }

        // Aplanar de vuelta a FloatArray(63)
        return FloatArray(63) { idx -> coords[idx / 3][idx % 3] }
    }

    /**
     * Normaliza los landmarks de mano izquierda, mano derecha y pose superior
     * usando los hombros como origen de coordenadas.
     *
     * @param leftHand raw FloatArray de 63 floats (21 landmarks x 3 coords) o null
     * @param rightHand raw FloatArray de 63 floats (21 landmarks x 3 coords) o null
     * @param poseLandmarks raw FloatArray de 99 floats (33 landmarks x 3 coords) o null
     * @return FloatArray(147) normalizado y aplanado
     */
    fun normalizeHolistic(
        leftHand: FloatArray?,
        rightHand: FloatArray?,
        poseLandmarks: FloatArray?
    ): FloatArray {
        val result = FloatArray(147)
        val origin = FloatArray(3)
        var scale = 1.0f

        // Si la pose superior esta disponible, calculamos el punto medio de los hombros y la escala
        if (poseLandmarks != null && poseLandmarks.size > 12 * 3) {
            val lShX = poseLandmarks[11 * 3]
            val lShY = poseLandmarks[11 * 3 + 1]
            val lShZ = poseLandmarks[11 * 3 + 2]

            val rShX = poseLandmarks[12 * 3]
            val rShY = poseLandmarks[12 * 3 + 1]
            val rShZ = poseLandmarks[12 * 3 + 2]

            origin[0] = (lShX + rShX) / 2.0f
            origin[1] = (lShY + rShY) / 2.0f
            origin[2] = (lShZ + rShZ) / 2.0f

            val dx = lShX - rShX
            val dy = lShY - rShY
            val dz = lShZ - rShZ
            scale = sqrt(dx * dx + dy * dy + dz * dz)
            if (scale < 1e-6f) {
                scale = 1.0f
            }
        }

        // 1. Mano Izquierda (indices 0 a 62)
        if (leftHand != null && leftHand.size >= 63) {
            for (i in 0 until 21) {
                result[i * 3] = (leftHand[i * 3] - origin[0]) / scale
                result[i * 3 + 1] = (leftHand[i * 3 + 1] - origin[1]) / scale
                result[i * 3 + 2] = (leftHand[i * 3 + 2] - origin[2]) / scale
            }
        }

        // 2. Mano Derecha (indices 63 a 125)
        if (rightHand != null && rightHand.size >= 63) {
            for (i in 0 until 21) {
                result[63 + i * 3] = (rightHand[i * 3] - origin[0]) / scale
                result[63 + i * 3 + 1] = (rightHand[i * 3 + 1] - origin[1]) / scale
                result[63 + i * 3 + 2] = (rightHand[i * 3 + 2] - origin[2]) / scale
            }
        }

        // 3. Puntos de Pose: nose (0), l_sh (11), r_sh (12), l_el (13), r_el (14), l_wr (15), r_wr (16) (indices 126 a 146)
        if (poseLandmarks != null) {
            val keyIndices = intArrayOf(0, 11, 12, 13, 14, 15, 16)
            for (j in keyIndices.indices) {
                val idx = keyIndices[j]
                if (idx * 3 + 2 < poseLandmarks.size) {
                    result[126 + j * 3] = (poseLandmarks[idx * 3] - origin[0]) / scale
                    result[126 + j * 3 + 1] = (poseLandmarks[idx * 3 + 1] - origin[1]) / scale
                    result[126 + j * 3 + 2] = (poseLandmarks[idx * 3 + 2] - origin[2]) / scale
                }
            }
        }

        return result
    }
}
