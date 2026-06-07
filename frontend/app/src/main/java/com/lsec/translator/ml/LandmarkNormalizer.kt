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
}
