package com.lsec.translator.ml

/**
 * Topología de conexiones entre landmarks para dibujar el overlay.
 *
 * Basado en las constantes oficiales de MediaPipe:
 *   - HAND_CONNECTIONS        (21 landmarks por mano, 20 aristas)
 *   - FACEMESH_TESSELATION    (468 landmarks cara, ~80 aristas clave)
 *   - FACEMESH_CONTOURS       (contornos de ojos/nariz/boca/cejas)
 *
 * Mantenemos versiones reducidas para cuidar el rendimiento al pintar
 * 30 veces por segundo, pero dejamos el código extensible para activar
 * la malla completa si el usuario lo necesita.
 */
object LandmarkConnections {

    /**
     * Aristas de la mano según MediaPipe Hands.
     * Cada par (a,b) conecta los índices de landmarks.
     */
    val HAND: List<Pair<Int, Int>> = listOf(
        // Pulgar
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        // Índice
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        // Medio
        5 to 9, 9 to 10, 10 to 11, 11 to 12,
        // Anular
        9 to 13, 13 to 14, 14 to 15, 15 to 16,
        // Meñique
        13 to 17, 17 to 18, 18 to 19, 19 to 20,
        // Palma (base)
        0 to 17,
    )

    /**
     * Contornos faciales según MediaPipe Face Mesh (FACEMESH_CONTOURS).
     * Versión reducida: solo los contornos principales que el usuario
     * reconoce fácilmente (no la malla tesselation completa de 80+ aristas).
     *
     * Los índices siguen la convención de FaceLandmarker 478 puntos.
     */
    val FACE_CONTOURS: List<Pair<Int, Int>> = buildList {
        // === Ceja izquierda (70..75 aprox, contornos 0..10) ===
        addAll(faceOval(70, 63, 68, 71, 72, 63))
        // === Ceja derecha (300..305 aprox) ===
        addAll(faceOval(300, 293, 298, 301, 302, 293))
        // === Ojos ===
        // Izquierdo: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        addAll(listOf(
            33 to 7, 7 to 163, 163 to 144, 144 to 145, 145 to 153, 153 to 154,
            154 to 155, 155 to 133, 133 to 173, 173 to 157, 157 to 158, 158 to 159,
            159 to 160, 160 to 161, 161 to 246, 246 to 33
        ))
        // Derecho: 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466
        addAll(listOf(
            263 to 249, 249 to 390, 390 to 373, 373 to 374, 374 to 380, 380 to 381,
            381 to 382, 382 to 362, 362 to 398, 398 to 384, 384 to 385, 385 to 386,
            386 to 387, 387 to 388, 388 to 466, 466 to 263
        ))
        // === Iris izquierdo (468..472) y derecho (473..477) ===
        addAll(listOf(468 to 469, 469 to 470, 470 to 471, 471 to 472, 472 to 468))
        addAll(listOf(473 to 474, 474 to 475, 475 to 476, 476 to 477, 477 to 473))
        // === Nariz (puente y fosas) ===
        addAll(listOf(168 to 6, 6 to 197, 197 to 195, 195 to 5, 5 to 4, 4 to 1, 1 to 19))
        addAll(listOf(2 to 94, 94 to 2))
        // === Boca exterior ===
        addAll(listOf(
            61 to 84, 84 to 17, 17 to 314, 314 to 405, 405 to 320, 320 to 307,
            307 to 375, 375 to 321, 321 to 308, 308 to 324, 324 to 318, 318 to 402,
            402 to 317, 317 to 14, 14 to 87, 87 to 178, 178 to 88, 88 to 95,
            95 to 78, 78 to 191, 191 to 80, 80 to 81, 81 to 82, 82 to 13, 13 to 312,
            312 to 311, 311 to 310, 310 to 415, 415 to 308
        ))
        // === Boca interior (labios internos) ===
        addAll(listOf(
            78 to 95, 95 to 88, 88 to 178, 178 to 87, 87 to 14, 14 to 317, 317 to 402,
            402 to 318, 318 to 324, 324 to 308
        ))
        // === Contorno oval de la cara (face oval) ===
        addAll(listOf(
            10 to 338, 338 to 297, 297 to 332, 332 to 284, 284 to 251, 251 to 389,
            389 to 356, 356 to 454, 454 to 323, 323 to 361, 361 to 288, 288 to 397,
            397 to 365, 365 to 379, 379 to 378, 378 to 400, 400 to 377, 377 to 152,
            152 to 148, 148 to 176, 176 to 149, 149 to 150, 150 to 136, 136 to 172,
            172 to 58, 58 to 132, 132 to 93, 93 to 234, 234 to 127, 127 to 162,
            162 to 21, 21 to 54, 54 to 103, 103 to 67, 67 to 109, 109 to 10
        ))
    }

    /**
     * Versión aún más ligera: solo los puntos CLAVE para depurar
     * señas dinámicas (nariz, ojos, boca, mentón, frente).
     * Útil cuando `batterySaverMode = true` o dispositivo gama baja.
     */
    val FACE_KEY_POINTS: Set<Int> = setOf(
        // Nariz y puente
        1, 2, 4, 5, 6, 19, 94, 168, 195, 197,
        // Ojos (centro de cada párpado)
        33, 133, 144, 145, 158, 159, 160, 161,
        263, 362, 373, 374, 384, 385, 386, 387,
        // Iris
        468, 473,
        // Boca
        13, 14, 17, 78, 87, 95, 178, 308, 317, 318, 324, 402,
        // Mentón y mandíbula
        152, 172, 200, 201, 202,
        // Frente
        10, 67, 109, 151, 337,
    )

    private fun faceObalHelper(vararg indices: Int): List<Pair<Int, Int>> =
        indices.toList().zipWithNext().filter { it.first != it.second }.map { it.first to it.second }

    /**
     * Helper para evitar errores: si la versión de FaceLandmarker
     * cambia el orden de landmarks, no peta. Esta función se usa en tests.
     */
    @Suppress("unused")
    fun faceOval(vararg idx: Int): List<Pair<Int, Int>> = faceObalHelper(*idx)
}