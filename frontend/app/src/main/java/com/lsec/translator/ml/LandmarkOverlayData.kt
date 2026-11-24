package com.lsec.translator.ml

/**
 * Snapshot inmutable de los landmarks crudos de UN frame, listos para
 * pintarse en el overlay de la cámara.
 *
 * Todas las coordenadas son NORMALIZADAS [0,1] en el espacio del bitmap
 * (que ya viene en orientación de display gracias a ImageAnalysis).
 *
 * Estructura:
 *   - Cada "landmarks" es un FloatArray de tamaño [N * 3] (x, y, z aplanados)
 *   - z está en unidades relativas a la profundidad del plano de la imagen
 *     (típicamente en [-0.1, 0.1] para manos; [-0.2, 0.2] para cara).
 *
 * Optimización: usamos FloatArray en lugar de List<List<Float>> para evitar
 * boxing y permitir dibujar 478 puntos × 30 fps sin GC pressure.
 */
data class LandmarkOverlayData(
    val leftHand: FloatArray?,   // 21 × 3 = 63 floats, o null si no hay
    val rightHand: FloatArray?,  // 21 × 3 = 63 floats, o null si no hay
    val face: FloatArray?,       // N × 3 (N puede ser 478, 468, o menos), o null
    val imageWidth: Int,
    val imageHeight: Int,
    val timestampMs: Long,
    /**
     * Rotación en grados del bitmap respecto a la orientación de display.
     * Viene de [androidx.camera.core.ImageInfo.getRotationDegrees].
     *
     * Típicamente:
     *   - 0   = sensor y display ya alineados
     *   - 90  = bitmap en landscape, display en portrait (caso típico)
     *   - 180 = display invertido (tablet boca abajo)
     *   - 270 = bitmap en portrait, display en landscape
     *
     * El CoordinateMapper usa este valor para girar los landmarks antes
     * de mapearlos a pixeles del Compose view.
     */
    val imageRotationDegrees: Int = 0,
    /**
     * Versión corta del face mesh (subset FACEMESH_LIPS, FACEMESH_LEFT_EYE, etc.)
     * Si es null usamos todos los landmarks de `face`.
     * Sirve para reducir dibujo en modo batería baja.
     */
    val faceContoursOnly: Boolean = false,
    /** Pose completo (33 x 3) para normalizar el clasificador sin re-inferirlo. */
    val pose: FloatArray? = null,
) {
    /**
     * Devuelve el número de landmarks de cara, o 0 si no hay.
     */
    fun faceCount(): Int = (face?.size ?: 0) / 3

    /**
     * Hand left = 21 landmarks?
     */
    fun hasLeftHand(): Boolean = leftHand != null && leftHand.size >= 63

    fun hasRightHand(): Boolean = rightHand != null && rightHand.size >= 63

    fun hasFace(): Boolean = (face?.size ?: 0) >= 30  // al menos 10 puntos

    /**
     * Implementa equals/hashCode manualmente porque FloatArray no los tiene.
     * Importante para que Compose detecte cambios estructurales
     * y no redibuje innecesariamente cuando solo cambia el timestamp.
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LandmarkOverlayData) return false
        if (imageWidth != other.imageWidth) return false
        if (imageHeight != other.imageHeight) return false
        if (imageRotationDegrees != other.imageRotationDegrees) return false
        if (timestampMs != other.timestampMs) return false
        if (faceContoursOnly != other.faceContoursOnly) return false
        if (leftHand != null && other.leftHand != null) {
            if (!leftHand.contentEquals(other.leftHand)) return false
        } else if (leftHand != other.leftHand) return false
        if (rightHand != null && other.rightHand != null) {
            if (!rightHand.contentEquals(other.rightHand)) return false
        } else if (rightHand != other.rightHand) return false
        if (face != null && other.face != null) {
            if (!face.contentEquals(other.face)) return false
        } else if (face != other.face) return false
        if (pose != null && other.pose != null) {
            if (!pose.contentEquals(other.pose)) return false
        } else if (pose != other.pose) return false
        return true
    }

    override fun hashCode(): Int {
        var result = leftHand?.contentHashCode() ?: 0
        result = 31 * result + (rightHand?.contentHashCode() ?: 0)
        result = 31 * result + (face?.contentHashCode() ?: 0)
        result = 31 * result + imageWidth
        result = 31 * result + imageHeight
        result = 31 * result + imageRotationDegrees
        result = 31 * result + timestampMs.hashCode()
        result = 31 * result + faceContoursOnly.hashCode()
        result = 31 * result + (pose?.contentHashCode() ?: 0)
        return result
    }
}

/**
 * Estado vacío reutilizable para evitar allocs cuando no se detecta nada.
 * El timestamp 0 marca "frame inválido".
 */
val EmptyLandmarkOverlay = LandmarkOverlayData(
    leftHand = null,
    rightHand = null,
    face = null,
    imageWidth = 0,
    imageHeight = 0,
    timestampMs = 0L,
    imageRotationDegrees = 0,
)
