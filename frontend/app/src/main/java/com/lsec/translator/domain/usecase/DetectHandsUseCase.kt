package com.lsec.translator.domain.usecase

import android.media.Image
import com.lsec.translator.ml.HandDetector
import javax.inject.Inject

/**
 * Caso de uso para detección de manos y extracción de landmarks.
 * Utiliza YOLOv8-nano para ROI y MediaPipe Hands para landmarks.
 */
class DetectHandsUseCase @Inject constructor(
    private val handDetector: HandDetector
) {
    /**
     * Detecta manos en una imagen y retorna los landmarks normalizados.
     * 
     * @param image Frame de la cámara en formato YUV_420_888
     * @return Array de 63 valores (21 landmarks x 3 coordenadas) o null si no hay manos
     */
    suspend operator fun invoke(image: Image?): FloatArray? {
        if (image == null) return null
        return handDetector.detectAndExtractLandmarks(image)
    }
}
