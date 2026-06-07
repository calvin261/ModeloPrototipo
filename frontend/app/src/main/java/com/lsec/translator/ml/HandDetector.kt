package com.lsec.translator.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.framework.image.BitmapImageBuilder
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Detector de manos usando MediaPipe HandLandmarker.
 * Extrae 21 landmarks 3D normalizados por mano detectada.
 *
 * Ya no requiere YOLO: MediaPipe HandLandmarker detecta manos
 * y extrae landmarks en un solo paso.
 */
@Singleton
class HandDetector @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var handLandmarker: HandLandmarker? = null
    private var timestampMs: Long = 0

    init {
        initializeMediaPipe()
    }

    private fun initializeMediaPipe() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            Log.i(TAG, "MediaPipe HandLandmarker initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe HandLandmarker", e)
        }
    }

    /**
     * Detecta manos en la imagen y extrae landmarks normalizados.
     *
     * Pipeline:
     * 1. Convertir Image a Bitmap
     * 2. MediaPipe HandLandmarker extrae 21 landmarks 3D
     * 3. LandmarkNormalizer normaliza (wrist al origen, escala por MCP9)
     *
     * @return FloatArray(63) normalizado, o null si no se detecto mano
     */
    suspend fun detectAndExtractLandmarks(image: Image): FloatArray? = withContext(Dispatchers.Default) {
        try {
            val bitmap = imageToBitmap(image)
            return@withContext detectFromBitmap(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting hands", e)
            return@withContext null
        }
    }

    /**
     * Detecta landmarks desde un Bitmap directamente.
     * Util para la pantalla de captura de datos.
     */
    fun detectFromBitmap(bitmap: Bitmap): FloatArray? {
        val landmarker = handLandmarker ?: return null

        val mpImage = BitmapImageBuilder(bitmap).build()
        timestampMs += 33 // ~30 fps

        val result = landmarker.detectForVideo(mpImage, timestampMs)

        if (result.landmarks().isEmpty()) {
            return null
        }

        // Extraer los 21 landmarks de la primera mano detectada
        val handLandmarks = result.landmarks()[0]
        val rawLandmarks = FloatArray(63)

        for (i in 0 until 21) {
            val landmark = handLandmarks[i]
            rawLandmarks[i * 3] = landmark.x()
            rawLandmarks[i * 3 + 1] = landmark.y()
            rawLandmarks[i * 3 + 2] = landmark.z()
        }

        // Normalizar usando el mismo algoritmo que Python
        return LandmarkNormalizer.normalize(rawLandmarks)
    }

    /**
     * Convierte Image (YUV_420_888) a Bitmap RGB.
     */
    private fun imageToBitmap(image: Image): Bitmap {
        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            android.graphics.Rect(0, 0, image.width, image.height),
            100,
            out
        )
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Libera recursos del detector.
     */
    fun close() {
        handLandmarker?.close()
    }

    companion object {
        private const val TAG = "HandDetector"
    }
}
