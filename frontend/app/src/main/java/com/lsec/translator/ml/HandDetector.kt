package com.lsec.translator.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
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
    private var poseLandmarker: PoseLandmarker? = null
    private var timestampMs: Long = 0

    init {
        initializeMediaPipe()
    }

    private fun initializeMediaPipe() {
        try {
            // Initialize HandLandmarker
            val handBaseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(handBaseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2) // Detect both hands
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.i(TAG, "MediaPipe HandLandmarker initialized")

            // Initialize PoseLandmarker
            val poseBaseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker.task")
                .build()

            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(poseBaseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
            Log.i(TAG, "MediaPipe PoseLandmarker initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe", e)
        }
    }

    /**
     * Detecta manos y cuerpo en la imagen y extrae landmarks normalizados holísticamente.
     *
     * @return FloatArray(147) normalizado holísticamente, o null si no se detectó nada
     */
    suspend fun detectAndExtractLandmarks(image: Image): FloatArray? = withContext(Dispatchers.Default) {
        try {
            val bitmap = imageToBitmap(image)
            return@withContext detectFromBitmap(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting hands/pose", e)
            return@withContext null
        }
    }

    /**
     * Detecta landmarks desde un Bitmap directamente.
     * Extrae 147 valores (2 manos + pose superior).
     */
    fun detectFromBitmap(bitmap: Bitmap): FloatArray? {
        val handDetector = handLandmarker ?: return null
        val poseDetector = poseLandmarker ?: return null

        val mpImage = BitmapImageBuilder(bitmap).build()
        timestampMs += 33 // ~30 fps

        val handResult = handDetector.detectForVideo(mpImage, timestampMs)
        val poseResult = poseDetector.detectForVideo(mpImage, timestampMs)

        val hasHands = handResult != null && handResult.landmarks().isNotEmpty()
        val hasPose = poseResult != null && poseResult.landmarks().isNotEmpty()

        if (!hasHands && !hasPose) {
            return null
        }

        // Map hands by handedness
        var leftHand: FloatArray? = null
        var rightHand: FloatArray? = null

        if (handResult != null && handResult.landmarks().isNotEmpty()) {
            val landmarksList = handResult.landmarks()
            val handednessesList = handResult.handednesses()
            
            for (i in landmarksList.indices) {
                val handLms = landmarksList[i]
                val handednessCategory = handednessesList.getOrNull(i)?.firstOrNull()
                val label = handednessCategory?.categoryName() ?: "Right"
                
                val rawHand = FloatArray(63)
                for (j in 0 until 21) {
                    val lm = handLms[j]
                    rawHand[j * 3] = lm.x()
                    rawHand[j * 3 + 1] = lm.y()
                    rawHand[j * 3 + 2] = lm.z()
                }
                
                if (label.equals("Left", ignoreCase = true)) {
                    leftHand = rawHand
                } else {
                    rightHand = rawHand
                }
            }
        }

        // Map pose landmarks
        var rawPose: FloatArray? = null
        if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
            val poseLms = poseResult.landmarks()[0]
            rawPose = FloatArray(33 * 3)
            for (i in 0 until 33) {
                if (i < poseLms.size) {
                    val lm = poseLms[i]
                    rawPose[i * 3] = lm.x()
                    rawPose[i * 3 + 1] = lm.y()
                    rawPose[i * 3 + 2] = lm.z()
                }
            }
        }

        // Normalizar usando el algoritmo holístico (147 dimensiones)
        return LandmarkNormalizer.normalizeHolistic(leftHand, rightHand, rawPose)
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
        poseLandmarker?.close()
    }

    companion object {
        private const val TAG = "HandDetector"
    }
}
