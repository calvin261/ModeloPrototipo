package com.lsec.translator.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.framework.image.BitmapImageBuilder
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Detector de manos + pose + cara usando MediaPipe Tasks Vision.
 *
 * Detecta:
 *   - Hasta 2 manos  (21 landmarks c/u)
 *   - 1 cuerpo completo (33 landmarks; solo usamos 7 para el normalizador)
 *   - 1 cara (478 landmarks, opcional — requiere face_landmarker.task en assets)
 *
 * Salidas:
 *   - [detectAndExtractLandmarks] / [detectFromBitmap]:
 *       pipeline de clasificación (FloatArray 147 dims, back-compat).
 *   - [detectRawLandmarks]: solo para el overlay visual, no clasifica.
 *
 * Si el modelo de cara no está presente en assets/, [detectRawLandmarks]
 * devuelve los landmarks de manos/pose sin error (degradación elegante).
 */
@Singleton
class HandDetector @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private var timestampMs: Long = 0

    /**
     * ¿Está disponible el detector facial?
     * True solo si el modelo se cargó sin error.
     */
    val isFaceLandmarkerReady: Boolean
        get() = faceLandmarker != null

    init {
        initializeMediaPipe()
    }

    private fun initializeMediaPipe() {
        try {
            // ── HandLandmarker ──────────────────────────────────────────────
            val handBaseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(handBaseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.i(TAG, "MediaPipe HandLandmarker initialized")

            // ── PoseLandmarker ──────────────────────────────────────────────
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
            Log.e(TAG, "Error initializing MediaPipe (hands/pose)", e)
        }

        // ── FaceLandmarker (opcional, separado para no romper si falta el .task) ──
        try {
            val faceBaseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()

            val faceOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(faceBaseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                // Blendshapes y mesh refinements off para cuidar batería.
                // Activarlos suma ~3 ms/frame.
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions)
            Log.i(TAG, "MediaPipe FaceLandmarker initialized")
        } catch (e: Exception) {
            // Caso esperado si el archivo face_landmarker.task no está en assets.
            // El overlay funcionará solo con manos/pose.
            Log.w(
                TAG,
                "FaceLandmarker no disponible (¿falta face_landmarker.task en assets?): ${e.message}",
            )
            faceLandmarker = null
        }
    }

    /**
     * Pipeline de clasificación: detecta manos/pose y retorna el vector
     * normalizado de 147 dims listo para el clasificador.
     *
     * El [Image] se decodifica UNA sola vez. Si también quieres los
     * landmarks crudos para el overlay, usa [detectAll].
     *
     * @return FloatArray(147) normalizado, o null si no se detectó nada.
     */
    suspend fun detectAndExtractLandmarks(image: Image): FloatArray? = withContext(Dispatchers.Default) {
        try {
            val bitmap = ImageDecoder.imageToBitmap(image)
            return@withContext detectFromBitmap(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting hands/pose", e)
            return@withContext null
        }
    }

    /**
     * Detecta manos/pose/cara y devuelve TANTO el vector normalizado (para
     * el clasificador) COMO los landmarks crudos (para el overlay).
     *
     * El [Image] se decodifica UNA SOLA VEZ — fundamental porque los
     * ByteBuffers de CameraX son de un solo uso y la segunda decodificación
     * causa SIGSEGV en `Yuv420SpToJpegEncoder::deinterleave`.
     */
    suspend fun detectAll(image: Image, imageRotationDegrees: Int = 0): DetectionResult =
        withContext(Dispatchers.Default) {
            try {
                val rawBitmap = ImageDecoder.imageToBitmap(image)
                // Rotamos el bitmap al espacio de display. Los landmarks
                // que entregue MediaPipe vendrán directamente en coordenadas
                // de display (Portrait), listos para mapear al PreviewView.
                val orientedBitmap = ImageDecoder.rotateBitmap(rawBitmap, imageRotationDegrees)
                val normalized = detectFromBitmap(orientedBitmap)
                val raw = detectRawLandmarksFromBitmap(orientedBitmap, imageRotationDegrees = 0)
                if (orientedBitmap !== rawBitmap) orientedBitmap.recycle()
                rawBitmap.recycle()
                DetectionResult(normalized, raw)
            } catch (e: Exception) {
                Log.e(TAG, "Error in detectAll", e)
                DetectionResult(normalized = null, raw = EmptyLandmarkOverlay)
            }
        }

    /**
     * Versión síncrona que recibe el bitmap ya convertido.
     * Usado por [com.lsec.translator.data.upload.NpyWriter] y tests.
     */
    fun detectFromBitmap(bitmap: Bitmap): FloatArray? {
        val handDetector = handLandmarker ?: return null
        val poseDetector = poseLandmarker ?: return null

        val mpImage = BitmapImageBuilder(bitmap).build()
        timestampMs += 33

        val handResult = handDetector.detectForVideo(mpImage, timestampMs)
        val poseResult = poseDetector.detectForVideo(mpImage, timestampMs)

        val hasHands = handResult != null && handResult.landmarks().isNotEmpty()
        val hasPose = poseResult != null && poseResult.landmarks().isNotEmpty()

        if (!hasHands && !hasPose) {
            return null
        }

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

        return LandmarkNormalizer.normalizeHolistic(leftHand, rightHand, rawPose)
    }

    /**
     * Wrapper del resultado de [detectAll] que entrega ambos formatos en una
     * sola llamada, evitando decodificar el [Image] dos veces.
     */
    data class DetectionResult(
        /** Vector normalizado de 147 dims para el clasificador, o null si nada detectado. */
        val normalized: FloatArray?,
        /** Snapshot crudo para el overlay visual. */
        val raw: LandmarkOverlayData,
    )

    /**
     * Detecta SOLO los landmarks crudos (sin normalizar) para pintar el overlay.
     *
     * Más barato que [detectAndExtractLandmarks] porque no construye el
     * FloatArray(147) ni llama a [LandmarkNormalizer].
     *
     * El bitmap debe estar en orientación de display (lo entrega
     * CameraX con `targetRotation` por defecto).
     *
     * @return [LandmarkOverlayData] con coordenadas normalizadas [0,1] en el
     *         espacio del bitmap, o un objeto "vacío" si no se detectó nada.
     */
    suspend fun detectRawLandmarks(image: Image, imageRotationDegrees: Int = 0): LandmarkOverlayData =
        withContext(Dispatchers.Default) {
            try {
                val rawBitmap = ImageDecoder.imageToBitmap(image)
                val orientedBitmap = ImageDecoder.rotateBitmap(rawBitmap, imageRotationDegrees)
                val result = detectRawLandmarksFromBitmap(orientedBitmap, imageRotationDegrees = 0)
                if (orientedBitmap !== rawBitmap) orientedBitmap.recycle()
                rawBitmap.recycle()
                return@withContext result
            } catch (e: Exception) {
                Log.e(TAG, "Error detecting raw landmarks", e)
                return@withContext EmptyLandmarkOverlay
            }
        }

    /**
     * Versión síncrona que ya recibe el bitmap.
     *
     * @param imageRotationDegrees rotación del bitmap vs display, propagada al
     *        overlay para que los landmarks se alineen con la imagen en pantalla.
     */
    fun detectRawLandmarksFromBitmap(
        bitmap: Bitmap,
        imageRotationDegrees: Int = 0,
    ): LandmarkOverlayData {
        val handDetector = handLandmarker
        val poseDetector = poseLandmarker
        val faceDetector = faceLandmarker

        // Si ni siquiera hay manos/pose inicializadas, devolvemos vacío.
        if (handDetector == null && poseDetector == null && faceDetector == null) {
            return EmptyLandmarkOverlay
        }

        val mpImage = BitmapImageBuilder(bitmap).build()
        timestampMs += 33
        val ts = timestampMs

        var leftHand: FloatArray? = null
        var rightHand: FloatArray? = null

        // ── Manos ──
        handDetector?.let { det ->
            try {
                val handResult = det.detectForVideo(mpImage, ts)
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
            } catch (e: Exception) {
                Log.w(TAG, "HandLandmarker falló en este frame", e)
            }
        }

        // ── Cara (opcional) ──
        var face: FloatArray? = null
        faceDetector?.let { det ->
            try {
                val faceResult = det.detectForVideo(mpImage, ts)
                if (faceResult != null && faceResult.faceLandmarks().isNotEmpty()) {
                    val faceLms = faceResult.faceLandmarks()[0]
                    val n = faceLms.size
                    face = FloatArray(n * 3)
                    for (i in 0 until n) {
                        val lm = faceLms[i]
                        face[i * 3] = lm.x()
                        face[i * 3 + 1] = lm.y()
                        face[i * 3 + 2] = lm.z()
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "FaceLandmarker falló en este frame", e)
            }
        }

        // Suprimimos warnings de poseDetector (de momento no se usa en overlay,
        // pero conservamos la referencia por si se añaden huesos después).
        @Suppress("UNUSED_VARIABLE")
        val _pose = poseDetector

        return LandmarkOverlayData(
            leftHand = leftHand,
            rightHand = rightHand,
            face = face,
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
            timestampMs = ts,
            imageRotationDegrees = imageRotationDegrees,
        )
    }

    /**
     * Libera recursos del detector.
     */
    fun close() {
        try { handLandmarker?.close() } catch (_: Throwable) {}
        try { poseLandmarker?.close() } catch (_: Throwable) {}
        try { faceLandmarker?.close() } catch (_: Throwable) {}
    }

    companion object {
        private const val TAG = "HandDetector"
    }
}