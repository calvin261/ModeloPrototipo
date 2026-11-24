package com.lsec.translator.ui.screens.camera

import android.media.Image
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.domain.usecase.ClassifySignUseCase
import com.lsec.translator.domain.usecase.DetectHandsUseCase
import com.lsec.translator.domain.usecase.SynthesizeSpeechUseCase
import com.lsec.translator.ml.DynamicSignClassifier
import com.lsec.translator.ml.EmptyLandmarkOverlay
import com.lsec.translator.ml.GestureCommandClassifier
import com.lsec.translator.ml.HandDetector
import com.lsec.translator.ml.LandmarkOverlayData
import com.lsec.translator.ml.SignClassificationResult
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import javax.inject.Inject

/**
 * Estado de la UI de la pantalla de cámara.
 *
 * Campos del overlay visual:
 *   - [landmarkOverlay]  : snapshot del último frame (manos + cara).
 *                          Se reemplaza atómicamente para que Compose no
 *                          redibuje con datos parciales.
 *   - [landmarkOverlayEnabled] : toggle local (rápido desde la toolbar)
 *                          multiplicado por la preferencia persistente
 *                          del modo actual (static/dynamic).
 *   - [landmarkOverlayReasonOff] : motivo por el que el overlay está oculto
 *                          (para mostrarlo como hint en la UI).
 */
data class CameraUiState(
    val isProcessing: Boolean = false,
    val handsDetected: Boolean = false,
    val detectedSign: String = "",
    val ecuadorianTranslation: String = "",
    val confidence: Float = 0f,
    val latencyMs: Long = 0,
    val isTTSEnabled: Boolean = true,
    val useFrontCamera: Boolean = true,
    val error: String? = null,
    // === State machine dinámica ===
    val recognitionMode: String = "static",
    val dynamicState: DynamicSignClassifier.State = DynamicSignClassifier.State.IDLE,
    val motionLevel: Float = 0f,
    val bufferFill: Float = 0f,
    val statusMessage: String = "Esperando seña...",

    // === Overlay de landmarks ===
    /** Último snapshot válido. Reemplazado de forma atómica (nunca a null). */
    val landmarkOverlay: LandmarkOverlayData = EmptyLandmarkOverlay,
    /** ¿Está visible el overlay ahora mismo? (combina pref + toggle rápido). */
    val landmarkOverlayVisible: Boolean = false,
    /** ¿El detector facial está disponible? (face_landmarker.task cargado). */
    val faceLandmarkerAvailable: Boolean = false,
    val gestureTrail: List<GestureCommandClassifier.Point> = emptyList(),
)

/**
 * ViewModel para la pantalla de cámara.
 *
 * Responsabilidades nuevas vs versión previa:
 *   1. Mantener un [landmarkOverlay] actualizado con el último frame.
 *   2. Combinar preferencia persistente + toggle rápido para decidir
 *      visibilidad.
 *   3. Exponer [faceLandmarkerAvailable] para que la UI pueda avisar
 *      al usuario si el modelo facial no está cargado.
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
    private val handDetector: HandDetector,
    private val detectHandsUseCase: DetectHandsUseCase,
    private val classifySignUseCase: ClassifySignUseCase,
    private val synthesizeSpeechUseCase: SynthesizeSpeechUseCase,
    private val userPreferences: UserPreferences,
    private val signClassifier: com.lsec.translator.ml.SignClassifier,
) : ViewModel() {

    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val prefs = userPreferences.preferencesFlow.first()
            val visibleForMode = prefs.recognitionMode == "dynamic" || isOverlayEnabledForMode(
                mode = prefs.recognitionMode,
                prefs = prefs,
            )
            _uiState.update {
                it.copy(
                    useFrontCamera = prefs.useFrontCamera,
                    recognitionMode = prefs.recognitionMode,
                    isTTSEnabled = prefs.isTTSEnabled,
                    faceLandmarkerAvailable = handDetector.isFaceLandmarkerReady,
                    landmarkOverlayVisible = visibleForMode,
                )
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    //  Buffer de landmarks recientes + TTS
    // ───────────────────────────────────────────────────────────────────────

    private val landmarkBuffer = ArrayDeque<FloatArray>()
    private val gestureClassifier = GestureCommandClassifier()
    private val ttsMutex = Mutex()
    private var lastSignSpoken: String = ""
    private var lastSpeakTime: Long = 0
    private val speakCooldownMs: Long = 2500

    // ───────────────────────────────────────────────────────────────────────
    //  Procesamiento de frame
    // ───────────────────────────────────────────────────────────────────────

    fun processFrame(image: Image?, imageRotationDegrees: Int = 0) {
        if (_uiState.value.isProcessing) return
        _uiState.update { it.copy(isProcessing = true) }

        val startTime = System.currentTimeMillis()
        val mode = _uiState.value.recognitionMode

        try {
            // Paso 1: Detección única (clasificación + overlay en una sola
            //   decodificación del Image). Importante: CameraX entrega cada
            //   Image una sola vez; si lo decodificamos 2 veces crashea
            //   con SIGSEGV en Yuv420SpToJpegEncoder.
            //
            //   imageRotationDegrees se propaga al overlay para alinear los
            //   landmarks con lo que el usuario ve en el PreviewView.
            val detection = if (image != null) {
                runBlocking { handDetector.detectAll(image, imageRotationDegrees) }
            } else {
                HandDetector.DetectionResult(normalized = null, raw = EmptyLandmarkOverlay)
            }
            val handLandmarks = detection.normalized
            val overlayData = detection.raw
            val handsDetected = handLandmarks != null

            when (mode) {
                "dynamic" -> processFrameDynamic(handLandmarks, startTime, handsDetected, overlayData)
                else -> processFrameStatic(handLandmarks, startTime, handsDetected, overlayData)
            }
        } catch (e: Throwable) {
            android.util.Log.e("CameraViewModel", "processFrame crashed", e)
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    error = "${e.javaClass.simpleName}: ${e.message ?: "error desconocido"}",
                    statusMessage = "Error en el pipeline",
                )
            }
        }
    }

    private fun processFrameDynamic(
        handLandmarks: FloatArray?,
        startTime: Long,
        handsDetected: Boolean,
        overlayData: LandmarkOverlayData,
    ) {
        val gesture = gestureClassifier.process(overlayData)
        val result = gesture.result
        val trail = toCameraTrail(gesture.trail, overlayData)
        val dynamicState = when (gesture.state) {
            GestureCommandClassifier.State.CAPTURING -> DynamicSignClassifier.State.BUFFERING
            GestureCommandClassifier.State.COOLDOWN -> DynamicSignClassifier.State.COOLDOWN
            else -> DynamicSignClassifier.State.IDLE
        }

        if (result != null) {
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = handsDetected,
                    detectedSign = result.signName,
                    ecuadorianTranslation = result.ecuadorianVariant,
                    confidence = result.confidence,
                    latencyMs = System.currentTimeMillis() - startTime,
                    dynamicState = dynamicState,
                    motionLevel = (trail.size / GestureCommandClassifier.MAX_SAMPLES.toFloat()).coerceIn(0f, 1f),
                    bufferFill = (trail.size / GestureCommandClassifier.MAX_SAMPLES.toFloat()).coerceIn(0f, 1f),
                    statusMessage = "Detectado: ${result.signName} (${(result.confidence * 100).toInt()}%)",
                    error = null,
                    landmarkOverlay = overlayData,
                    gestureTrail = trail,
                )
            }
            viewModelScope.launch { speakIfNeeded(result) }
        } else {
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = handsDetected,
                    dynamicState = dynamicState,
                    motionLevel = (trail.size / GestureCommandClassifier.MAX_SAMPLES.toFloat()).coerceIn(0f, 1f),
                    bufferFill = (trail.size / GestureCommandClassifier.MAX_SAMPLES.toFloat()).coerceIn(0f, 1f),
                    statusMessage = gesture.status,
                    error = null,
                    landmarkOverlay = overlayData,
                    gestureTrail = trail,
                )
            }
        }
    }

    private fun toCameraTrail(
        trail: List<GestureCommandClassifier.Point>,
        overlay: LandmarkOverlayData,
    ): List<GestureCommandClassifier.Point> {
        val face = overlay.face ?: return emptyList()
        if (face.size < 264 * 3) return emptyList()
        fun point(index: Int) = GestureCommandClassifier.Point(face[index * 3], face[index * 3 + 1])
        val nose = point(1)
        val leftEye = point(33)
        val rightEye = point(263)
        val scale = kotlin.math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y)
        if (scale < 0.01f) return emptyList()
        return trail.map { GestureCommandClassifier.Point(nose.x + it.x * scale, nose.y + it.y * scale) }
    }

    private fun processFrameStatic(
        handLandmarks: FloatArray?,
        startTime: Long,
        handsDetected: Boolean,
        overlayData: LandmarkOverlayData,
    ) {
        if (handLandmarks != null) {
            landmarkBuffer.addLast(handLandmarks)
            if (landmarkBuffer.size > 30) landmarkBuffer.removeFirst()

            if (landmarkBuffer.size >= 1) {
                val snapshot = landmarkBuffer.toList()
                viewModelScope.launch {
                    val result = classifySignUseCase(snapshot)
                    val latency = System.currentTimeMillis() - startTime
                    _uiState.update {
                        it.copy(
                            isProcessing = false,
                            handsDetected = true,
                            detectedSign = result.signName,
                            ecuadorianTranslation = result.ecuadorianVariant,
                            confidence = result.confidence,
                            latencyMs = latency,
                            statusMessage = "Detectado: ${result.signName}",
                            error = null,
                            landmarkOverlay = overlayData,
                        )
                    }
                    speakIfNeeded(result)
                }
                return
            }
        } else {
            landmarkBuffer.clear()
        }
        _uiState.update {
            it.copy(
                isProcessing = false,
                handsDetected = handsDetected,
                statusMessage = if (handsDetected) "Detectando..." else "Acerca una mano a la cámara",
                error = null,
                landmarkOverlay = overlayData,
            )
        }
    }

    private suspend fun speakIfNeeded(result: SignClassificationResult) {
        if (!_uiState.value.isTTSEnabled) return
        if (result.confidence < 0.7f) return

        val textToSpeak = result.ecuadorianVariant.ifEmpty { result.signName }
        val now = System.currentTimeMillis()

        if (textToSpeak == lastSignSpoken && now - lastSpeakTime < speakCooldownMs) return
        if (!ttsMutex.tryLock()) return
        try {
            val now2 = System.currentTimeMillis()
            if (textToSpeak == lastSignSpoken && now2 - lastSpeakTime < speakCooldownMs) return
            lastSignSpoken = textToSpeak
            lastSpeakTime = System.currentTimeMillis()
            synthesizeSpeechUseCase(textToSpeak)
        } finally {
            ttsMutex.unlock()
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    //  Toggles rápidos y sincronización con preferencias
    // ───────────────────────────────────────────────────────────────────────

    fun toggleTTS() {
        _uiState.update { it.copy(isTTSEnabled = !it.isTTSEnabled) }
        viewModelScope.launch { userPreferences.setTTSEnabled(_uiState.value.isTTSEnabled) }
    }

    fun toggleCamera() {
        val newValue = !_uiState.value.useFrontCamera
        _uiState.update { it.copy(useFrontCamera = newValue) }
        viewModelScope.launch { userPreferences.setUseFrontCamera(newValue) }
    }

    /**
     * Toggle rápido desde la toolbar: invierte el estado visible del overlay
     * para el modo de reconocimiento actual, y lo persiste en DataStore.
     *
     * UX: el cambio es inmediato en pantalla (latency < 16 ms) y se guarda
     * para la próxima vez que abras la cámara en ese modo.
     */
    fun toggleLandmarkOverlay() {
        val current = _uiState.value
        val newVisible = !current.landmarkOverlayVisible
        _uiState.update { it.copy(landmarkOverlayVisible = newVisible) }
        viewModelScope.launch {
            when (current.recognitionMode) {
                "dynamic" -> userPreferences.setShowLandmarksDynamic(newVisible)
                else -> userPreferences.setShowLandmarksStatic(newVisible)
            }
        }
    }

    fun setRecognitionMode(mode: String) {
        if (mode !in listOf("static", "dynamic")) return
        val oldMode = _uiState.value.recognitionMode
        _uiState.update { it.copy(recognitionMode = mode) }
        viewModelScope.launch {
            userPreferences.setRecognitionMode(mode)
            landmarkBuffer.clear()
            signClassifier.clearDynamicBuffer()
            gestureClassifier.clear()
            // Re-leer la pref correspondiente al nuevo modo para el overlay
            val prefs = userPreferences.preferencesFlow.first()
            val visible = isOverlayEnabledForMode(mode, prefs)
            _uiState.update { it.copy(landmarkOverlayVisible = visible) }
        }
    }

    /**
     * Recalcula la visibilidad del overlay a partir de las prefs actuales.
     * Llamado después de que el usuario cambia los toggles en Settings.
     */
    fun refreshOverlayVisibility() {
        viewModelScope.launch {
            val prefs = userPreferences.preferencesFlow.first()
            val visible = isOverlayEnabledForMode(_uiState.value.recognitionMode, prefs)
            _uiState.update { it.copy(landmarkOverlayVisible = visible) }
        }
    }

    private fun isOverlayEnabledForMode(
        mode: String,
        prefs: com.lsec.translator.data.preferences.UserPreferencesData,
    ): Boolean = when (mode) {
        "dynamic" -> prefs.showLandmarksDynamic
        else -> prefs.showLandmarksStatic
    }

    override fun onCleared() {
        super.onCleared()
        landmarkBuffer.clear()
    }
}
