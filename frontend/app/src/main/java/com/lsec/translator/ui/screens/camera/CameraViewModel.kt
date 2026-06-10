package com.lsec.translator.ui.screens.camera

import android.media.Image
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.domain.usecase.ClassifySignUseCase
import com.lsec.translator.domain.usecase.DetectHandsUseCase
import com.lsec.translator.domain.usecase.SynthesizeSpeechUseCase
import com.lsec.translator.ml.DynamicSignClassifier
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
    // === Nuevos campos para la state machine dinámica ===
    val recognitionMode: String = "static",   // "static" o "dynamic"
    val dynamicState: DynamicSignClassifier.State = DynamicSignClassifier.State.IDLE,
    val motionLevel: Float = 0f,               // 0 = quieto, 1 = mucho movimiento
    val bufferFill: Float = 0f,                // 0..1, % del buffer lleno
    val statusMessage: String = "Esperando seña...",
)

/**
 * ViewModel para la pantalla de cámara.
 * Coordina el pipeline de detección de manos → clasificación → TTS.
 *
 * Modos:
 *   - "static":  clasifica cada frame con el MLP (rápido, sin buffer)
 *   - "dynamic": clasifica cuando termina la seña usando la state machine
 *                del DynamicSignClassifier (detección de movimiento→quietud)
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
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
            _uiState.update {
                it.copy(
                    useFrontCamera = prefs.useFrontCamera,
                    recognitionMode = prefs.recognitionMode,
                    isTTSEnabled = prefs.isTTSEnabled,
                )
            }
        }
    }

    // Buffer para modo estático (MLP necesita un solo frame, pero mantenemos
    // los últimos N por si el modo dinámico lo pide).
    private val landmarkBuffer = ArrayDeque<FloatArray>()

    // TTS: mutex garantiza que solo una pronunciación corre a la vez.
    private val ttsMutex = Mutex()
    private var lastSignSpoken: String = ""
    private var lastSpeakTime: Long = 0
    private val speakCooldownMs: Long = 2500

    /**
     * Procesa un frame de la cámara.
     * En modo "static" clasifica inmediatamente.
     * En modo "dynamic" alimenta la state machine que clasifica al final de la seña.
     */
    fun processFrame(image: Image?) {
        if (_uiState.value.isProcessing) return
        _uiState.update { it.copy(isProcessing = true) }

        val startTime = System.currentTimeMillis()
        val mode = _uiState.value.recognitionMode

        try {
            // Paso 1: Detección síncrona
            val handLandmarks = runBlocking { detectHandsUseCase(image) }
            val handsDetected = handLandmarks != null

            when (mode) {
                "dynamic" -> processFrameDynamic(handLandmarks, startTime, handsDetected)
                else -> processFrameStatic(handLandmarks, startTime, handsDetected)
            }
        } catch (e: Throwable) {
            // Catch Throwable (no solo Exception) para que NADA crashee la app.
            // Mostramos el error en la UI en vez de matar el proceso.
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

    /**
     * Modo dinámico: alimenta la state machine del DynamicSignClassifier.
     * La state machine decide cuándo clasificar (al detectar fin de seña).
     */
    private fun processFrameDynamic(
        handLandmarks: FloatArray?,
        startTime: Long,
        handsDetected: Boolean,
    ) {
        // Pasa cada frame al clasificador dinámico
        val result = signClassifier.processDynamicFrame(handLandmarks)
        val stateInfo = signClassifier.dynamicStateInfo()

        val statusMessage = when {
            !handsDetected -> "Acerca una mano a la cámara"
            stateInfo.state == DynamicSignClassifier.State.BUFFERING ->
                "Capturando seña (${stateInfo.bufferSize} frames)"
            stateInfo.state == DynamicSignClassifier.State.COOLDOWN ->
                "Listo, realiza otra seña"
            stateInfo.motionLevel > 0.04f -> "Detectando movimiento..."
            else -> "Mueve la mano para iniciar una seña"
        }

        if (result != null) {
            // Se clasificó una seña: enriquecer y mostrar
            val sign = result
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = handsDetected,
                    detectedSign = sign.signName,
                    ecuadorianTranslation = sign.ecuadorianVariant,
                    confidence = sign.confidence,
                    latencyMs = System.currentTimeMillis() - startTime,
                    dynamicState = stateInfo.state,
                    motionLevel = stateInfo.motionLevel,
                    bufferFill = (stateInfo.bufferSize.toFloat() / 60f).coerceIn(0f, 1f),
                    statusMessage = "Detectado: ${sign.signName} (${(sign.confidence * 100).toInt()}%)",
                    error = null,
                )
            }
            viewModelScope.launch { speakIfNeeded(sign) }
        } else {
            // Solo actualizar indicadores visuales
            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = handsDetected,
                    dynamicState = stateInfo.state,
                    motionLevel = stateInfo.motionLevel,
                    bufferFill = (stateInfo.bufferSize.toFloat() / 60f).coerceIn(0f, 1f),
                    statusMessage = statusMessage,
                    error = null,
                )
            }
        }
    }

    /**
     * Modo estático: clasifica cada frame con el MLP (rápido, sin buffer).
     */
    private fun processFrameStatic(
        handLandmarks: FloatArray?,
        startTime: Long,
        handsDetected: Boolean,
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

    fun toggleTTS() {
        _uiState.update { it.copy(isTTSEnabled = !it.isTTSEnabled) }
        viewModelScope.launch { userPreferences.setTTSEnabled(_uiState.value.isTTSEnabled) }
    }

    fun toggleCamera() {
        val newValue = !_uiState.value.useFrontCamera
        _uiState.update { it.copy(useFrontCamera = newValue) }
        viewModelScope.launch { userPreferences.setUseFrontCamera(newValue) }
    }

    fun setRecognitionMode(mode: String) {
        if (mode !in listOf("static", "dynamic")) return
        _uiState.update { it.copy(recognitionMode = mode) }
        viewModelScope.launch {
            userPreferences.setRecognitionMode(mode)
            // Al cambiar de modo, limpiar buffers
            landmarkBuffer.clear()
            signClassifier.clearDynamicBuffer()
        }
    }

    override fun onCleared() {
        super.onCleared()
        landmarkBuffer.clear()
    }
}
