package com.lsec.translator.ui.screens.camera

import android.media.Image
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.domain.usecase.ClassifySignUseCase
import com.lsec.translator.domain.usecase.DetectHandsUseCase
import com.lsec.translator.domain.usecase.SynthesizeSpeechUseCase
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
    val error: String? = null
)

/**
 * ViewModel para la pantalla de cámara.
 * Coordina el pipeline de detección de manos → clasificación → TTS.
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
    private val detectHandsUseCase: DetectHandsUseCase,
    private val classifySignUseCase: ClassifySignUseCase,
    private val synthesizeSpeechUseCase: SynthesizeSpeechUseCase,
    private val userPreferences: UserPreferences
) : ViewModel() {

    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val prefs = userPreferences.preferencesFlow.first()
            _uiState.update { it.copy(useFrontCamera = prefs.useFrontCamera) }
        }
    }

    // Buffer para secuencias temporales (landmarks de múltiples frames)
    private val landmarkBuffer = mutableListOf<FloatArray>()
    private val bufferSize = 30 // ~1 segundo a 30 FPS

    // TTS: mutex garantiza que solo una pronunciación corre a la vez.
    // tryLock() descarta intentos mientras ya se está hablando (no acola).
    private val ttsMutex = Mutex()
    private var lastSignSpoken: String = ""
    private var lastSpeakTime: Long = 0
    private val speakCooldownMs: Long = 3000 // Mínimo entre pronunciaciones (ms)
    
    /**
     * Procesa un frame de la cámara para detección de señas.
     * La detección es síncrona (runBlocking) para que imageProxy permanezca
     * válido durante todo el procesamiento. La clasificación es asíncrona.
     */
    fun processFrame(image: Image?) {
        if (_uiState.value.isProcessing) return
        _uiState.update { it.copy(isProcessing = true) }

        val startTime = System.currentTimeMillis()

        try {
            // Paso 1: Detección síncrona — la imagen debe estar disponible aquí
            val handLandmarks = runBlocking { detectHandsUseCase(image) }
            val handsDetected = handLandmarks != null

            if (handLandmarks != null) {
                landmarkBuffer.add(handLandmarks)
                if (landmarkBuffer.size > bufferSize) {
                    landmarkBuffer.removeAt(0)
                }

                if (landmarkBuffer.size >= 15) {
                    // Paso 2: Clasificación asíncrona (ya no necesita la imagen)
                    val snapshotBuffer = landmarkBuffer.toList()
                    viewModelScope.launch {
                        val result = classifySignUseCase(snapshotBuffer)
                        val latency = System.currentTimeMillis() - startTime
                        _uiState.update {
                            it.copy(
                                isProcessing = false,
                                handsDetected = true,
                                detectedSign = result.signName,
                                ecuadorianTranslation = result.ecuadorianVariant,
                                confidence = result.confidence,
                                latencyMs = latency,
                                error = null
                            )
                        }
                        speakIfNeeded(result)
                    }
                } else {
                    _uiState.update {
                        it.copy(isProcessing = false, handsDetected = true, error = null)
                    }
                }
            } else {
                landmarkBuffer.clear()
                _uiState.update {
                    it.copy(
                        isProcessing = false,
                        handsDetected = false,
                        detectedSign = "",
                        ecuadorianTranslation = "",
                        confidence = 0f,
                        error = null
                    )
                }
            }
        } catch (e: Exception) {
            _uiState.update { it.copy(isProcessing = false, error = e.message) }
        }
    }
    
    /**
     * Sintetiza voz si:
     * - La confianza supera el umbral
     * - No se está pronunciando otra palabra en este momento (tryLock)
     * - La misma seña no se ha pronunciado dentro del cooldown
     *
     * El Mutex (con tryLock) garantiza que las múltiples coroutines de clasificación
     * no acolen llamadas TTS: si ya se está hablando, la nueva petición se descarta.
     * lastSpeakTime se actualiza ANTES de llamar a TTS para que otras coroutines
     * que lleguen entre medias vean el cooldown actualizado de inmediato.
     */
    private suspend fun speakIfNeeded(result: SignClassificationResult) {
        if (!_uiState.value.isTTSEnabled) return
        if (result.confidence < 0.7f) return

        val textToSpeak = result.ecuadorianVariant.ifEmpty { result.signName }
        val now = System.currentTimeMillis()

        // Descartar si es la misma seña dentro del cooldown (fast-path sin lock)
        if (textToSpeak == lastSignSpoken && now - lastSpeakTime < speakCooldownMs) return

        // tryLock: si el mutex ya está tomado, otra pronunciación está en curso → descartar
        if (!ttsMutex.tryLock()) return
        try {
            // Segunda verificación dentro del lock (otro hilo pudo haber ganado)
            val now2 = System.currentTimeMillis()
            if (textToSpeak == lastSignSpoken && now2 - lastSpeakTime < speakCooldownMs) return

            // Registrar ANTES de llamar a TTS para que coroutines concurrentes vean el cooldown
            lastSignSpoken = textToSpeak
            lastSpeakTime = System.currentTimeMillis()

            synthesizeSpeechUseCase(textToSpeak) // suspende hasta que termina de hablar
        } finally {
            ttsMutex.unlock()
        }
    }
    
    /**
     * Alterna el estado del TTS.
     */
    fun toggleTTS() {
        _uiState.update { it.copy(isTTSEnabled = !it.isTTSEnabled) }
    }

    /**
     * Alterna entre cámara frontal y trasera, guardando la preferencia.
     */
    fun toggleCamera() {
        val newValue = !_uiState.value.useFrontCamera
        _uiState.update { it.copy(useFrontCamera = newValue) }
        viewModelScope.launch {
            userPreferences.setUseFrontCamera(newValue)
        }
    }
    
    /**
     * Limpia recursos cuando el ViewModel es destruido.
     */
    override fun onCleared() {
        super.onCleared()
        landmarkBuffer.clear()
    }
}
