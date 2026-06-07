package com.lsec.translator.ui.screens.capture

import android.media.Image
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.domain.usecase.DetectHandsUseCase
import com.lsec.translator.domain.usecase.UploadLandmarksUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import javax.inject.Inject

data class DataCaptureUiState(
    val isProcessing: Boolean = false,
    val handsDetected: Boolean = false,
    val selectedLabel: String = "A",
    val captureMode: CaptureMode = CaptureMode.STATIC,
    val capturedSamples: Map<String, Int> = emptyMap(),
    val totalCaptured: Int = 0,
    val isUploading: Boolean = false,
    val uploadMessage: String? = null,
    val isRecording: Boolean = false,
    val recordedFrames: Int = 0,
    val useFrontCamera: Boolean = true,
    val error: String? = null
)

enum class CaptureMode { STATIC, SEQUENCE }

@HiltViewModel
class DataCaptureViewModel @Inject constructor(
    private val detectHandsUseCase: DetectHandsUseCase,
    private val uploadLandmarksUseCase: UploadLandmarksUseCase,
    private val userPreferences: UserPreferences
) : ViewModel() {

    private val _uiState = MutableStateFlow(DataCaptureUiState())
    val uiState: StateFlow<DataCaptureUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val prefs = userPreferences.preferencesFlow.first()
            _uiState.update { it.copy(useFrontCamera = prefs.useFrontCamera) }
        }
    }

    fun toggleCamera() {
        val newValue = !_uiState.value.useFrontCamera
        _uiState.update { it.copy(useFrontCamera = newValue) }
        viewModelScope.launch { userPreferences.setUseFrontCamera(newValue) }
    }

    // Buffer local de muestras capturadas (label -> landmarks)
    private val capturedData = mutableListOf<Pair<String, FloatArray>>()

    // Buffer de frames para secuencias
    private val sequenceBuffer = mutableListOf<FloatArray>()

    // Ultimos landmarks detectados en tiempo real (para captura estatica)
    private var lastDetectedLandmarks: FloatArray? = null

    // Labels disponibles para seleccion
    val staticLabels = ('A'..'Z').map { it.toString() } + listOf("SPACE")
    val dynamicLabels = listOf(
        "hola", "gracias", "por_favor", "si", "no", "ayuda", "agua",
        "comida", "bano", "nombre", "como_estas", "bien", "mal",
        "perdon", "adios"
    )

    /**
     * Procesa un frame para detectar landmarks (preview en tiempo real).
     */
    fun processFrame(image: Image?) {
        if (_uiState.value.isProcessing) return
        _uiState.update { it.copy(isProcessing = true) }

        try {
            val landmarks = runBlocking { detectHandsUseCase(image) }
            val detected = landmarks != null

            if (landmarks != null) {
                lastDetectedLandmarks = landmarks.copyOf()
            } else {
                lastDetectedLandmarks = null
            }

            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = detected,
                    error = null
                )
            }

            // Si estamos grabando secuencia y se detecto mano, agregar al buffer
            if (_uiState.value.isRecording && landmarks != null) {
                sequenceBuffer.add(landmarks.copyOf())
                _uiState.update { it.copy(recordedFrames = sequenceBuffer.size) }
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(isProcessing = false, error = e.message)
            }
        }
    }

    /**
     * Captura una muestra estatica (un frame de landmarks + label).
     */
    fun captureStaticSample() {
        viewModelScope.launch {
            val landmarks = lastDetectedLandmarks?.copyOf() ?: return@launch
            val label = _uiState.value.selectedLabel

            capturedData.add(label to landmarks.copyOf())

            val counts = _uiState.value.capturedSamples.toMutableMap()
            counts[label] = (counts[label] ?: 0) + 1

            _uiState.update {
                it.copy(
                    capturedSamples = counts,
                    totalCaptured = capturedData.size
                )
            }
        }
    }

    /**
     * Inicia grabacion de secuencia.
     */
    fun startRecording() {
        sequenceBuffer.clear()
        _uiState.update { it.copy(isRecording = true, recordedFrames = 0) }
    }

    /**
     * Detiene grabacion de secuencia y guarda.
     */
    fun stopRecording() {
        if (sequenceBuffer.size > 10) {
            // Guardar la secuencia como multiples samples con el mismo label
            val label = _uiState.value.selectedLabel
            for (frame in sequenceBuffer) {
                capturedData.add(label to frame)
            }

            val counts = _uiState.value.capturedSamples.toMutableMap()
            counts[label] = (counts[label] ?: 0) + sequenceBuffer.size

            _uiState.update {
                it.copy(
                    capturedSamples = counts,
                    totalCaptured = capturedData.size
                )
            }
        }

        sequenceBuffer.clear()
        _uiState.update { it.copy(isRecording = false, recordedFrames = 0) }
    }

    fun selectLabel(label: String) {
        _uiState.update { it.copy(selectedLabel = label) }
    }

    fun setCaptureMode(mode: CaptureMode) {
        _uiState.update { it.copy(captureMode = mode) }
    }

    /**
     * Sube todas las muestras capturadas al servidor.
     */
    fun uploadCapturedData() {
        if (capturedData.isEmpty()) return

        viewModelScope.launch {
            _uiState.update { it.copy(isUploading = true, uploadMessage = null) }

            val result = uploadLandmarksUseCase(capturedData.toList())

            result.fold(
                onSuccess = { response ->
                    capturedData.clear()
                    _uiState.update {
                        it.copy(
                            isUploading = false,
                            uploadMessage = "Subidas ${response.samplesAdded} muestras (total: ${response.totalSamples})",
                            capturedSamples = emptyMap(),
                            totalCaptured = 0
                        )
                    }
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(
                            isUploading = false,
                            uploadMessage = "Error: ${error.message}"
                        )
                    }
                }
            )
        }
    }

    fun clearUploadMessage() {
        _uiState.update { it.copy(uploadMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        capturedData.clear()
        sequenceBuffer.clear()
        lastDetectedLandmarks = null
    }
}
