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
    val capturedStatic: Map<String, Int> = emptyMap(),
    val capturedSequences: Map<String, Int> = emptyMap(),
    val totalStatic: Int = 0,
    val totalSequences: Int = 0,
    val isUploading: Boolean = false,
    val uploadMessage: String? = null,
    val isRecording: Boolean = false,
    val recordedFrames: Int = 0,
    val recordingTargetFrames: Int = DEFAULT_SEQUENCE_TARGET_FRAMES,
    val recordingMinFrames: Int = DEFAULT_SEQUENCE_MIN_FRAMES,
    val useFrontCamera: Boolean = true,
    val error: String? = null
) {
    val hasContent: Boolean get() = totalStatic > 0 || totalSequences > 0
    val recordingProgress: Float
        get() = (recordedFrames.toFloat() / recordingTargetFrames.coerceAtLeast(1))
            .coerceIn(0f, 1f)
    val canStopRecording: Boolean get() = recordedFrames >= recordingMinFrames
}

enum class CaptureMode { STATIC, SEQUENCE }

private const val DEFAULT_SEQUENCE_MIN_FRAMES = 15
private const val DEFAULT_SEQUENCE_TARGET_FRAMES = 45
private const val DEFAULT_SEQUENCE_MAX_FRAMES = 120

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

    // Buffer local de muestras estáticas (label -> landmarks)
    private val staticSamples = mutableListOf<Pair<String, FloatArray>>()

    // Buffer local de secuencias (label -> lista de frames de 63 floats)
    private val sequenceSamples = mutableListOf<Pair<String, List<FloatArray>>>()

    // Buffer de frames durante grabación activa
    private val sequenceBuffer = mutableListOf<FloatArray>()

    // Últimos landmarks detectados en tiempo real (para captura estática)
    private var lastDetectedLandmarks: FloatArray? = null

    val staticLabels = ('A'..'Z').map { it.toString() } + listOf("SPACE")
    val dynamicLabels = listOf(
        "hola", "gracias", "por_favor", "si", "no", "ayuda", "agua",
        "comida", "bano", "nombre", "como_estas", "bien", "mal",
        "perdon", "adios"
    )

    /**
     * Procesa un frame para detectar landmarks (preview en tiempo real).
     * Si estamos grabando una secuencia, agrega el frame al buffer activo.
     * Si alcanzamos el máximo, auto-detener la grabación.
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

            var autoStop = false
            if (_uiState.value.isRecording && landmarks != null) {
                sequenceBuffer.add(landmarks.copyOf())
                if (sequenceBuffer.size >= DEFAULT_SEQUENCE_MAX_FRAMES) {
                    autoStop = true
                }
            }

            _uiState.update {
                it.copy(
                    isProcessing = false,
                    handsDetected = detected,
                    recordedFrames = sequenceBuffer.size,
                    error = null
                )
            }

            if (autoStop) {
                stopRecording()
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(isProcessing = false, error = e.message)
            }
        }
    }

    fun captureStaticSample() {
        viewModelScope.launch {
            val landmarks = lastDetectedLandmarks?.copyOf() ?: return@launch
            val label = _uiState.value.selectedLabel

            staticSamples.add(label to landmarks.copyOf())
            val counts = _uiState.value.capturedStatic.toMutableMap()
            counts[label] = (counts[label] ?: 0) + 1
            _uiState.update {
                it.copy(
                    capturedStatic = counts,
                    totalStatic = staticSamples.size
                )
            }
        }
    }

    fun startRecording() {
        if (_uiState.value.isRecording) return
        sequenceBuffer.clear()
        _uiState.update {
            it.copy(
                isRecording = true,
                recordedFrames = 0,
                error = null
            )
        }
    }

    /**
     * Detiene la grabación y guarda el buffer como UNA secuencia (no múltiples
     * samples estáticos). Antes del fix (F7-validacion 2026-06-23) este método
     * aplanaba cada frame en un sample independiente con el mismo label, lo que
     * destruía la estructura temporal y hacía imposible entrenar el modelo
     * dinámico.
     */
    fun stopRecording() {
        val frames = sequenceBuffer.toList()
        sequenceBuffer.clear()

        if (frames.size >= _uiState.value.recordingMinFrames) {
            val label = _uiState.value.selectedLabel
            sequenceSamples.add(label to frames)

            val counts = _uiState.value.capturedSequences.toMutableMap()
            counts[label] = (counts[label] ?: 0) + 1
            _uiState.update {
                it.copy(
                    isRecording = false,
                    recordedFrames = 0,
                    capturedSequences = counts,
                    totalSequences = sequenceSamples.size
                )
            }
        } else {
            _uiState.update {
                it.copy(
                    isRecording = false,
                    recordedFrames = 0,
                    error = "Demasiado corta (${frames.size} frames, mínimo ${_uiState.value.recordingMinFrames})"
                )
            }
        }
    }

    fun selectLabel(label: String) {
        _uiState.update { it.copy(selectedLabel = label) }
    }

    fun setCaptureMode(mode: CaptureMode) {
        val defaultLabel = if (mode == CaptureMode.SEQUENCE) dynamicLabels.first() else "A"
        _uiState.update {
            it.copy(
                captureMode = mode,
                // Si el label seleccionado no existe en el vocabulario del modo,
                // resetear al primero disponible para evitar subir con label
                // equivocado (ej. "A" cuando se capturan secuencias dinámicas).
                selectedLabel = if (it.selectedLabel in when (mode) {
                    CaptureMode.STATIC -> staticLabels
                    CaptureMode.SEQUENCE -> dynamicLabels
                }) it.selectedLabel else defaultLabel
            )
        }
    }

    fun discardLastSequence() {
        if (sequenceSamples.isEmpty()) return
        val removed = sequenceSamples.removeLast()
        val counts = _uiState.value.capturedSequences.toMutableMap()
        counts[removed.first] = (counts[removed.first] ?: 1) - 1
        if ((counts[removed.first] ?: 0) <= 0) counts.remove(removed.first)
        _uiState.update {
            it.copy(
                capturedSequences = counts,
                totalSequences = sequenceSamples.size
            )
        }
    }

    fun discardLastStatic() {
        if (staticSamples.isEmpty()) return
        val removed = staticSamples.removeLast()
        val counts = _uiState.value.capturedStatic.toMutableMap()
        counts[removed.first] = (counts[removed.first] ?: 1) - 1
        if ((counts[removed.first] ?: 0) <= 0) counts.remove(removed.first)
        _uiState.update {
            it.copy(
                capturedStatic = counts,
                totalStatic = staticSamples.size
            )
        }
    }

    /**
     * Sube ambos buffers al backend usando endpoints separados:
     * - estáticas → /api/v1/data/upload/static
     * - secuencias → /api/v1/data/upload/sequence (multipart .npy)
     */
    fun uploadCapturedData() {
        if (staticSamples.isEmpty() && sequenceSamples.isEmpty()) return

        viewModelScope.launch {
            _uiState.update { it.copy(isUploading = true, uploadMessage = null) }

            val staticSnapshot = staticSamples.toList()
            val seqSnapshot = sequenceSamples.toList()

            val result = uploadLandmarksUseCase.uploadAll(
                staticSamples = staticSnapshot,
                sequences = seqSnapshot
            )

            result.fold(
                onSuccess = { summary ->
                    staticSamples.clear()
                    sequenceSamples.clear()
                    val msg = buildString {
                        if (summary.staticAdded > 0) {
                            append("Estáticas +${summary.staticAdded} (total ${summary.staticTotal}). ")
                        }
                        if (summary.sequencesUploaded > 0) {
                            append("Secuencias +${summary.sequencesUploaded} (")
                            append(summary.sequenceFramesByLabel.entries.joinToString(", ") { "${it.key}: ${it.value} frames" })
                            append(").")
                        }
                        if (summary.staticAdded == 0 && summary.sequencesUploaded == 0) {
                            append("Nada que subir.")
                        }
                    }
                    _uiState.update {
                        it.copy(
                            isUploading = false,
                            uploadMessage = msg.trim(),
                            capturedStatic = emptyMap(),
                            capturedSequences = emptyMap(),
                            totalStatic = 0,
                            totalSequences = 0
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
        staticSamples.clear()
        sequenceSamples.clear()
        sequenceBuffer.clear()
        lastDetectedLandmarks = null
    }
}