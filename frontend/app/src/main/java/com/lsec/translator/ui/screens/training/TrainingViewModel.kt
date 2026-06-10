package com.lsec.translator.ui.screens.training

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.data.preferences.UserPreferencesData
import com.lsec.translator.data.remote.DataStatsResponseDto
import com.lsec.translator.data.remote.ModelInfoDto
import com.lsec.translator.data.remote.TrainingHistoryEntryDto
import com.lsec.translator.data.repository.ModelRepository
import com.lsec.translator.data.repository.TrainingRepository
import com.lsec.translator.domain.usecase.CheckTrainingStatusUseCase
import com.lsec.translator.domain.usecase.DownloadModelUseCase
import com.lsec.translator.domain.usecase.StartTrainingUseCase
import com.lsec.translator.ml.SignClassifier
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class TrainingUiState(
    val isConnected: Boolean = false,
    val activeTaskId: String? = null,
    val activeModelType: String? = null,
    val trainingStatus: String = "idle",
    val epoch: Int = 0,
    val totalEpochs: Int = 0,
    val trainLoss: Float = 0f,
    val valAccuracy: Float = 0f,
    val bestAccuracy: Float = 0f,
    val error: String? = null,
    val history: List<TrainingHistoryEntryDto> = emptyList(),
    val dataStats: DataStatsResponseDto? = null,
    val isDownloading: Boolean = false,
    val downloadingType: String? = null,    // "classifier" | "sequence" | null
    val downloadMessage: String? = null,
    // === Nuevos campos para la sección de modelos ===
    val serverModels: List<ModelInfoDto> = emptyList(),
    val activeClassifierVersion: String = "bundled",
    val activeSequenceVersion: String = "bundled",
    val hasLocalClassifier: Boolean = false,
    val hasLocalSequence: Boolean = false,
)

@HiltViewModel
class TrainingViewModel @Inject constructor(
    private val startTrainingUseCase: StartTrainingUseCase,
    private val checkTrainingStatusUseCase: CheckTrainingStatusUseCase,
    private val downloadModelUseCase: DownloadModelUseCase,
    private val trainingRepository: TrainingRepository,
    private val modelRepository: ModelRepository,
    private val userPreferences: UserPreferences,
    private val signClassifier: SignClassifier,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null

    init {
        checkConnection()
    }

    private fun checkConnection() {
        viewModelScope.launch {
            val statsResult = trainingRepository.getDataStats()
            statsResult.fold(
                onSuccess = { stats ->
                    _uiState.update { it.copy(isConnected = true, dataStats = stats) }
                    loadHistory()
                    loadAvailableModels()
                    loadActiveVersions()
                },
                onFailure = {
                    _uiState.update { it.copy(isConnected = false, error = "No se pudo conectar al servidor") }
                }
            )
        }
    }

    private fun loadHistory() {
        viewModelScope.launch {
            val result = trainingRepository.getTrainingHistory()
            result.onSuccess { history ->
                _uiState.update { it.copy(history = history) }
            }
        }
    }

    /**
     * Carga la lista de modelos disponibles del backend.
     */
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val result = modelRepository.listAvailableModels()
            result.onSuccess { models ->
                _uiState.update { it.copy(serverModels = models) }
            }
        }
    }

    /**
     * Lee las versiones locales activas (guardadas en DataStore) y si los
     * modelos están descargados físicamente.
     */
    private fun loadActiveVersions() {
        viewModelScope.launch {
            val prefs: UserPreferencesData = userPreferences.preferencesFlow.first()
            val hasClassifier = modelRepository.hasLocalModel("classifier")
            val hasSequence = modelRepository.hasLocalModel("sequence")
            _uiState.update {
                it.copy(
                    activeClassifierVersion = prefs.classifierModelVersion,
                    activeSequenceVersion = prefs.sequenceModelVersion,
                    hasLocalClassifier = hasClassifier,
                    hasLocalSequence = hasSequence,
                )
            }
        }
    }

    fun startClassifierTraining(epochs: Int = 50) {
        viewModelScope.launch {
            _uiState.update { it.copy(trainingStatus = "pending", error = null, activeModelType = "classifier") }
            val result = startTrainingUseCase.classifier(epochs = epochs)
            result.fold(
                onSuccess = { response ->
                    _uiState.update {
                        it.copy(activeTaskId = response.taskId, trainingStatus = "running", totalEpochs = epochs)
                    }
                    startPolling(response.taskId)
                },
                onFailure = { error ->
                    _uiState.update { it.copy(trainingStatus = "failed", error = error.message) }
                }
            )
        }
    }

    fun startSequenceTraining(epochs: Int = 80) {
        viewModelScope.launch {
            _uiState.update { it.copy(trainingStatus = "pending", error = null, activeModelType = "sequence") }
            val result = startTrainingUseCase.sequence(epochs = epochs)
            result.fold(
                onSuccess = { response ->
                    _uiState.update {
                        it.copy(activeTaskId = response.taskId, trainingStatus = "running", totalEpochs = epochs)
                    }
                    startPolling(response.taskId)
                },
                onFailure = { error ->
                    _uiState.update { it.copy(trainingStatus = "failed", error = error.message) }
                }
            )
        }
    }

    private fun startPolling(taskId: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (true) {
                delay(3000)
                val result = checkTrainingStatusUseCase(taskId)
                result.fold(
                    onSuccess = { status ->
                        _uiState.update {
                            it.copy(
                                trainingStatus = status.status,
                                epoch = status.epoch,
                                totalEpochs = status.totalEpochs,
                                trainLoss = status.trainLoss,
                                valAccuracy = status.valAccuracy,
                                bestAccuracy = status.bestAccuracy,
                                error = status.error,
                            )
                        }
                        if (status.status == "completed" || status.status == "failed") {
                            loadHistory()
                            loadAvailableModels() // puede haber nuevas versiones
                            return@launch
                        }
                    },
                    onFailure = { error ->
                        _uiState.update { it.copy(error = "Error consultando estado: ${error.message}") }
                    }
                )
            }
        }
    }

    /**
     * Descarga la version mas reciente del modelo solicitado.
     */
    fun downloadLatest(modelType: String) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isDownloading = true, downloadingType = modelType, downloadMessage = "Buscando última versión...")
            }
            val models = _uiState.value.serverModels
            val latest = models
                .filter { m -> m.name == modelType }
                .maxByOrNull { m -> m.createdAt }
            if (latest == null) {
                _uiState.update {
                    it.copy(
                        isDownloading = false,
                        downloadingType = null,
                        downloadMessage = "No hay versiones disponibles en el servidor",
                    )
                }
                return@launch
            }
            performDownload(modelType, latest.version)
        }
    }

    /**
     * Descarga una version especifica.
     */
    fun downloadVersion(modelType: String, version: String) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(isDownloading = true, downloadingType = modelType, downloadMessage = "Descargando $version...")
            }
            performDownload(modelType, version)
        }
    }

    private suspend fun performDownload(modelType: String, version: String) {
        val success = downloadModelUseCase(modelType, version)
        val finalMessage = if (success) {
            // Refrescar versiones locales
            when (modelType) {
                "classifier" -> userPreferences.setClassifierModelVersion(version)
                "sequence" -> userPreferences.setSequenceModelVersion(version)
            }
            loadActiveVersions()
            "Modelo $modelType $version descargado y activado"
        } else {
            "Error al descargar $version. Verifica que el servidor está activo."
        }
        _uiState.update {
            it.copy(
                isDownloading = false,
                downloadingType = null,
                downloadMessage = finalMessage,
            )
        }
    }

    fun deleteStaticLabel(label: String) {
        viewModelScope.launch {
            trainingRepository.deleteStaticLabel(label).fold(
                onSuccess = { refreshStats() },
                onFailure = { e ->
                    _uiState.update { it.copy(error = "Error eliminando \"$label\": ${e.message}") }
                }
            )
        }
    }

    fun deleteSequenceLabel(label: String) {
        viewModelScope.launch {
            trainingRepository.deleteSequenceLabel(label).fold(
                onSuccess = { refreshStats() },
                onFailure = { e ->
                    _uiState.update { it.copy(error = "Error eliminando secuencia \"$label\": ${e.message}") }
                }
            )
        }
    }

    fun refreshStats() {
        checkConnection()
    }

    fun clearError() { _uiState.update { it.copy(error = null) } }
    fun clearDownloadMessage() { _uiState.update { it.copy(downloadMessage = null) } }

    override fun onCleared() {
        super.onCleared()
        pollingJob?.cancel()
    }
}
