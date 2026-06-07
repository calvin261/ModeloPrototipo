package com.lsec.translator.ui.screens.training

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.remote.DataStatsResponseDto
import com.lsec.translator.data.remote.TrainingHistoryEntryDto
import com.lsec.translator.data.repository.TrainingRepository
import com.lsec.translator.domain.usecase.CheckTrainingStatusUseCase
import com.lsec.translator.domain.usecase.DownloadModelUseCase
import com.lsec.translator.domain.usecase.StartTrainingUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class TrainingUiState(
    val isConnected: Boolean = false,
    val activeTaskId: String? = null,
    val activeModelType: String? = null,
    val trainingStatus: String = "idle", // idle, pending, running, completed, failed
    val epoch: Int = 0,
    val totalEpochs: Int = 0,
    val trainLoss: Float = 0f,
    val valAccuracy: Float = 0f,
    val bestAccuracy: Float = 0f,
    val error: String? = null,
    val history: List<TrainingHistoryEntryDto> = emptyList(),
    val dataStats: DataStatsResponseDto? = null,
    val isDownloading: Boolean = false,
    val downloadMessage: String? = null
)

@HiltViewModel
class TrainingViewModel @Inject constructor(
    private val startTrainingUseCase: StartTrainingUseCase,
    private val checkTrainingStatusUseCase: CheckTrainingStatusUseCase,
    private val downloadModelUseCase: DownloadModelUseCase,
    private val trainingRepository: TrainingRepository
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
                    _uiState.update {
                        it.copy(isConnected = true, dataStats = stats)
                    }
                    loadHistory()
                },
                onFailure = {
                    _uiState.update {
                        it.copy(isConnected = false, error = "No se pudo conectar al servidor")
                    }
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

    fun startClassifierTraining(epochs: Int = 50) {
        viewModelScope.launch {
            _uiState.update { it.copy(trainingStatus = "pending", error = null, activeModelType = "classifier") }

            val result = startTrainingUseCase.classifier(epochs = epochs)
            result.fold(
                onSuccess = { response ->
                    _uiState.update {
                        it.copy(
                            activeTaskId = response.taskId,
                            trainingStatus = "running",
                            totalEpochs = epochs
                        )
                    }
                    startPolling(response.taskId)
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(trainingStatus = "failed", error = error.message)
                    }
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
                        it.copy(
                            activeTaskId = response.taskId,
                            trainingStatus = "running",
                            totalEpochs = epochs
                        )
                    }
                    startPolling(response.taskId)
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(trainingStatus = "failed", error = error.message)
                    }
                }
            )
        }
    }

    private fun startPolling(taskId: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (true) {
                delay(3000) // Poll cada 3 segundos

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
                                error = status.error
                            )
                        }

                        if (status.status == "completed" || status.status == "failed") {
                            loadHistory()
                            return@launch
                        }
                    },
                    onFailure = { error ->
                        _uiState.update {
                            it.copy(error = "Error consultando estado: ${error.message}")
                        }
                    }
                )
            }
        }
    }

    fun downloadModel(modelType: String = _uiState.value.activeModelType ?: "classifier") {
        viewModelScope.launch {
            _uiState.update { it.copy(isDownloading = true, downloadMessage = "Preparando modelo...") }

            // Try download; if it fails, trigger server export then retry once
            var success = downloadModelUseCase(modelType)
            if (!success) {
                _uiState.update { it.copy(downloadMessage = "Exportando modelo en el servidor...") }
                trainingRepository.exportModel(modelType)
                delay(8000) // Give the server time to export
                _uiState.update { it.copy(downloadMessage = "Descargando modelo...") }
                success = downloadModelUseCase(modelType)
            }

            _uiState.update {
                it.copy(
                    isDownloading = false,
                    downloadMessage = if (success)
                        "Modelo $modelType descargado exitosamente"
                    else
                        "Error al descargar. Verifica que el servidor esta activo y reinicialo si acabas de entrenar."
                )
            }
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

    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    fun clearDownloadMessage() {
        _uiState.update { it.copy(downloadMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        pollingJob?.cancel()
    }
}
