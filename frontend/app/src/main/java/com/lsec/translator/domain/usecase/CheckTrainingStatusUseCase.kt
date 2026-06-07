package com.lsec.translator.domain.usecase

import com.lsec.translator.data.remote.TrainingStatusResponseDto
import com.lsec.translator.data.repository.TrainingRepository
import javax.inject.Inject

/**
 * Consulta el estado de un entrenamiento en progreso.
 */
class CheckTrainingStatusUseCase @Inject constructor(
    private val trainingRepository: TrainingRepository
) {
    suspend operator fun invoke(taskId: String): Result<TrainingStatusResponseDto> {
        return trainingRepository.getTrainingStatus(taskId)
    }
}
