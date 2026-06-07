package com.lsec.translator.domain.usecase

import com.lsec.translator.data.remote.TrainingStartResponseDto
import com.lsec.translator.data.repository.TrainingRepository
import javax.inject.Inject

/**
 * Dispara entrenamiento en el servidor.
 */
class StartTrainingUseCase @Inject constructor(
    private val trainingRepository: TrainingRepository
) {
    suspend fun classifier(
        epochs: Int = 50,
        learningRate: Float = 0.001f,
        batchSize: Int = 64
    ): Result<TrainingStartResponseDto> {
        return trainingRepository.startClassifierTraining(epochs, learningRate, batchSize)
    }

    suspend fun sequence(
        epochs: Int = 80,
        learningRate: Float = 0.001f,
        batchSize: Int = 32
    ): Result<TrainingStartResponseDto> {
        return trainingRepository.startSequenceTraining(epochs, learningRate, batchSize)
    }
}
