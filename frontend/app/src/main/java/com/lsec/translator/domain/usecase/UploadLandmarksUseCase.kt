package com.lsec.translator.domain.usecase

import com.lsec.translator.data.remote.LandmarkSampleDto
import com.lsec.translator.data.remote.UploadResponseDto
import com.lsec.translator.data.repository.TrainingRepository
import javax.inject.Inject

/**
 * Sube landmarks capturados al servidor para entrenamiento.
 */
class UploadLandmarksUseCase @Inject constructor(
    private val trainingRepository: TrainingRepository
) {
    suspend operator fun invoke(
        samples: List<Pair<String, FloatArray>>
    ): Result<UploadResponseDto> {
        val dtos = samples.map { (label, landmarks) ->
            LandmarkSampleDto(
                label = label,
                landmarks = landmarks.toList()
            )
        }
        return trainingRepository.uploadStaticLandmarks(dtos)
    }
}
