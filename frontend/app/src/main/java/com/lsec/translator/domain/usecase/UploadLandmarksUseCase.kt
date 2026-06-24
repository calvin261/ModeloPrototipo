package com.lsec.translator.domain.usecase

import com.lsec.translator.data.remote.LandmarkSampleDto
import com.lsec.translator.data.remote.SequenceUploadResponseDto
import com.lsec.translator.data.remote.StaticLandmarkUploadDto
import com.lsec.translator.data.remote.UploadResponseDto
import com.lsec.translator.data.repository.TrainingRepository
import com.lsec.translator.data.upload.NpyWriter
import javax.inject.Inject

/**
 * Resultado agregado de subir ambas modalidades (estática + secuencias).
 */
data class UploadSummary(
    val staticAdded: Int,
    val staticTotal: Int,
    val sequencesUploaded: Int,
    val sequenceFramesByLabel: Map<String, Int>
)

/**
 * Sube landmarks al servidor. Dos modalidades:
 * - Estática: muestras individuales de 63 floats → /api/v1/data/upload/static
 * - Secuencias: matrices (T, 63) serializadas a .npy → /api/v1/data/upload/sequence
 *
 * Antes del fix (F7-validacion 2026-06-23) este use case aplanaba cada
 * secuencia capturada en N samples estáticos, lo que destruía la estructura
 * temporal y hacía imposible entrenar el modelo dinámico.
 */
class UploadLandmarksUseCase @Inject constructor(
    private val trainingRepository: TrainingRepository
) {

    suspend fun uploadStatic(
        samples: List<Pair<String, FloatArray>>
    ): Result<UploadResponseDto> {
        if (samples.isEmpty()) return Result.failure(IllegalArgumentException("Sin muestras"))
        val dtos = samples.map { (label, landmarks) ->
            LandmarkSampleDto(label = label, landmarks = landmarks.toList())
        }
        return trainingRepository.uploadStaticLandmarks(dtos)
    }

    suspend fun uploadSequence(
        label: String,
        frames: List<FloatArray>
    ): Result<SequenceUploadResponseDto> {
        if (frames.isEmpty()) return Result.failure(IllegalArgumentException("Secuencia vacía"))
        val npy = NpyWriter.writeFloat32Matrix(frames, columns = 63)
        return trainingRepository.uploadSequence(label, npy)
    }

    /**
     * Sube ambos buffers en una sola llamada. Útil para el botón "Subir todo"
     * de la pantalla de captura. Devuelve un resumen agregado.
     */
    suspend fun uploadAll(
        staticSamples: List<Pair<String, FloatArray>>,
        sequences: List<Pair<String, List<FloatArray>>>
    ): Result<UploadSummary> {
        var staticAdded = 0
        var staticTotal = 0
        val perLabel = mutableMapOf<String, Int>()
        val errors = mutableListOf<String>()

        if (staticSamples.isNotEmpty()) {
            val r = uploadStatic(staticSamples)
            r.fold(
                onSuccess = {
                    staticAdded = it.samplesAdded
                    staticTotal = it.totalSamples
                },
                onFailure = { errors += "estáticas: ${it.message}" }
            )
        }

        var sequencesUploaded = 0
        for ((label, frames) in sequences) {
            val r = uploadSequence(label, frames)
            r.fold(
                onSuccess = {
                    sequencesUploaded++
                    perLabel[label] = (perLabel[label] ?: 0) + it.frames
                },
                onFailure = { errors += "secuencia '$label': ${it.message}" }
            )
        }

        return if (errors.isEmpty() || (staticAdded + sequencesUploaded > 0 && errors.size < sequences.size + if (staticSamples.isNotEmpty()) 1 else 0)) {
            Result.success(
                UploadSummary(
                    staticAdded = staticAdded,
                    staticTotal = staticTotal,
                    sequencesUploaded = sequencesUploaded,
                    sequenceFramesByLabel = perLabel
                )
            )
        } else {
            Result.failure(Exception(errors.joinToString("; ")))
        }
    }
}