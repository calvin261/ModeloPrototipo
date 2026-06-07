package com.lsec.translator.data.repository

import com.lsec.translator.data.remote.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repositorio para operaciones de entrenamiento remoto.
 */
@Singleton
class TrainingRepository @Inject constructor(
    private val apiService: LSECApiService
) {
    private fun <T : Any> bodyOrFailure(body: T?, operation: String): Result<T> {
        return body?.let { Result.success(it) }
            ?: Result.failure(Exception("$operation returned an empty response body"))
    }

    suspend fun startClassifierTraining(
        epochs: Int = 50,
        learningRate: Float = 0.001f,
        batchSize: Int = 64
    ): Result<TrainingStartResponseDto> {
        return try {
            val response = apiService.startClassifierTraining(
                TrainingParamsDto(epochs, learningRate, batchSize)
            )
            if (response.isSuccessful) {
                bodyOrFailure(response.body(), "Training start")
            } else {
                Result.failure(Exception("Training start failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun startSequenceTraining(
        epochs: Int = 80,
        learningRate: Float = 0.001f,
        batchSize: Int = 32
    ): Result<TrainingStartResponseDto> {
        return try {
            val response = apiService.startSequenceTraining(
                TrainingParamsDto(epochs, learningRate, batchSize)
            )
            if (response.isSuccessful) {
                bodyOrFailure(response.body(), "Training start")
            } else {
                Result.failure(Exception("Training start failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getTrainingStatus(taskId: String): Result<TrainingStatusResponseDto> {
        return try {
            val response = apiService.getTrainingStatus(taskId)
            if (response.isSuccessful) {
                bodyOrFailure(response.body(), "Status check")
            } else {
                Result.failure(Exception("Status check failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getTrainingHistory(): Result<List<TrainingHistoryEntryDto>> {
        return try {
            val response = apiService.getTrainingHistory()
            if (response.isSuccessful) {
                Result.success(response.body() ?: emptyList())
            } else {
                Result.failure(Exception("History fetch failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun uploadStaticLandmarks(
        samples: List<LandmarkSampleDto>
    ): Result<UploadResponseDto> {
        return try {
            val response = apiService.uploadStaticLandmarks(
                StaticLandmarkUploadDto(samples)
            )
            if (response.isSuccessful) {
                bodyOrFailure(response.body(), "Upload")
            } else {
                Result.failure(Exception("Upload failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getDataStats(): Result<DataStatsResponseDto> {
        return try {
            val response = apiService.getDataStats()
            if (response.isSuccessful) {
                bodyOrFailure(response.body(), "Stats fetch")
            } else {
                Result.failure(Exception("Stats fetch failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun deleteStaticLabel(label: String): Result<Unit> {
        return try {
            val response = apiService.deleteStaticLabel(label)
            if (response.isSuccessful) Result.success(Unit)
            else Result.failure(Exception("Delete failed: ${response.code()}"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun deleteSequenceLabel(label: String): Result<Unit> {
        return try {
            val response = apiService.deleteSequenceLabel(label)
            if (response.isSuccessful) Result.success(Unit)
            else Result.failure(Exception("Delete failed: ${response.code()}"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun exportModel(modelType: String): Result<Unit> {
        return try {
            val response = apiService.exportModel(modelType)
            if (response.isSuccessful) Result.success(Unit)
            else Result.failure(Exception("Export failed: ${response.code()}"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
