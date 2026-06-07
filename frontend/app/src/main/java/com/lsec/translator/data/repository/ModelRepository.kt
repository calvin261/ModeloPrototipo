package com.lsec.translator.data.repository

import com.lsec.translator.data.remote.LSECApiService
import com.lsec.translator.data.remote.ModelDownloadManager
import com.lsec.translator.data.remote.ModelLatestResponseDto
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repositorio para gestion de modelos ML (versiones, descarga, actualizacion).
 */
@Singleton
class ModelRepository @Inject constructor(
    private val apiService: LSECApiService,
    private val downloadManager: ModelDownloadManager
) {
    suspend fun checkForUpdate(
        modelType: String,
        currentVersion: String
    ): ModelLatestResponseDto? {
        return downloadManager.checkForUpdate(modelType, currentVersion)
    }

    suspend fun downloadLatestModel(modelType: String): Boolean {
        return downloadManager.downloadLatestModel(modelType)
    }

    fun hasLocalModel(modelType: String): Boolean {
        return downloadManager.hasLocalModel(modelType)
    }

    suspend fun getAvailableLabels(modelType: String): Result<Map<String, Int>> {
        return try {
            val response = apiService.getLabels(modelType)
            if (response.isSuccessful) {
                Result.success(response.body() ?: emptyMap())
            } else {
                Result.failure(Exception("Labels fetch failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
