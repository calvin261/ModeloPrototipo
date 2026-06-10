package com.lsec.translator.data.repository

import com.lsec.translator.data.remote.LSECApiService
import com.lsec.translator.data.remote.ModelDownloadManager
import com.lsec.translator.data.remote.ModelInfoDto
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

    /**
     * Descarga una version especifica de un modelo.
     * Retorna true si descargo OK.
     */
    suspend fun downloadModelVersion(modelType: String, version: String): Boolean {
        return downloadManager.downloadModelVersion(modelType, version)
    }

    /**
     * Lista todas las versiones disponibles de todos los modelos en el backend.
     * Retorna Result con la lista de ModelInfoDto.
     */
    suspend fun listAvailableModels(): Result<List<ModelInfoDto>> {
        return try {
            val response = apiService.listModels()
            if (response.isSuccessful) {
                Result.success(response.body() ?: emptyList())
            } else {
                Result.failure(Exception("listModels failed: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
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
