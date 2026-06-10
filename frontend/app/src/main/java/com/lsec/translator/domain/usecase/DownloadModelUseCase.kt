package com.lsec.translator.domain.usecase

import com.lsec.translator.data.repository.ModelRepository
import com.lsec.translator.ml.SignClassifier
import javax.inject.Inject

/**
 * Descarga un modelo del servidor (latest o version especifica) y recarga
 * el clasificador con los nuevos labels.
 */
class DownloadModelUseCase @Inject constructor(
    private val modelRepository: ModelRepository,
    private val signClassifier: SignClassifier
) {
    /**
     * Descarga la ultima version del tipo de modelo solicitado.
     * @return true si descargo y recargo exitosamente
     */
    suspend operator fun invoke(modelType: String): Boolean {
        return invoke(modelType, version = null)
    }

    /**
     * Descarga una version especifica (o la ultima si version == null).
     */
    suspend operator fun invoke(modelType: String, version: String?): Boolean {
        val downloaded = if (version == null) {
            modelRepository.downloadLatestModel(modelType)
        } else {
            modelRepository.downloadModelVersion(modelType, version)
        }
        if (downloaded) {
            val labelsResult = modelRepository.getAvailableLabels(modelType)
            val newLabels = labelsResult.getOrNull()
            when (modelType) {
                "classifier" -> signClassifier.reloadModels(newClassifierLabels = newLabels)
                "sequence" -> signClassifier.reloadModels(newSequenceLabels = newLabels)
            }
        }
        return downloaded
    }
}
