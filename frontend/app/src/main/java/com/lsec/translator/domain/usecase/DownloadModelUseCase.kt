package com.lsec.translator.domain.usecase

import com.lsec.translator.data.repository.ModelRepository
import com.lsec.translator.ml.SignClassifier
import javax.inject.Inject

/**
 * Descarga un modelo actualizado del servidor y recarga el clasificador.
 */
class DownloadModelUseCase @Inject constructor(
    private val modelRepository: ModelRepository,
    private val signClassifier: SignClassifier
) {
    /**
     * @param modelType "classifier" o "sequence"
     * @return true si descargo y recargo exitosamente
     */
    suspend operator fun invoke(modelType: String): Boolean {
        val downloaded = modelRepository.downloadLatestModel(modelType)
        if (downloaded) {
            // Obtener nuevos labels
            val labelsResult = modelRepository.getAvailableLabels(modelType)
            val newLabels = labelsResult.getOrNull()

            // Recargar clasificadores con nuevos modelos/labels
            when (modelType) {
                "classifier" -> signClassifier.reloadModels(newClassifierLabels = newLabels)
                "sequence" -> signClassifier.reloadModels(newSequenceLabels = newLabels)
            }
        }
        return downloaded
    }
}
