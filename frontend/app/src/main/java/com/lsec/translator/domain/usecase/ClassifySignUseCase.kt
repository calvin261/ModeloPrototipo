package com.lsec.translator.domain.usecase

import com.lsec.translator.ml.SignClassificationResult
import com.lsec.translator.ml.SignClassifier
import javax.inject.Inject

/**
 * Caso de uso para clasificación de secuencias de señas LSEC.
 * Utiliza modelo LSTM/Bi-LSTM para análisis temporal.
 */
class ClassifySignUseCase @Inject constructor(
    private val signClassifier: SignClassifier
) {
    /**
     * Clasifica una secuencia de landmarks para identificar una seña LSEC.
     * 
     * @param landmarkSequence Lista de arrays de landmarks (secuencia temporal)
     * @return Resultado de clasificación con nombre de seña y confianza
     */
    suspend operator fun invoke(landmarkSequence: List<FloatArray>): SignClassificationResult {
        return signClassifier.classify(landmarkSequence)
    }
}
