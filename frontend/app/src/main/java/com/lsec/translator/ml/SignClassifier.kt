package com.lsec.translator.ml

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Resultado de la clasificacion de una sena LSEC.
 */
data class SignClassificationResult(
    val signName: String,
    val ecuadorianVariant: String,
    val confidence: Float,
    val classIndex: Int
)

/**
 * Facade que delega la clasificacion a StaticSignClassifier o DynamicSignClassifier
 * segun el modo actual.
 *
 * Carga labels desde archivos JSON (descargados del servidor o bundled en assets).
 * Mantiene compatibilidad con el contrato existente de ClassifySignUseCase.
 */
@Singleton
class SignClassifier @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var staticClassifier: StaticSignClassifier? = null
    private var dynamicClassifier: DynamicSignClassifier? = null

    // Mapeo de ecuatorianismos para enriquecer resultados
    private val ecuadorianVariants = mapOf(
        "hola" to "¡Qué más!",
        "gracias" to "Gracias, pana",
        "por_favor" to "Por fa",
        "si" to "Sí, pues",
        "no" to "Nel",
        "ayuda" to "Dame una mano",
        "agua" to "Agüita",
        "comida" to "Morfar",
        "bien" to "Chévere",
        "mal" to "Está feo",
        "perdon" to "Disculpe",
        "adios" to "¡Chao, nos vemos!",
        "como_estas" to "¿Qué fue, cómo vas?",
        "nombre" to "Me llamo",
        "bano" to "El baño"
    )

    init {
        initialize()
    }

    private fun initialize() {
        // Cargar labels del clasificador estatico
        val classifierLabels = loadLabelsJson("classifier_labels.json")
            ?: defaultClassifierLabels()

        staticClassifier = StaticSignClassifier(context, classifierLabels)

        // Cargar vocabulario del clasificador dinamico
        val sequenceLabels = loadLabelsJson("sequence_labels.json")
            ?: defaultSequenceLabels()

        dynamicClassifier = DynamicSignClassifier(context, sequenceLabels)

        Log.i(TAG, "SignClassifier facade initialized (static: ${classifierLabels.size} classes, dynamic: ${sequenceLabels.size} classes)")
    }

    /**
     * Clasifica una secuencia de landmarks (modo compatible con CameraViewModel existente).
     *
     * Para secuencias cortas (<= 5 frames), usa clasificacion estatica del ultimo frame.
     * Para secuencias largas, usa clasificacion dinamica.
     */
    suspend fun classify(landmarkSequence: List<FloatArray>): SignClassificationResult =
        withContext(Dispatchers.Default) {
            if (landmarkSequence.isEmpty()) {
                return@withContext SignClassificationResult(
                    signName = "Desconocido",
                    ecuadorianVariant = "",
                    confidence = 0f,
                    classIndex = -1
                )
            }

            // Modo estatico: clasificar el frame mas reciente
            val lastFrame = landmarkSequence.last()
            val staticResult = staticClassifier?.classify(lastFrame)

            if (staticResult != null) {
                return@withContext enrichWithEcuadorianVariant(staticResult)
            }

            // Fallback si no hay modelo disponible
            return@withContext SignClassificationResult(
                signName = "Desconocido",
                ecuadorianVariant = "",
                confidence = 0f,
                classIndex = -1
            )
        }

    /**
     * Clasifica un frame individual (modo estatico directo).
     */
    fun classifyStatic(landmarks: FloatArray): SignClassificationResult? {
        val result = staticClassifier?.classify(landmarks) ?: return null
        return enrichWithEcuadorianVariant(result)
    }

    /**
     * Procesa un frame para clasificacion dinamica.
     * Retorna resultado cuando completa una secuencia, null mientras bufferea.
     */
    fun processDynamicFrame(landmarks: FloatArray?): SignClassificationResult? {
        val result = dynamicClassifier?.processFrame(landmarks) ?: return null
        return enrichWithEcuadorianVariant(result)
    }

    /**
     * Limpia el buffer del clasificador dinamico.
     */
    fun clearDynamicBuffer() {
        dynamicClassifier?.clearBuffer()
    }

    /**
     * Recarga los modelos (e.g. despues de descargar nuevas versiones).
     */
    fun reloadModels(
        newClassifierLabels: Map<String, Int>? = null,
        newSequenceLabels: Map<String, Int>? = null
    ) {
        staticClassifier?.reload(newClassifierLabels)
        dynamicClassifier?.reload(newSequenceLabels)
    }

    private fun enrichWithEcuadorianVariant(result: SignClassificationResult): SignClassificationResult {
        val key = result.signName.lowercase().replace(" ", "_")
        val variant = ecuadorianVariants[key]
        return if (variant != null) {
            result.copy(ecuadorianVariant = variant)
        } else {
            result
        }
    }

    /**
     * Carga labels desde un archivo JSON en filesDir/models/ o assets/.
     */
    private fun loadLabelsJson(filename: String): Map<String, Int>? {
        // Intentar desde filesDir (descargado del servidor)
        val downloadedFile = File(context.filesDir, "models/$filename")
        if (downloadedFile.exists()) {
            return parseLabelsJson(downloadedFile.readText())
        }

        // Intentar desde assets
        return try {
            val text = context.assets.open(filename).bufferedReader().readText()
            parseLabelsJson(text)
        } catch (e: Exception) {
            null
        }
    }

    private fun parseLabelsJson(jsonText: String): Map<String, Int> {
        val json = JSONObject(jsonText)
        val map = mutableMapOf<String, Int>()
        json.keys().forEach { key ->
            map[key] = json.getInt(key)
        }
        return map
    }

    private fun defaultClassifierLabels(): Map<String, Int> {
        return mapOf(
            "A" to 0, "B" to 1, "C" to 2, "D" to 3, "E" to 4,
            "F" to 5, "G" to 6, "H" to 7, "I" to 8, "J" to 9,
            "K" to 10, "L" to 11, "M" to 12, "N" to 13, "O" to 14,
            "P" to 15, "Q" to 16, "R" to 17, "S" to 18, "T" to 19,
            "U" to 20, "V" to 21, "W" to 22, "X" to 23, "Y" to 24,
            "Z" to 25, "SPACE" to 26
        )
    }

    private fun defaultSequenceLabels(): Map<String, Int> {
        return mapOf(
            "hola" to 0, "gracias" to 1, "por_favor" to 2, "si" to 3,
            "no" to 4, "ayuda" to 5, "agua" to 6, "comida" to 7,
            "bano" to 8, "nombre" to 9, "como_estas" to 10, "bien" to 11,
            "mal" to 12, "perdon" to 13, "adios" to 14,
            "A" to 15, "B" to 16, "C" to 17, "D" to 18, "E" to 19,
            "F" to 20, "G" to 21, "H" to 22, "I" to 23, "J" to 24,
            "K" to 25, "L" to 26, "M" to 27, "N" to 28, "O" to 29,
            "P" to 30, "Q" to 31, "R" to 32, "S" to 33, "T" to 34,
            "U" to 35, "V" to 36, "W" to 37, "X" to 38, "Y" to 39,
            "Z" to 40
        )
    }

    fun close() {
        staticClassifier?.close()
        dynamicClassifier?.close()
    }

    companion object {
        private const val TAG = "SignClassifier"
    }
}

/**
 * Representa una etiqueta de sena LSEC con su variante ecuatoriana.
 */
data class SignLabel(
    val index: Int,
    val name: String,
    val ecuadorianVariant: String
)
