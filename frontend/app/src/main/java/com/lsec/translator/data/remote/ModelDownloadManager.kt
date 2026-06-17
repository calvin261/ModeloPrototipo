package com.lsec.translator.data.remote

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.FileOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Gestiona la descarga y almacenamiento de modelos ONNX/TFLite desde el
 * servidor.
 *
 * Los modelos se guardan en `filesDir/models/` con nombres fijos para que
 * los clasificadores los encuentren automaticamente.
 *
 * Modelos soportados:
 *   - "classifier"   -> sign_mlp.onnx / sign_mlp.tflite            (StaticSignClassifier)
 *   - "sequence"     -> sign_gru.onnx / sign_gru.tflite            (DynamicSignClassifier, legacy)
 *   - "transformer"  -> sign_transformer.onnx / sign_transformer_labels.json
 *                       (SignTransformerClassifier — modelo nuevo)
 */
@Singleton
class ModelDownloadManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val apiService: LSECApiService
) {
    private val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

    /**
     * Descarga el modelo mas reciente del tipo especificado.
     *
     * @param modelType "classifier" | "sequence" | "transformer"
     * @return true si se descargo exitosamente
     */
    suspend fun downloadLatestModel(modelType: String): Boolean {
        return try {
            val infoResponse = apiService.getLatestModel(modelType)
            if (!infoResponse.isSuccessful) {
                Log.w(TAG, "Failed to get latest $modelType info: ${infoResponse.code()}")
                return false
            }
            val modelInfo = infoResponse.body() ?: return false

            val downloadResponse = apiService.downloadModel(modelType, modelInfo.version)
            if (!downloadResponse.isSuccessful) {
                Log.w(TAG, "Failed to download $modelType model: ${downloadResponse.code()}")
                return false
            }
            val body = downloadResponse.body() ?: return false

            val formatHeader = downloadResponse.headers()["X-Model-Format"] ?: "onnx"
            val ext = if (formatHeader == "tflite") "tflite" else "onnx"
            val filename = modelFilename(modelType, ext)

            val targetFile = File(modelsDir, filename)
            val tempFile = File(modelsDir, "$filename.tmp")

            FileOutputStream(tempFile).use { output ->
                body.byteStream().use { input -> input.copyTo(output) }
            }
            tempFile.renameTo(targetFile)

            // Labels: el transformer los baja de un endpoint dedicado.
            if (modelType == "transformer") {
                downloadTransformerLabels()
            } else {
                downloadLabels(modelType)
            }

            Log.i(TAG, "Downloaded $modelType model ${modelInfo.version} (${modelInfo.sizeKb}KB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading $modelType model", e)
            false
        }
    }

    /**
     * Descarga una version especifica de un modelo (no necesariamente la ultima).
     */
    suspend fun downloadModelVersion(modelType: String, version: String): Boolean {
        return try {
            val downloadResponse = apiService.downloadModel(modelType, version)
            if (!downloadResponse.isSuccessful) {
                Log.w(TAG, "Failed to download $modelType $version: ${downloadResponse.code()}")
                return false
            }
            val body = downloadResponse.body() ?: return false

            val formatHeader = downloadResponse.headers()["X-Model-Format"] ?: "onnx"
            val ext = if (formatHeader == "tflite") "tflite" else "onnx"
            val filename = modelFilename(modelType, ext)

            val targetFile = File(modelsDir, filename)
            val tempFile = File(modelsDir, "$filename.tmp")
            FileOutputStream(tempFile).use { out ->
                body.byteStream().use { input -> input.copyTo(out) }
            }
            tempFile.renameTo(targetFile)

            if (modelType == "transformer") {
                downloadTransformerLabels()
            } else {
                downloadLabels(modelType)
            }

            Log.i(TAG, "Downloaded $modelType model $version (${targetFile.length() / 1024}KB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading $modelType $version", e)
            false
        }
    }

    /**
     * Descarga el label map para un modelo (classifier/sequence).
     * Para el transformer se usa [downloadTransformerLabels].
     */
    suspend fun downloadLabels(modelType: String): Boolean {
        return try {
            val response = apiService.getLabels(modelType)
            if (!response.isSuccessful) return false
            val labels = response.body() ?: return false

            val filename = when (modelType) {
                "classifier" -> "classifier_labels.json"
                "sequence" -> "sequence_labels.json"
                else -> return false
            }
            val file = File(modelsDir, filename)
            val json = org.json.JSONObject(labels as Map<*, *>).toString(2)
            file.writeText(json)
            Log.i(TAG, "Downloaded $modelType labels (${labels.size} entries)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading $modelType labels", e)
            false
        }
    }

    /**
     * Descarga el label map del transformer. El backend lo expone como
     * `sign_transformer_labels.json` (mismo endpoint /api/v1/models/labels/transformer
     * introducido en el commit que anade el tipo "transformer" al router).
     */
    private suspend fun downloadTransformerLabels(): Boolean {
        return try {
            val response = apiService.getLabels("transformer")
            if (!response.isSuccessful) {
                Log.w(TAG, "Transformer labels endpoint returned ${response.code()}")
                return false
            }
            val labels = response.body() ?: return false
            val file = File(modelsDir, "sign_transformer_labels.json")
            val json = org.json.JSONObject(labels as Map<*, *>).toString(2)
            file.writeText(json)
            Log.i(TAG, "Downloaded transformer labels (${labels.size} entries)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading transformer labels", e)
            false
        }
    }

    /**
     * Verifica si hay una version mas reciente disponible.
     */
    suspend fun checkForUpdate(
        modelType: String,
        currentVersion: String
    ): ModelLatestResponseDto? {
        return try {
            val response = apiService.getLatestModel(modelType)
            if (!response.isSuccessful) return null
            val latest = response.body() ?: return null
            if (latest.version != currentVersion) latest else null
        } catch (e: Exception) {
            Log.w(TAG, "Error checking for $modelType update", e)
            null
        }
    }

    /**
     * Verifica si hay modelos descargados localmente.
     */
    fun hasLocalModel(modelType: String): Boolean {
        val baseName = when (modelType) {
            "classifier" -> "sign_mlp"
            "sequence" -> "sign_gru"
            "transformer" -> "sign_transformer"
            else -> return false
        }
        return File(modelsDir, "$baseName.onnx").exists() ||
               File(modelsDir, "$baseName.tflite").exists()
    }

    private fun modelFilename(modelType: String, ext: String): String = when (modelType) {
        "classifier" -> "sign_mlp.$ext"
        "sequence" -> "sign_gru.$ext"
        "transformer" -> "sign_transformer.$ext"
        else -> "unknown_model.$ext"
    }

    companion object {
        private const val TAG = "ModelDownloadManager"
    }
}
