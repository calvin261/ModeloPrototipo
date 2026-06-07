package com.lsec.translator.data.remote

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.io.FileOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Gestiona la descarga y almacenamiento de modelos TFLite desde el servidor.
 *
 * Los modelos se guardan en filesDir/models/ con nombres fijos para que
 * StaticSignClassifier y DynamicSignClassifier los encuentren automaticamente.
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
     * @param modelType "classifier" o "sequence"
     * @return true si se descargo exitosamente
     */
    suspend fun downloadLatestModel(modelType: String): Boolean {
        return try {
            // Obtener info del modelo mas reciente
            val infoResponse = apiService.getLatestModel(modelType)
            if (!infoResponse.isSuccessful) {
                Log.w(TAG, "Failed to get latest $modelType info: ${infoResponse.code()}")
                return false
            }

            val modelInfo = infoResponse.body() ?: return false

            // Descargar el modelo
            val downloadResponse = apiService.downloadModel(modelType, modelInfo.version)
            if (!downloadResponse.isSuccessful) {
                Log.w(TAG, "Failed to download $modelType model: ${downloadResponse.code()}")
                return false
            }

            val body = downloadResponse.body() ?: return false

            // Determinar extensión según Content-Disposition o header X-Model-Format
            val formatHeader = downloadResponse.headers()["X-Model-Format"] ?: "onnx"
            val ext = if (formatHeader == "tflite") "tflite" else "onnx"
            val filename = when (modelType) {
                "classifier" -> "sign_mlp.$ext"
                "sequence" -> "sign_gru.$ext"
                else -> return false
            }

            val targetFile = File(modelsDir, filename)
            val tempFile = File(modelsDir, "$filename.tmp")

            // Escribir a archivo temporal y luego renombrar (atomico)
            FileOutputStream(tempFile).use { output ->
                body.byteStream().use { input ->
                    input.copyTo(output)
                }
            }
            tempFile.renameTo(targetFile)

            // Descargar labels tambien
            downloadLabels(modelType)

            Log.i(TAG, "Downloaded $modelType model ${modelInfo.version} (${modelInfo.sizeKb}KB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading $modelType model", e)
            false
        }
    }

    /**
     * Descarga el label map para un tipo de modelo.
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
     * Verifica si hay una version mas reciente disponible.
     *
     * @param modelType "classifier" o "sequence"
     * @param currentVersion version local actual
     * @return ModelLatestResponseDto si hay nueva version, null si no
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
            else -> return false
        }
        return File(modelsDir, "$baseName.onnx").exists() ||
               File(modelsDir, "$baseName.tflite").exists()
    }

    companion object {
        private const val TAG = "ModelDownloadManager"
    }
}
