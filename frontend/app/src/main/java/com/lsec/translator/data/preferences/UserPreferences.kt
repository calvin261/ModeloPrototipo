package com.lsec.translator.data.preferences

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import com.lsec.translator.BuildConfig
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_settings")

/**
 * Preferencias del usuario guardadas con DataStore.
 */
data class UserPreferencesData(
    // Voz
    val isTTSEnabled: Boolean = true,
    val speechRate: Float = 1.0f,
    val pitch: Float = 1.0f,
    val preferEcuadorianVariants: Boolean = true,

    // Cámara
    val useFrontCamera: Boolean = true,
    val minConfidence: Float = 0.70f,
    val batterySaverMode: Boolean = false,

    // Apariencia
    val darkMode: Boolean = false,

    // Servidor y modelos
    val serverUrl: String = BuildConfig.DEFAULT_BACKEND_URL,
    val apiKey: String = "",
    val classifierModelVersion: String = "bundled",
    val sequenceModelVersion: String = "bundled",
    val autoUpdateModels: Boolean = false,
    val recognitionMode: String = "static", // "static" o "dynamic"

    // LLM post-processor (opcional, off por defecto)
    val llmEnabled: Boolean = false,
    val llmAutoTrigger: Boolean = false,
)

/**
 * Gestiona las preferencias del usuario usando DataStore.
 */
@Singleton
class UserPreferences @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private object PreferencesKeys {
        val TTS_ENABLED = booleanPreferencesKey("tts_enabled")
        val SPEECH_RATE = floatPreferencesKey("speech_rate")
        val PITCH = floatPreferencesKey("pitch")
        val PREFER_ECUADORIAN = booleanPreferencesKey("prefer_ecuadorian")
        val USE_FRONT_CAMERA = booleanPreferencesKey("use_front_camera")
        val MIN_CONFIDENCE = floatPreferencesKey("min_confidence")
        val BATTERY_SAVER = booleanPreferencesKey("battery_saver")
        val DARK_MODE = booleanPreferencesKey("dark_mode")
        val SERVER_URL = stringPreferencesKey("server_url")
        val API_KEY = stringPreferencesKey("api_key")
        val CLASSIFIER_MODEL_VERSION = stringPreferencesKey("classifier_model_version")
        val SEQUENCE_MODEL_VERSION = stringPreferencesKey("sequence_model_version")
        val AUTO_UPDATE_MODELS = booleanPreferencesKey("auto_update_models")
        val RECOGNITION_MODE = stringPreferencesKey("recognition_mode")
        val LLM_ENABLED = booleanPreferencesKey("llm_enabled")
        val LLM_AUTO_TRIGGER = booleanPreferencesKey("llm_auto_trigger")
    }

    /**
     * Flow de preferencias del usuario.
     */
    val preferencesFlow: Flow<UserPreferencesData> = context.dataStore.data
        .catch { exception ->
            if (exception is IOException) {
                emit(emptyPreferences())
            } else {
                throw exception
            }
        }
        .map { preferences ->
            UserPreferencesData(
                isTTSEnabled = preferences[PreferencesKeys.TTS_ENABLED] ?: true,
                speechRate = preferences[PreferencesKeys.SPEECH_RATE] ?: 1.0f,
                pitch = preferences[PreferencesKeys.PITCH] ?: 1.0f,
                preferEcuadorianVariants = preferences[PreferencesKeys.PREFER_ECUADORIAN] ?: true,
                useFrontCamera = preferences[PreferencesKeys.USE_FRONT_CAMERA] ?: true,
                minConfidence = preferences[PreferencesKeys.MIN_CONFIDENCE] ?: 0.70f,
                batterySaverMode = preferences[PreferencesKeys.BATTERY_SAVER] ?: false,
                darkMode = preferences[PreferencesKeys.DARK_MODE] ?: false,
                serverUrl = preferences[PreferencesKeys.SERVER_URL] ?: BuildConfig.DEFAULT_BACKEND_URL,
                apiKey = preferences[PreferencesKeys.API_KEY] ?: "",
                classifierModelVersion = preferences[PreferencesKeys.CLASSIFIER_MODEL_VERSION] ?: "bundled",
                sequenceModelVersion = preferences[PreferencesKeys.SEQUENCE_MODEL_VERSION] ?: "bundled",
                autoUpdateModels = preferences[PreferencesKeys.AUTO_UPDATE_MODELS] ?: false,
                recognitionMode = preferences[PreferencesKeys.RECOGNITION_MODE] ?: "static",
                llmEnabled = preferences[PreferencesKeys.LLM_ENABLED] ?: false,
                llmAutoTrigger = preferences[PreferencesKeys.LLM_AUTO_TRIGGER] ?: false,
            )
        }

    // Setters para cada preferencia

    suspend fun setTTSEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.TTS_ENABLED] = enabled
        }
    }

    suspend fun setSpeechRate(rate: Float) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.SPEECH_RATE] = rate
        }
    }

    suspend fun setPitch(pitch: Float) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.PITCH] = pitch
        }
    }

    suspend fun setPreferEcuadorianVariants(prefer: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.PREFER_ECUADORIAN] = prefer
        }
    }

    suspend fun setUseFrontCamera(useFront: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.USE_FRONT_CAMERA] = useFront
        }
    }

    suspend fun setMinConfidence(confidence: Float) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.MIN_CONFIDENCE] = confidence
        }
    }

    suspend fun setBatterySaverMode(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.BATTERY_SAVER] = enabled
        }
    }

    suspend fun setDarkMode(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.DARK_MODE] = enabled
        }
    }

    suspend fun setServerUrl(url: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.SERVER_URL] = url
        }
    }

    suspend fun setApiKey(key: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.API_KEY] = key
        }
    }

    suspend fun setClassifierModelVersion(version: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.CLASSIFIER_MODEL_VERSION] = version
        }
    }

    suspend fun setSequenceModelVersion(version: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.SEQUENCE_MODEL_VERSION] = version
        }
    }

    suspend fun setAutoUpdateModels(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.AUTO_UPDATE_MODELS] = enabled
        }
    }

    suspend fun setRecognitionMode(mode: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.RECOGNITION_MODE] = mode
        }
    }

    suspend fun setLlmEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.LLM_ENABLED] = enabled
        }
    }

    suspend fun setLlmAutoTrigger(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.LLM_AUTO_TRIGGER] = enabled
        }
    }
}
