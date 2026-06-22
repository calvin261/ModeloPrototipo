package com.lsec.translator.ui.screens.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.BuildConfig
import com.lsec.translator.data.preferences.UserPreferences
import com.lsec.translator.data.repository.SignRepository
import com.lsec.translator.tts.TTSEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Estado de la UI de configuración.
 */
data class SettingsUiState(
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

    // Servidor
    val serverUrl: String = BuildConfig.DEFAULT_BACKEND_URL,
    val apiKey: String = "",

    // Datos
    val signCount: Int = 0,

    // Estados de UI
    val isLoading: Boolean = false,
    val message: String? = null
)

/**
 * ViewModel para la pantalla de configuración.
 */
@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val userPreferences: UserPreferences,
    private val ttsEngine: TTSEngine,
    private val signRepository: SignRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    init {
        loadPreferences()
        loadSignCount()
    }

    private fun loadPreferences() {
        viewModelScope.launch {
            userPreferences.preferencesFlow.collect { prefs ->
                _uiState.update {
                    it.copy(
                        isTTSEnabled = prefs.isTTSEnabled,
                        speechRate = prefs.speechRate,
                        pitch = prefs.pitch,
                        preferEcuadorianVariants = prefs.preferEcuadorianVariants,
                        useFrontCamera = prefs.useFrontCamera,
                        minConfidence = prefs.minConfidence,
                        batterySaverMode = prefs.batterySaverMode,
                        darkMode = prefs.darkMode,
                        serverUrl = prefs.serverUrl,
                        apiKey = prefs.apiKey
                    )
                }
            }
        }
    }

    private fun loadSignCount() {
        viewModelScope.launch {
            val count = signRepository.getTotalCount()
            _uiState.update { it.copy(signCount = count) }
        }
    }

    // Configuración de Voz

    fun setTTSEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setTTSEnabled(enabled)
            _uiState.update { it.copy(isTTSEnabled = enabled) }
        }
    }

    fun setSpeechRate(rate: Float) {
        viewModelScope.launch {
            userPreferences.setSpeechRate(rate)
            ttsEngine.setSpeechRate(rate)
            _uiState.update { it.copy(speechRate = rate) }
        }
    }

    fun setPitch(pitch: Float) {
        viewModelScope.launch {
            userPreferences.setPitch(pitch)
            ttsEngine.setPitch(pitch)
            _uiState.update { it.copy(pitch = pitch) }
        }
    }

    fun setPreferEcuadorianVariants(prefer: Boolean) {
        viewModelScope.launch {
            userPreferences.setPreferEcuadorianVariants(prefer)
            _uiState.update { it.copy(preferEcuadorianVariants = prefer) }
        }
    }

    // Configuración de Cámara

    fun setUseFrontCamera(useFront: Boolean) {
        viewModelScope.launch {
            userPreferences.setUseFrontCamera(useFront)
            _uiState.update { it.copy(useFrontCamera = useFront) }
        }
    }

    fun setMinConfidence(confidence: Float) {
        viewModelScope.launch {
            userPreferences.setMinConfidence(confidence)
            _uiState.update { it.copy(minConfidence = confidence) }
        }
    }

    fun setBatterySaverMode(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setBatterySaverMode(enabled)
            _uiState.update { it.copy(batterySaverMode = enabled) }
        }
    }

    // Configuración de Apariencia

    fun setDarkMode(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setDarkMode(enabled)
            _uiState.update { it.copy(darkMode = enabled) }
        }
    }

    // Configuración de Servidor

    fun setServerUrl(url: String) {
        viewModelScope.launch {
            val normalizedUrl = if (url.endsWith("/")) url else "$url/"
            userPreferences.setServerUrl(normalizedUrl)
            _uiState.update {
                it.copy(
                    serverUrl = normalizedUrl,
                    message = "URL guardada. Reinicia la app para aplicar."
                )
            }
        }
    }

    fun setApiKey(key: String) {
        viewModelScope.launch {
            userPreferences.setApiKey(key.trim())
            _uiState.update { it.copy(apiKey = key.trim(), message = "Clave API guardada.") }
        }
    }

    // Acciones de Datos

    fun updateDictionary() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            // TODO: Implementar descarga de diccionario actualizado desde servidor
            kotlinx.coroutines.delay(1500)
            _uiState.update {
                it.copy(
                    isLoading = false,
                    message = "Diccionario actualizado"
                )
            }
        }
    }

    fun clearCache() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true) }
            // TODO: Implementar limpieza de caché
            kotlinx.coroutines.delay(500)
            _uiState.update {
                it.copy(
                    isLoading = false,
                    message = "Caché limpiada"
                )
            }
        }
    }

    fun openHelp() {
        // TODO: Navegar a pantalla de ayuda o abrir URL de documentación
        _uiState.update { it.copy(message = "Próximamente: sección de ayuda") }
    }

    fun clearMessage() {
        _uiState.update { it.copy(message = null) }
    }
}
