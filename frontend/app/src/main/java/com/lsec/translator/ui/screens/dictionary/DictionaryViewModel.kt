package com.lsec.translator.ui.screens.dictionary

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.data.local.entity.SignCategory
import com.lsec.translator.data.local.entity.SignEntity
import com.lsec.translator.data.repository.SignRepository
import com.lsec.translator.tts.TTSEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Estado de la UI del diccionario.
 */
data class DictionaryUiState(
    val signs: List<SignEntity> = emptyList(),
    val selectedCategory: SignCategory? = null,
    val selectedTab: Int = 0, // 0 = Todas, 1 = Favoritas, 2 = Frecuentes
    val searchQuery: String = "",
    val isLoading: Boolean = true,
    val totalCount: Int = 0
)

/**
 * ViewModel para la pantalla de diccionario.
 */
@HiltViewModel
class DictionaryViewModel @Inject constructor(
    private val repository: SignRepository,
    private val ttsEngine: TTSEngine
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(DictionaryUiState())
    val uiState: StateFlow<DictionaryUiState> = _uiState.asStateFlow()
    
    private val searchQueryFlow = MutableStateFlow("")
    private val selectedCategoryFlow = MutableStateFlow<SignCategory?>(null)
    private val selectedTabFlow = MutableStateFlow(0)
    
    init {
        observeSigns()
        loadInitialData()
    }
    
    @OptIn(ExperimentalCoroutinesApi::class)
    private fun observeSigns() {
        viewModelScope.launch {
            combine(
                searchQueryFlow,
                selectedCategoryFlow,
                selectedTabFlow
            ) { query, category, tab ->
                Triple(query, category, tab)
            }.flatMapLatest { (query, category, tab) ->
                when {
                    query.isNotEmpty() -> repository.searchSigns(query)
                    tab == 1 -> repository.getFavoriteSigns()
                    tab == 2 -> repository.getMostUsedSigns()
                    category != null -> repository.getSignsByCategory(category)
                    else -> repository.getAllSigns()
                }
            }.collect { signs ->
                _uiState.update { 
                    it.copy(
                        signs = signs,
                        isLoading = false
                    )
                }
            }
        }
    }
    
    private fun loadInitialData() {
        viewModelScope.launch {
            val totalCount = repository.getTotalCount()
            _uiState.update { it.copy(totalCount = totalCount) }
            
            // Si no hay datos, cargar datos iniciales de ejemplo
            if (totalCount == 0) {
                populateInitialData()
            }
        }
    }
    
    /**
     * Pobla la base de datos con datos iniciales de señas LSEC.
     */
    private suspend fun populateInitialData() {
        val initialSigns = listOf(
            // Saludos
            SignEntity(
                name = "Hola",
                spanishTranslation = "Hola",
                ecuadorianVariant = "¡Qué más!",
                description = "Saludo informal. Mano abierta moviéndose de lado a lado.",
                category = SignCategory.GREETINGS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            SignEntity(
                name = "Buenos días",
                spanishTranslation = "Buenos días",
                ecuadorianVariant = "Buenos días",
                description = "Saludo formal de la mañana.",
                category = SignCategory.GREETINGS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            SignEntity(
                name = "Gracias",
                spanishTranslation = "Gracias",
                ecuadorianVariant = "Gracias, pana",
                description = "Expresión de agradecimiento.",
                category = SignCategory.GREETINGS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Adiós",
                spanishTranslation = "Adiós",
                ecuadorianVariant = "¡Chao, nos vemos!",
                description = "Despedida. Mano abierta moviéndose hacia adelante.",
                category = SignCategory.GREETINGS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            
            // Familia
            SignEntity(
                name = "Mamá",
                spanishTranslation = "Madre",
                ecuadorianVariant = "La vieja",
                description = "Seña para referirse a la madre.",
                category = SignCategory.FAMILY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Papá",
                spanishTranslation = "Padre",
                ecuadorianVariant = "El viejo",
                description = "Seña para referirse al padre.",
                category = SignCategory.FAMILY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Hermano",
                spanishTranslation = "Hermano",
                ecuadorianVariant = "El ñaño",
                description = "Seña para referirse al hermano.",
                category = SignCategory.FAMILY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Hijo",
                spanishTranslation = "Hijo/Niño",
                ecuadorianVariant = "El guagua",
                description = "Seña para referirse a un niño o hijo.",
                category = SignCategory.FAMILY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            
            // Frases comunes
            SignEntity(
                name = "¿Cómo estás?",
                spanishTranslation = "¿Cómo estás?",
                ecuadorianVariant = "¿Qué fue, cómo vas?",
                description = "Pregunta sobre el estado de ánimo.",
                category = SignCategory.COMMON_PHRASES,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            SignEntity(
                name = "Bien",
                spanishTranslation = "Bien",
                ecuadorianVariant = "Chévere",
                description = "Respuesta positiva sobre estado.",
                category = SignCategory.COMMON_PHRASES,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Amigo",
                spanishTranslation = "Amigo",
                ecuadorianVariant = "Pana",
                description = "Seña para referirse a un amigo cercano.",
                category = SignCategory.FAMILY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "Mucho",
                spanishTranslation = "Mucho/Bastante",
                ecuadorianVariant = "Full",
                description = "Indica cantidad grande.",
                category = SignCategory.COMMON_PHRASES,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            
            // Números
            SignEntity(
                name = "1",
                spanishTranslation = "Uno",
                ecuadorianVariant = "Uno",
                description = "Número uno - dedo índice levantado.",
                category = SignCategory.NUMBERS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "2",
                spanishTranslation = "Dos",
                ecuadorianVariant = "Dos",
                description = "Número dos - dedos índice y medio levantados.",
                category = SignCategory.NUMBERS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "5",
                spanishTranslation = "Cinco",
                ecuadorianVariant = "Cinco",
                description = "Número cinco - mano abierta.",
                category = SignCategory.NUMBERS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            
            // Letras
            SignEntity(
                name = "A",
                spanishTranslation = "A",
                ecuadorianVariant = "A",
                description = "Letra A del abecedario dactilológico.",
                category = SignCategory.ALPHABET,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            SignEntity(
                name = "E",
                spanishTranslation = "E",
                ecuadorianVariant = "E",
                description = "Letra E del abecedario dactilológico.",
                category = SignCategory.ALPHABET,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = false
            ),
            
            // Preguntas
            SignEntity(
                name = "¿Qué?",
                spanishTranslation = "¿Qué?",
                ecuadorianVariant = "¿Qué cosa?",
                description = "Pregunta interrogativa.",
                category = SignCategory.QUESTIONS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            SignEntity(
                name = "¿Dónde?",
                spanishTranslation = "¿Dónde?",
                ecuadorianVariant = "¿Dónde es?",
                description = "Pregunta sobre ubicación.",
                category = SignCategory.QUESTIONS,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            ),
            
            // Emergencias
            SignEntity(
                name = "Ayuda",
                spanishTranslation = "Ayuda",
                ecuadorianVariant = "Dame una mano",
                description = "Señal de solicitud de ayuda.",
                category = SignCategory.EMERGENCY,
                difficulty = com.lsec.translator.data.local.entity.SignDifficulty.BEGINNER,
                isDynamic = true
            )
        )
        
        repository.saveSigns(initialSigns)
        
        _uiState.update { it.copy(totalCount = initialSigns.size) }
    }
    
    /**
     * Busca señas por texto.
     */
    fun search(query: String) {
        searchQueryFlow.value = query
        _uiState.update { it.copy(searchQuery = query, isLoading = true) }
    }
    
    /**
     * Filtra por categoría.
     */
    fun filterByCategory(category: SignCategory?) {
        selectedCategoryFlow.value = category
        _uiState.update { it.copy(selectedCategory = category, isLoading = true) }
    }
    
    /**
     * Selecciona tab (Todas, Favoritas, Frecuentes).
     */
    fun selectTab(tabIndex: Int) {
        selectedTabFlow.value = tabIndex
        _uiState.update { it.copy(selectedTab = tabIndex, isLoading = true) }
    }
    
    /**
     * Alterna favorito de una seña.
     */
    fun toggleFavorite(signId: Long) {
        viewModelScope.launch {
            repository.toggleFavorite(signId)
        }
    }
    
    /**
     * Reproduce el audio de una seña.
     */
    fun speakSign(sign: SignEntity) {
        viewModelScope.launch {
            repository.incrementUsage(sign.id)
            val textToSpeak = sign.ecuadorianVariant.ifEmpty { sign.spanishTranslation }
            ttsEngine.speak(textToSpeak)
        }
    }
}
