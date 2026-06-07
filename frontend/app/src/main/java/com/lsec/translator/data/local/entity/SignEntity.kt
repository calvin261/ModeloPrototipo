package com.lsec.translator.data.local.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Entidad que representa una seña LSEC en la base de datos.
 */
@Entity(tableName = "signs")
data class SignEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    
    @ColumnInfo(name = "name")
    val name: String,
    
    @ColumnInfo(name = "spanish_translation")
    val spanishTranslation: String,
    
    @ColumnInfo(name = "ecuadorian_variant")
    val ecuadorianVariant: String,
    
    @ColumnInfo(name = "description")
    val description: String,
    
    @ColumnInfo(name = "category")
    val category: SignCategory,
    
    @ColumnInfo(name = "difficulty")
    val difficulty: SignDifficulty,
    
    @ColumnInfo(name = "video_url")
    val videoUrl: String? = null,
    
    @ColumnInfo(name = "image_url")
    val imageUrl: String? = null,
    
    @ColumnInfo(name = "is_dynamic")
    val isDynamic: Boolean = true, // true = requiere movimiento, false = estática
    
    @ColumnInfo(name = "regional_variant")
    val regionalVariant: RegionalVariant = RegionalVariant.QUITO,
    
    @ColumnInfo(name = "usage_count")
    val usageCount: Int = 0,
    
    @ColumnInfo(name = "is_favorite")
    val isFavorite: Boolean = false,
    
    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis(),
    
    @ColumnInfo(name = "updated_at")
    val updatedAt: Long = System.currentTimeMillis()
)

/**
 * Categorías de señas LSEC.
 */
enum class SignCategory {
    ALPHABET,       // Letras A-Z, Ñ
    NUMBERS,        // Números 0-99
    GREETINGS,      // Saludos y despedidas
    FAMILY,         // Familia y relaciones
    FOOD,           // Comida y bebida
    HEALTH,         // Salud y cuerpo
    EDUCATION,      // Educación y trabajo
    EMOTIONS,       // Emociones y sentimientos
    TIME,           // Tiempo y calendario
    PLACES,         // Lugares y direcciones
    ACTIONS,        // Verbos y acciones
    QUESTIONS,      // Preguntas (qué, cómo, dónde...)
    COMMON_PHRASES, // Frases cotidianas
    EMERGENCY,      // Señas de emergencia
    OTHER           // Otras señas
}

/**
 * Nivel de dificultad de la seña.
 */
enum class SignDifficulty {
    BEGINNER,       // Principiante
    INTERMEDIATE,   // Intermedio
    ADVANCED        // Avanzado
}

/**
 * Variantes regionales del LSEC.
 */
enum class RegionalVariant {
    QUITO,          // Quito y Sierra Norte
    GUAYAQUIL,      // Guayaquil y Costa
    CUENCA,         // Cuenca y Sierra Sur
    NATIONAL        // Estándar nacional
}
