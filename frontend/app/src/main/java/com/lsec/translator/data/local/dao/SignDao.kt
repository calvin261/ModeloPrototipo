package com.lsec.translator.data.local.dao

import androidx.room.*
import com.lsec.translator.data.local.entity.SignCategory
import com.lsec.translator.data.local.entity.SignEntity
import kotlinx.coroutines.flow.Flow

/**
 * DAO para operaciones con señas LSEC.
 */
@Dao
interface SignDao {
    
    /**
     * Obtiene todas las señas.
     */
    @Query("SELECT * FROM signs ORDER BY name ASC")
    fun getAllSigns(): Flow<List<SignEntity>>
    
    /**
     * Obtiene señas por categoría.
     */
    @Query("SELECT * FROM signs WHERE category = :category ORDER BY name ASC")
    fun getSignsByCategory(category: SignCategory): Flow<List<SignEntity>>
    
    /**
     * Obtiene señas favoritas.
     */
    @Query("SELECT * FROM signs WHERE is_favorite = 1 ORDER BY name ASC")
    fun getFavoriteSigns(): Flow<List<SignEntity>>
    
    /**
     * Obtiene señas más usadas.
     */
    @Query("SELECT * FROM signs ORDER BY usage_count DESC LIMIT :limit")
    fun getMostUsedSigns(limit: Int = 20): Flow<List<SignEntity>>
    
    /**
     * Busca señas por nombre o traducción.
     */
    @Query("""
        SELECT * FROM signs 
        WHERE name LIKE '%' || :query || '%' 
        OR spanish_translation LIKE '%' || :query || '%'
        OR ecuadorian_variant LIKE '%' || :query || '%'
        ORDER BY name ASC
    """)
    fun searchSigns(query: String): Flow<List<SignEntity>>
    
    /**
     * Obtiene una seña por ID.
     */
    @Query("SELECT * FROM signs WHERE id = :id")
    suspend fun getSignById(id: Long): SignEntity?
    
    /**
     * Obtiene una seña por nombre.
     */
    @Query("SELECT * FROM signs WHERE name = :name LIMIT 1")
    suspend fun getSignByName(name: String): SignEntity?
    
    /**
     * Inserta una nueva seña.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSign(sign: SignEntity): Long
    
    /**
     * Inserta múltiples señas.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSigns(signs: List<SignEntity>)
    
    /**
     * Actualiza una seña existente.
     */
    @Update
    suspend fun updateSign(sign: SignEntity)
    
    /**
     * Elimina una seña.
     */
    @Delete
    suspend fun deleteSign(sign: SignEntity)
    
    /**
     * Incrementa el contador de uso de una seña.
     */
    @Query("UPDATE signs SET usage_count = usage_count + 1 WHERE id = :signId")
    suspend fun incrementUsageCount(signId: Long)
    
    /**
     * Alterna el estado de favorito de una seña.
     */
    @Query("UPDATE signs SET is_favorite = NOT is_favorite WHERE id = :signId")
    suspend fun toggleFavorite(signId: Long)
    
    /**
     * Obtiene el conteo de señas por categoría.
     */
    @Query("SELECT COUNT(*) FROM signs WHERE category = :category")
    suspend fun getCountByCategory(category: SignCategory): Int
    
    /**
     * Obtiene el conteo total de señas.
     */
    @Query("SELECT COUNT(*) FROM signs")
    suspend fun getTotalCount(): Int
}
