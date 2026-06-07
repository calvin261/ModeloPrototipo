package com.lsec.translator.data.repository

import com.lsec.translator.data.local.dao.SignDao
import com.lsec.translator.data.local.entity.SignCategory
import com.lsec.translator.data.local.entity.SignEntity
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repositorio para gestión de señas LSEC.
 * Abstrae la fuente de datos (local/remota) de la capa de dominio.
 */
@Singleton
class SignRepository @Inject constructor(
    private val signDao: SignDao
) {
    /**
     * Obtiene todas las señas.
     */
    fun getAllSigns(): Flow<List<SignEntity>> = signDao.getAllSigns()
    
    /**
     * Obtiene señas por categoría.
     */
    fun getSignsByCategory(category: SignCategory): Flow<List<SignEntity>> = 
        signDao.getSignsByCategory(category)
    
    /**
     * Obtiene señas favoritas.
     */
    fun getFavoriteSigns(): Flow<List<SignEntity>> = signDao.getFavoriteSigns()
    
    /**
     * Obtiene señas más usadas.
     */
    fun getMostUsedSigns(limit: Int = 20): Flow<List<SignEntity>> = 
        signDao.getMostUsedSigns(limit)
    
    /**
     * Busca señas.
     */
    fun searchSigns(query: String): Flow<List<SignEntity>> = signDao.searchSigns(query)
    
    /**
     * Obtiene una seña por ID.
     */
    suspend fun getSignById(id: Long): SignEntity? = signDao.getSignById(id)
    
    /**
     * Obtiene una seña por nombre.
     */
    suspend fun getSignByName(name: String): SignEntity? = signDao.getSignByName(name)
    
    /**
     * Guarda una nueva seña.
     */
    suspend fun saveSign(sign: SignEntity): Long = signDao.insertSign(sign)
    
    /**
     * Guarda múltiples señas.
     */
    suspend fun saveSigns(signs: List<SignEntity>) = signDao.insertSigns(signs)
    
    /**
     * Actualiza una seña.
     */
    suspend fun updateSign(sign: SignEntity) = signDao.updateSign(sign)
    
    /**
     * Elimina una seña.
     */
    suspend fun deleteSign(sign: SignEntity) = signDao.deleteSign(sign)
    
    /**
     * Incrementa el contador de uso.
     */
    suspend fun incrementUsage(signId: Long) = signDao.incrementUsageCount(signId)
    
    /**
     * Alterna favorito.
     */
    suspend fun toggleFavorite(signId: Long) = signDao.toggleFavorite(signId)
    
    /**
     * Obtiene estadísticas.
     */
    suspend fun getTotalCount(): Int = signDao.getTotalCount()
    
    suspend fun getCountByCategory(category: SignCategory): Int = 
        signDao.getCountByCategory(category)
}
