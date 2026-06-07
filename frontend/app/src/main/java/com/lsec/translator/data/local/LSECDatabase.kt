package com.lsec.translator.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.lsec.translator.data.local.dao.SignDao
import com.lsec.translator.data.local.entity.SignEntity

/**
 * Base de datos Room para el diccionario LSEC.
 */
@Database(
    entities = [SignEntity::class],
    version = 1,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class LSECDatabase : RoomDatabase() {
    abstract fun signDao(): SignDao
    
    companion object {
        const val DATABASE_NAME = "lsec_database"
    }
}
