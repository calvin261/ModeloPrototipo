package com.lsec.translator.di

import android.content.Context
import androidx.room.Room
import com.lsec.translator.data.local.LSECDatabase
import com.lsec.translator.data.local.dao.SignDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Módulo Hilt para proveer la base de datos Room.
 */
@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    
    @Provides
    @Singleton
    fun provideLSECDatabase(
        @ApplicationContext context: Context
    ): LSECDatabase {
        return Room.databaseBuilder(
            context,
            LSECDatabase::class.java,
            LSECDatabase.DATABASE_NAME
        )
        .fallbackToDestructiveMigration()
        .build()
    }
    
    @Provides
    @Singleton
    fun provideSignDao(database: LSECDatabase): SignDao {
        return database.signDao()
    }
}
