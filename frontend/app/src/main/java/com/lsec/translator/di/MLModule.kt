package com.lsec.translator.di

import android.content.Context
import com.lsec.translator.ml.HandDetector
import com.lsec.translator.ml.SignClassifier
import com.lsec.translator.tts.TTSEngine
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Módulo Hilt para proveer componentes de Machine Learning.
 */
@Module
@InstallIn(SingletonComponent::class)
object MLModule {
    
    @Provides
    @Singleton
    fun provideHandDetector(
        @ApplicationContext context: Context
    ): HandDetector {
        return HandDetector(context)
    }
    
    @Provides
    @Singleton
    fun provideSignClassifier(
        @ApplicationContext context: Context
    ): SignClassifier {
        return SignClassifier(context)
    }
    
    @Provides
    @Singleton
    fun provideTTSEngine(
        @ApplicationContext context: Context
    ): TTSEngine {
        return TTSEngine(context)
    }
}
