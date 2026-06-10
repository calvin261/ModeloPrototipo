package com.lsec.translator.di

import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

/**
 * Módulo Hilt para proveer componentes de Machine Learning.
 *
 * NOTA: HandDetector, SignClassifier y TTSEngine ya declaran
 * `@Inject constructor(@ApplicationContext context: Context)` + `@Singleton`
 * en sus propias clases, por lo que se inyectan automáticamente y NO
 * necesitan un `@Provides` aquí. Mantenerlos duplicados provoca
 * `Hilt/DuplicateBindings` en tiempo de ejecución y ViewModels que
 * dependen de ellos lanzan `NoSuchMethodException` al no poder
 * resolver el grafo.
 */
@Module
@InstallIn(SingletonComponent::class)
object MLModule
