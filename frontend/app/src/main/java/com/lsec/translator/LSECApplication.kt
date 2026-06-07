package com.lsec.translator

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Aplicación principal de LSEC Translator.
 * Inicializa Hilt para inyección de dependencias.
 */
@HiltAndroidApp
class LSECApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        // Inicialización adicional si es necesaria
    }
}
