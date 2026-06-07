package com.lsec.translator.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.lsec.translator.ui.navigation.LSECNavHost
import com.lsec.translator.ui.theme.LSECTranslatorTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Activity principal de la aplicación LSEC Translator.
 * Configura el tema y la navegación principal.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        setContent {
            LSECTranslatorTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    LSECNavHost()
                }
            }
        }
    }
}
