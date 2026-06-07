package com.lsec.translator.ui.screens.home

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * Pantalla principal de la aplicación LSEC Translator.
 * Muestra opciones para acceder a la cámara, diccionario y configuración.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onNavigateToCamera: () -> Unit,
    onNavigateToDictionary: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToDataCapture: () -> Unit = {},
    onNavigateToTraining: () -> Unit = {}
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "LSEC Translator",
                        style = MaterialTheme.typography.headlineMedium
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Título descriptivo
            Text(
                text = "Traductor de Lengua de Señas Ecuatoriana",
                style = MaterialTheme.typography.titleLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            Text(
                text = "Reconocimiento en tiempo real con voz sintetizada",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            // Botón principal - Iniciar Traducción
            HomeButton(
                icon = Icons.Default.CameraAlt,
                title = "Iniciar Traducción",
                description = "Captura señas LSEC en tiempo real",
                onClick = onNavigateToCamera,
                isPrimary = true
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Botón secundario - Diccionario
            HomeButton(
                icon = Icons.Default.MenuBook,
                title = "Diccionario LSEC",
                description = "Explora el vocabulario disponible",
                onClick = onNavigateToDictionary,
                isPrimary = false
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Botón secundario - Configuración
            HomeButton(
                icon = Icons.Default.Settings,
                title = "Configuración",
                description = "Ajusta preferencias y voz",
                onClick = onNavigateToSettings,
                isPrimary = false
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Botón - Captura de Datos
            HomeButton(
                icon = Icons.Default.DatasetLinked,
                title = "Captura de Datos",
                description = "Contribuye datos para entrenar el modelo",
                onClick = onNavigateToDataCapture,
                isPrimary = false
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Botón - Entrenamiento
            HomeButton(
                icon = Icons.Default.ModelTraining,
                title = "Entrenamiento",
                description = "Entrena y actualiza los modelos",
                onClick = onNavigateToTraining,
                isPrimary = false
            )

            Spacer(modifier = Modifier.height(32.dp))
            
            // Información de versión
            Text(
                text = "Versión 1.0.0 • Precisión objetivo: 85%+",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun HomeButton(
    icon: ImageVector,
    title: String,
    description: String,
    onClick: () -> Unit,
    isPrimary: Boolean
) {
    val containerColor = if (isPrimary) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.surfaceVariant
    }
    
    val contentColor = if (isPrimary) {
        MaterialTheme.colorScheme.onPrimary
    } else {
        MaterialTheme.colorScheme.onSurfaceVariant
    }

    Card(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .height(100.dp),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = containerColor)
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = title,
                modifier = Modifier.size(48.dp),
                tint = contentColor
            )
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    color = contentColor
                )
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = contentColor.copy(alpha = 0.8f)
                )
            }
        }
    }
}
