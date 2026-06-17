package com.lsec.translator.ui.screens.camera

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Pantalla de calibracion de 30 segundos.
 *
 * El usuario hace 3-5 muestras de cada una de las 12 senas del
 * vocabulario. Por cada muestra:
 *   1. La camera captura el gesto
 *   2. LandmarksNormalization produce (T, 63)
 *   3. SignTransformerClassifier.extractEmbedding() devuelve FloatArray(128)
 *   4. CentroidClassifier.setCentroids() promedia los embeddings por clase
 *
 * Al final de la calibracion el facade (`SignClassifier.processDynamicFrame`)
 * usara cosine similarity contra los centroides en vez del label del
 * transformer para re-decidir el resultado.
 *
 * Esta version es un stub: contiene el esqueleto de navegacion y
 * composables para que el desarrollador de Android Studio rellene el
 * flujo de captura y el countdown de 30s.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CalibrationScreen(
    onNavigateBack: () -> Unit,
    // viewModel: CalibrationViewModel = hiltViewModel()  // TODO: inyectar cuando exista
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Calibración (30s)") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Volver")
                    }
                },
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentAlignment = Alignment.Center,
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.padding(24.dp)
            ) {
                Text(
                    text = "Calibración de señas",
                    style = MaterialTheme.typography.headlineMedium,
                )
                Text(
                    text = "Captura 3-5 ejemplos de cada seña para personalizar el reconocedor.",
                    style = MaterialTheme.typography.bodyMedium,
                )

                // TODO(android-dev): mostrar un progress indicator con el
                // tiempo restante del countdown de 30s. Fuente:
                //   val deadline = System.currentTimeMillis() + 30_000
                //   val remaining = (deadline - System.currentTimeMillis()) / 1000
                //   Text("Tiempo restante: ${remaining}s")

                // TODO(android-dev): preview de la camara + overlay con la
                // lista de las 12 senas. Marcar visualmente la que se esta
                // capturando. Reusar CameraPreviewContent de CameraScreen.kt.

                // TODO(android-dev): boton "Iniciar" que arranca el
                // countdown. Mientras corre, capturar N muestras de cada
                // clase y llamar:
                //   val embedding = signClassifier.transformer()
                //       ?.extractEmbedding(sequence)
                //   perSignEmbeddings[currentSign] = perSignEmbeddings
                //       .getOrDefault(currentSign, emptyList()) + listOf(embedding)
                //   signClassifier.centroids()?.setCentroids(perSignEmbeddings)

                // TODO(android-dev): al terminar, mostrar resumen
                // ("Capturadas 36 muestras en 30s") y boton "Listo" que
                // navega de vuelta a CameraScreen con popBackStack().
            }
        }
    }
}
