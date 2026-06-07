package com.lsec.translator.ui.screens.training

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

private fun sampleText(count: Int) = if (count == 1) "1 muestra" else "$count muestras"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingScreen(
    onNavigateBack: () -> Unit,
    viewModel: TrainingViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    // Pair<label, type> where type = "static" or "sequence"
    var labelDeleteConfirm by remember { mutableStateOf<Pair<String, String>?>(null) }

    // Confirmation dialog for label deletion
    labelDeleteConfirm?.let { (label, type) ->
        AlertDialog(
            onDismissRequest = { labelDeleteConfirm = null },
            title = { Text("Eliminar \"$label\"") },
            text = {
                Text("¿Eliminar todas las muestras de \"$label\"? Esta acción no se puede deshacer.")
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (type == "static") viewModel.deleteStaticLabel(label)
                        else viewModel.deleteSequenceLabel(label)
                        labelDeleteConfirm = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                ) { Text("Eliminar") }
            },
            dismissButton = {
                OutlinedButton(onClick = { labelDeleteConfirm = null }) { Text("Cancelar") }
            }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Entrenamiento") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Volver")
                    }
                },
                actions = {
                    IconButton(onClick = { viewModel.refreshStats() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Actualizar")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Estado de conexion
            item {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (uiState.isConnected)
                            MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            if (uiState.isConnected) Icons.Default.Cloud else Icons.Default.CloudOff,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            if (uiState.isConnected) "Conectado al servidor"
                            else "Sin conexion al servidor"
                        )
                    }
                }
            }

            // === Señas estáticas ===
            uiState.dataStats?.let { stats ->
                if (stats.staticLabels.isNotEmpty()) {
                    item {
                        LabelManagementCard(
                            title = "Señas estáticas",
                            totalSamples = stats.staticSamples,
                            labelCounts = stats.staticLabels,
                            recommendedMin = 5,
                            onDelete = { label -> labelDeleteConfirm = Pair(label, "static") }
                        )
                    }
                } else if (uiState.isConnected) {
                    item {
                        EmptyDataCard(
                            message = "Sin señas estáticas capturadas. Ve a la pantalla de Captura de Datos para comenzar."
                        )
                    }
                }

                // === Secuencias ===
                if (stats.sequenceLabels.isNotEmpty()) {
                    item {
                        LabelManagementCard(
                            title = "Secuencias",
                            totalSamples = stats.sequenceSamples,
                            labelCounts = stats.sequenceLabels,
                            recommendedMin = 3,
                            onDelete = { label -> labelDeleteConfirm = Pair(label, "sequence") }
                        )
                    }
                } else if (uiState.isConnected) {
                    item {
                        EmptyDataCard(
                            message = "Sin secuencias capturadas. Graba señas dinámicas desde la pantalla de Captura."
                        )
                    }
                }
            }

            // Descargar modelos (siempre visible cuando hay conexión)
            if (uiState.isConnected) {
                item {
                    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                            Text("Descargar modelos", style = MaterialTheme.typography.titleSmall)
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                "Descarga el modelo entrenado al dispositivo para usar en traducción en vivo.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Button(
                                    onClick = { viewModel.downloadModel("classifier") },
                                    modifier = Modifier.weight(1f),
                                    enabled = !uiState.isDownloading
                                ) {
                                    Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Clasificador")
                                }
                                OutlinedButton(
                                    onClick = { viewModel.downloadModel("sequence") },
                                    modifier = Modifier.weight(1f),
                                    enabled = !uiState.isDownloading
                                ) {
                                    Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text("Vocabulario")
                                }
                            }
                            if (uiState.isDownloading) {
                                Spacer(modifier = Modifier.height(8.dp))
                                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    uiState.downloadMessage ?: "Descargando...",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }
            }

            // Botones de entrenamiento
            item {
                Text("Iniciar entrenamiento", style = MaterialTheme.typography.titleMedium)
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    val isTraining = uiState.trainingStatus == "running" || uiState.trainingStatus == "pending"

                    OutlinedButton(
                        onClick = { viewModel.startClassifierTraining() },
                        modifier = Modifier.weight(1f),
                        enabled = uiState.isConnected && !isTraining
                    ) {
                        Icon(Icons.Default.Psychology, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Clasificador")
                    }

                    OutlinedButton(
                        onClick = { viewModel.startSequenceTraining() },
                        modifier = Modifier.weight(1f),
                        enabled = uiState.isConnected && !isTraining
                    ) {
                        Icon(Icons.Default.Timeline, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Secuencias")
                    }
                }
            }

            // Progreso de entrenamiento activo
            if (uiState.trainingStatus != "idle") {
                item {
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = when (uiState.trainingStatus) {
                                "completed" -> MaterialTheme.colorScheme.primaryContainer
                                "failed" -> MaterialTheme.colorScheme.errorContainer
                                else -> MaterialTheme.colorScheme.surfaceVariant
                            }
                        )
                    ) {
                        Column(
                            modifier = Modifier.fillMaxWidth().padding(16.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "Entrenamiento: ${uiState.activeModelType ?: ""}",
                                    style = MaterialTheme.typography.titleSmall
                                )
                                AssistChip(onClick = {}, label = { Text(uiState.trainingStatus) })
                            }

                            if (uiState.trainingStatus == "running") {
                                Spacer(modifier = Modifier.height(8.dp))
                                LinearProgressIndicator(
                                    progress = {
                                        if (uiState.totalEpochs > 0)
                                            uiState.epoch.toFloat() / uiState.totalEpochs
                                        else 0f
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    "Epoca ${uiState.epoch}/${uiState.totalEpochs}",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }

                            Spacer(modifier = Modifier.height(8.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text("Loss: ${"%.4f".format(uiState.trainLoss)}", style = MaterialTheme.typography.bodySmall)
                                Text("Acc: ${"%.1f".format(uiState.valAccuracy * 100)}%", style = MaterialTheme.typography.bodySmall)
                                Text("Mejor: ${"%.1f".format(uiState.bestAccuracy * 100)}%", style = MaterialTheme.typography.bodySmall)
                            }

                            if (uiState.trainingStatus == "completed") {
                                Spacer(modifier = Modifier.height(12.dp))
                                Button(
                                    onClick = { viewModel.downloadModel() },
                                    modifier = Modifier.fillMaxWidth(),
                                    enabled = !uiState.isDownloading
                                ) {
                                    if (uiState.isDownloading) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(18.dp),
                                            strokeWidth = 2.dp,
                                            color = MaterialTheme.colorScheme.onPrimary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                    } else {
                                        Icon(Icons.Default.Download, contentDescription = null)
                                        Spacer(modifier = Modifier.width(4.dp))
                                    }
                                    Text(
                                        if (uiState.isDownloading)
                                            uiState.downloadMessage ?: "Descargando..."
                                        else
                                            "Descargar modelo actualizado"
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // Mensaje de descarga (cuando no está descargando activamente)
            if (!uiState.isDownloading) {
                uiState.downloadMessage?.let { message ->
                    item {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = if (message.startsWith("Error"))
                                    MaterialTheme.colorScheme.errorContainer
                                else MaterialTheme.colorScheme.tertiaryContainer
                            )
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(12.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    if (message.startsWith("Error")) Icons.Default.Warning else Icons.Default.CheckCircle,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    message,
                                    style = MaterialTheme.typography.bodySmall,
                                    modifier = Modifier.weight(1f)
                                )
                                IconButton(onClick = { viewModel.clearDownloadMessage() }) {
                                    Icon(Icons.Default.Close, contentDescription = "Cerrar", modifier = Modifier.size(16.dp))
                                }
                            }
                        }
                    }
                }
            }

            // Error
            uiState.error?.let { error ->
                item {
                    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(error, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                            IconButton(onClick = { viewModel.clearError() }) {
                                Icon(Icons.Default.Close, contentDescription = "Cerrar", modifier = Modifier.size(16.dp))
                            }
                        }
                    }
                }
            }

            // Historial de entrenamientos
            if (uiState.history.isNotEmpty()) {
                item {
                    Text("Historial", style = MaterialTheme.typography.titleMedium)
                }
                items(uiState.history.reversed()) { entry ->
                    Card {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(12.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(entry.modelType, style = MaterialTheme.typography.bodyMedium)
                                Text(
                                    "${entry.epochs} épocas · Mejor acc: ${"%.1f".format(entry.bestAccuracy * 100)}%",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                            AssistChip(onClick = {}, label = { Text(entry.status) })
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LabelManagementCard(
    title: String,
    totalSamples: Int,
    labelCounts: Map<String, Int>,
    recommendedMin: Int,
    onDelete: (String) -> Unit
) {
    val poorLabels = labelCounts.count { it.value < recommendedMin }
    val sortedLabels = labelCounts.entries.sortedBy { it.key }

    Card {
        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            // Header row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(title, style = MaterialTheme.typography.titleMedium)
                    Text(
                        "${sampleText(totalSamples)} · ${labelCounts.size} clases",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (poorLabels > 0) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.tertiaryContainer
                    ) {
                        Text(
                            "$poorLabels con pocos datos",
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            HorizontalDivider()
            Spacer(modifier = Modifier.height(4.dp))

            // Label rows
            sortedLabels.forEach { (label, count) ->
                LabelRow(
                    label = label,
                    count = count,
                    recommendedMin = recommendedMin,
                    onDelete = { onDelete(label) }
                )
            }

            // Warning if some labels have few samples
            if (poorLabels > 0) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.Info,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        "Recomendado: al menos $recommendedMin muestras por clase",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun LabelRow(
    label: String,
    count: Int,
    recommendedMin: Int,
    onDelete: () -> Unit
) {
    val badgeColor = when {
        count >= recommendedMin * 2 -> MaterialTheme.colorScheme.primary
        count >= recommendedMin -> MaterialTheme.colorScheme.secondary
        count > 1 -> MaterialTheme.colorScheme.tertiary
        else -> MaterialTheme.colorScheme.error
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Label name
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.weight(1f)
        )

        // Count badge
        Surface(
            shape = RoundedCornerShape(12.dp),
            color = badgeColor.copy(alpha = 0.15f)
        ) {
            Text(
                sampleText(count),
                style = MaterialTheme.typography.labelSmall,
                color = badgeColor,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
            )
        }

        // Delete button
        IconButton(
            onClick = onDelete,
            modifier = Modifier.size(32.dp)
        ) {
            Icon(
                Icons.Default.Delete,
                contentDescription = "Eliminar $label",
                modifier = Modifier.size(16.dp),
                tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun EmptyDataCard(message: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.Info,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
