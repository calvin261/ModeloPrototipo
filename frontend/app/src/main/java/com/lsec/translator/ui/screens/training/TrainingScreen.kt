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
import com.lsec.translator.data.remote.ModelInfoDto

private fun sampleText(count: Int) = if (count == 1) "1 muestra" else "$count muestras"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingScreen(
    onNavigateBack: () -> Unit,
    viewModel: TrainingViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var labelDeleteConfirm by remember { mutableStateOf<Pair<String, String>?>(null) }

    labelDeleteConfirm?.let { (label, type) ->
        AlertDialog(
            onDismissRequest = { labelDeleteConfirm = null },
            title = { Text("Eliminar \"$label\"") },
            text = { Text("¿Eliminar todas las muestras de \"$label\"? Esta acción no se puede deshacer.") },
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
            // === Estado de conexión ===
            item {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (uiState.isConnected)
                            MaterialTheme.colorScheme.primaryContainer
                        else
                            MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            if (uiState.isConnected) Icons.Default.Cloud else Icons.Default.CloudOff,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            if (uiState.isConnected) "Conectado al servidor"
                            else "Sin conexión al servidor"
                        )
                    }
                }
            }

            // === Resumen de datos capturados ===
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
                }
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
                }
            }

            // === Sección de Modelos (mejorada) ===
            if (uiState.isConnected) {
                item {
                    Text("Modelos", style = MaterialTheme.typography.titleMedium)
                }
                // Modelos del clasificador
                val classifierModels = uiState.serverModels.filter { it.name == "classifier" }
                    .sortedByDescending { it.createdAt }
                val sequenceModels = uiState.serverModels.filter { it.name == "sequence" }
                    .sortedByDescending { it.createdAt }
                item {
                    ModelSectionCard(
                        title = "Clasificador (alfabeto)",
                        icon = Icons.Default.TextFields,
                        iconTint = MaterialTheme.colorScheme.primary,
                        serverModels = classifierModels,
                        activeVersion = uiState.activeClassifierVersion,
                        hasLocal = uiState.hasLocalClassifier,
                        isDownloading = uiState.isDownloading && uiState.downloadingType == "classifier",
                        onDownload = { viewModel.downloadLatest("classifier") },
                        onDownloadVersion = { v -> viewModel.downloadVersion("classifier", v) },
                    )
                }
                item {
                    ModelSectionCard(
                        title = "Vocabulario (palabras dinámicas)",
                        icon = Icons.Default.Timeline,
                        iconTint = MaterialTheme.colorScheme.tertiary,
                        serverModels = sequenceModels,
                        activeVersion = uiState.activeSequenceVersion,
                        hasLocal = uiState.hasLocalSequence,
                        isDownloading = uiState.isDownloading && uiState.downloadingType == "sequence",
                        onDownload = { viewModel.downloadLatest("sequence") },
                        onDownloadVersion = { v -> viewModel.downloadVersion("sequence", v) },
                    )
                }

                // Mensaje de descarga activa
                if (uiState.isDownloading) {
                    item {
                        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                            Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(18.dp),
                                        strokeWidth = 2.dp,
                                    )
                                    Spacer(Modifier.width(12.dp))
                                    Text(
                                        uiState.downloadMessage ?: "Descargando...",
                                        style = MaterialTheme.typography.bodyMedium,
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // === Entrenamiento ===
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

            // === Progreso del entrenamiento activo ===
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
                        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
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
                                    "Época ${uiState.epoch}/${uiState.totalEpochs}",
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
                                    onClick = {
                                        val type = uiState.activeModelType ?: "classifier"
                                        viewModel.downloadLatest(type)
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    enabled = !uiState.isDownloading,
                                ) {
                                    Icon(Icons.Default.Download, contentDescription = null)
                                    Spacer(Modifier.width(4.dp))
                                    Text("Descargar modelo actualizado")
                                }
                            }
                        }
                    }
                }
            }

            // === Mensaje de descarga / error ===
            if (!uiState.isDownloading) {
                uiState.downloadMessage?.let { message ->
                    item {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = if (message.startsWith("Error"))
                                    MaterialTheme.colorScheme.errorContainer
                                else
                                    MaterialTheme.colorScheme.tertiaryContainer
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

            // === Historial ===
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

/**
 * Card de sección de modelo. Muestra:
 *  - Versión activa actual (local) con badge "ACTIVO"
 *  - Versión más reciente del servidor con botón de descarga
 *  - Lista de versiones históricas con su accuracy
 */
@Composable
private fun ModelSectionCard(
    title: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    iconTint: Color,
    serverModels: List<ModelInfoDto>,
    activeVersion: String,
    hasLocal: Boolean,
    isDownloading: Boolean,
    onDownload: () -> Unit,
    onDownloadVersion: (String) -> Unit,
) {
    val latest = serverModels.firstOrNull()
    val isLatestActive = latest != null && activeVersion == latest.version
    val showDownloadLatest = latest != null && !isLatestActive

    Card(
        colors = CardDefaults.cardColors(
            containerColor = if (isLatestActive)
                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
            else
                MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            // Header
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, contentDescription = null, tint = iconTint)
                Spacer(Modifier.width(8.dp))
                Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            }
            Spacer(Modifier.height(10.dp))

            // ACTIVE
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    if (hasLocal) Icons.Default.CheckCircle else Icons.Default.CloudDownload,
                    contentDescription = null,
                    tint = if (hasLocal) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(18.dp),
                )
                Spacer(Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (hasLocal) "Activo" else "Sin modelo local",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        activeVersion,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (isLatestActive) {
                    AssistChip(
                        onClick = {},
                        label = { Text("AL DÍA") },
                        leadingIcon = {
                            Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(14.dp))
                        },
                    )
                }
            }

            // LATEST ON SERVER
            if (latest != null) {
                Spacer(Modifier.height(8.dp))
                HorizontalDivider()
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.Cloud,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Última versión",
                            style = MaterialTheme.typography.labelMedium,
                        )
                        Text(
                            "${latest.version} · ${"%.1f".format(latest.accuracy * 100)}% · ${latest.numClasses} clases",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (showDownloadLatest) {
                        Button(
                            onClick = onDownload,
                            enabled = !isDownloading,
                        ) {
                            if (isDownloading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary,
                                )
                                Spacer(Modifier.width(6.dp))
                            } else {
                                Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(4.dp))
                            }
                            Text("Descargar", style = MaterialTheme.typography.labelMedium)
                        }
                    } else {
                        Text(
                            "✓ ya activo",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            } else {
                Spacer(Modifier.height(8.dp))
                Text(
                    "No hay versiones en el servidor.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // HISTORY
            if (serverModels.size > 1) {
                Spacer(Modifier.height(10.dp))
                Text(
                    "Otras versiones (${serverModels.size - 1})",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.height(4.dp))
                serverModels.drop(1).forEach { m ->
                    ModelHistoryRow(
                        model = m,
                        isActive = m.version == activeVersion,
                        isDownloading = isDownloading,
                        onDownload = { onDownloadVersion(m.version) },
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelHistoryRow(
    model: ModelInfoDto,
    isActive: Boolean,
    isDownloading: Boolean,
    onDownload: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            Icons.Default.History,
            contentDescription = null,
            modifier = Modifier.size(14.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.width(8.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                model.version,
                style = MaterialTheme.typography.bodySmall,
                fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
                color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
            )
            Text(
                "${"%.1f".format(model.accuracy * 100)}% · ${model.sizeKb.toInt()}KB · ${model.numClasses} clases",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (isActive) {
            AssistChip(
                onClick = {},
                label = { Text("ACTIVO", style = MaterialTheme.typography.labelSmall) },
            )
        } else {
            IconButton(
                onClick = onDownload,
                enabled = !isDownloading,
                modifier = Modifier.size(32.dp),
            ) {
                Icon(
                    Icons.Default.Download,
                    contentDescription = "Descargar ${model.version}",
                    modifier = Modifier.size(16.dp),
                )
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
            sortedLabels.forEach { (label, count) ->
                LabelRow(
                    label = label,
                    count = count,
                    recommendedMin = recommendedMin,
                    onDelete = { onDelete(label) }
                )
            }
            if (poorLabels > 0) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
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
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.weight(1f)
        )
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
