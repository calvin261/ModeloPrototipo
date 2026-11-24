package com.lsec.translator.ui.screens.settings

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Pantalla de configuración de la aplicación.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    viewModel: SettingsViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(uiState.message) {
        uiState.message?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Configuración") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Volver"
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Sección de Voz
            SettingsSection(title = "Configuración de Voz") {
                // Habilitar/deshabilitar TTS
                SettingsSwitch(
                    icon = Icons.Default.VolumeUp,
                    title = "Síntesis de voz",
                    subtitle = "Habilitar traducción a voz",
                    checked = uiState.isTTSEnabled,
                    onCheckedChange = { viewModel.setTTSEnabled(it) }
                )

                // Velocidad del habla
                SettingsSlider(
                    icon = Icons.Default.Speed,
                    title = "Velocidad del habla",
                    value = uiState.speechRate,
                    valueRange = 0.5f..2f,
                    onValueChange = { viewModel.setSpeechRate(it) },
                    valueLabel = when {
                        uiState.speechRate < 0.8f -> "Lento"
                        uiState.speechRate > 1.3f -> "Rápido"
                        else -> "Normal"
                    }
                )

                // Tono de voz
                SettingsSlider(
                    icon = Icons.Default.Tune,
                    title = "Tono de voz",
                    value = uiState.pitch,
                    valueRange = 0.5f..2f,
                    onValueChange = { viewModel.setPitch(it) },
                    valueLabel = when {
                        uiState.pitch < 0.8f -> "Grave"
                        uiState.pitch > 1.3f -> "Agudo"
                        else -> "Normal"
                    }
                )

                // Preferir ecuatorianismos
                SettingsSwitch(
                    icon = Icons.Default.Flag,
                    title = "Usar ecuatorianismos",
                    subtitle = "Preferir expresiones locales al hablar",
                    checked = uiState.preferEcuadorianVariants,
                    onCheckedChange = { viewModel.setPreferEcuadorianVariants(it) }
                )
            }

            // Sección de Cámara
            SettingsSection(title = "Configuración de Cámara") {
                // Cámara frontal/trasera
                SettingsSwitch(
                    icon = Icons.Default.CameraFront,
                    title = "Usar cámara frontal",
                    subtitle = "Cambiar entre cámara frontal y trasera",
                    checked = uiState.useFrontCamera,
                    onCheckedChange = { viewModel.setUseFrontCamera(it) }
                )

                // Confianza mínima
                SettingsSlider(
                    icon = Icons.Default.Security,
                    title = "Confianza mínima",
                    value = uiState.minConfidence,
                    valueRange = 0.5f..0.95f,
                    onValueChange = { viewModel.setMinConfidence(it) },
                    valueLabel = "${(uiState.minConfidence * 100).toInt()}%"
                )

                // Modo ahorro de batería
                SettingsSwitch(
                    icon = Icons.Default.BatteryChargingFull,
                    title = "Modo ahorro de batería",
                    subtitle = "Reduce la frecuencia de procesamiento",
                    checked = uiState.batterySaverMode,
                    onCheckedChange = { viewModel.setBatterySaverMode(it) }
                )
            }

            // Sección de Apariencia
            SettingsSection(title = "Apariencia") {
                SettingsSwitch(
                    icon = Icons.Default.DarkMode,
                    title = "Modo oscuro",
                    subtitle = "Usar tema oscuro",
                    checked = uiState.darkMode,
                    onCheckedChange = { viewModel.setDarkMode(it) }
                )
            }

            // Sección de Overlay Visual (landmarks)
            SettingsSection(title = "Overlay de landmarks") {
                SettingsSwitch(
                    icon = Icons.Default.Gesture,
                    title = "Mostrar landmarks (estático)",
                    subtitle = "Puntos y conexiones de manos sobre la cámara en modo letras",
                    checked = uiState.showLandmarksStatic,
                    onCheckedChange = { viewModel.setShowLandmarksStatic(it) }
                )

                SettingsSwitch(
                    icon = Icons.Default.Face,
                    title = "Mostrar landmarks (dinámico)",
                    subtitle = "Manos + cara sobre la cámara en modo palabras (recomendado)",
                    checked = uiState.showLandmarksDynamic,
                    onCheckedChange = { viewModel.setShowLandmarksDynamic(it) }
                )

                SettingsItem(
                    icon = Icons.Default.Info,
                    title = "Acerca del overlay",
                    subtitle = "Muestra los 21 puntos de cada mano y hasta 478 puntos faciales " +
                        "para depurar y mejorar la detección de señas dinámicas. " +
                        "Consume batería adicional."
                )
            }

            // Sección de Servidor
            SettingsSection(title = "Servidor Backend") {
                ServerUrlField(
                    currentUrl = uiState.serverUrl,
                    onSave = { viewModel.setServerUrl(it) }
                )
                Spacer(modifier = Modifier.height(8.dp))
                ApiKeyField(
                    currentKey = uiState.apiKey,
                    onSave = { viewModel.setApiKey(it) }
                )
            }

            // Sección de Datos
            SettingsSection(title = "Datos y Almacenamiento") {
                SettingsItem(
                    icon = Icons.Default.Storage,
                    title = "Diccionario local",
                    subtitle = "${uiState.signCount} señas guardadas"
                )

                SettingsButton(
                    icon = Icons.Default.CloudDownload,
                    title = "Actualizar diccionario",
                    subtitle = "Descargar nuevas señas LSEC",
                    onClick = { viewModel.updateDictionary() }
                )

                SettingsButton(
                    icon = Icons.Default.Delete,
                    title = "Limpiar caché",
                    subtitle = "Liberar espacio de almacenamiento",
                    onClick = { viewModel.clearCache() },
                    isDanger = true
                )
            }

            // Sección de Información
            SettingsSection(title = "Acerca de") {
                SettingsItem(
                    icon = Icons.Default.Info,
                    title = "Versión",
                    subtitle = "1.0.0"
                )

                SettingsItem(
                    icon = Icons.Default.Code,
                    title = "Desarrollado para",
                    subtitle = "Comunidad sorda ecuatoriana"
                )

                SettingsButton(
                    icon = Icons.Default.Help,
                    title = "Ayuda y soporte",
                    subtitle = "Guías de uso y contacto",
                    onClick = { viewModel.openHelp() }
                )
            }
        }
    }
}

@Composable
private fun ServerUrlField(
    currentUrl: String,
    onSave: (String) -> Unit
) {
    var urlText by remember(currentUrl) { mutableStateOf(currentUrl) }
    val keyboardController = LocalSoftwareKeyboardController.current

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 4.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Cloud,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text("URL del servidor", style = MaterialTheme.typography.bodyLarge)
                Text(
                    "Dispositivo físico: http://<IP-de-tu-PC>:8000/",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        OutlinedTextField(
            value = urlText,
            onValueChange = { urlText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("URL del backend") },
            placeholder = { Text("http://192.168.1.x:8000/") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(
                keyboardType = KeyboardType.Uri,
                imeAction = ImeAction.Done
            ),
            keyboardActions = KeyboardActions(
                onDone = {
                    keyboardController?.hide()
                    onSave(urlText)
                }
            ),
            trailingIcon = {
                IconButton(onClick = {
                    keyboardController?.hide()
                    onSave(urlText)
                }) {
                    Icon(
                        imageVector = Icons.Default.Save,
                        contentDescription = "Guardar URL",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
        )

        Text(
            text = "Reinicia la app después de cambiar la URL para que tome efecto.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp, start = 40.dp)
        )
    }
}

@Composable
private fun ApiKeyField(
    currentKey: String,
    onSave: (String) -> Unit
) {
    var keyText by remember(currentKey) { mutableStateOf(currentKey) }
    var keyVisible by remember { mutableStateOf(false) }
    val keyboardController = LocalSoftwareKeyboardController.current

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 4.dp)
        ) {
            Icon(
                imageVector = Icons.Default.VpnKey,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text("Clave API (X-API-Key)", style = MaterialTheme.typography.bodyLarge)
                Text(
                    "Requerida cuando el servidor tiene autenticación activada",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        OutlinedTextField(
            value = keyText,
            onValueChange = { keyText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("API Key") },
            placeholder = { Text("Dejar vacío si no hay autenticación") },
            singleLine = true,
            visualTransformation = if (keyVisible) VisualTransformation.None
                                   else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(
                keyboardType = KeyboardType.Password,
                imeAction = ImeAction.Done
            ),
            keyboardActions = KeyboardActions(
                onDone = {
                    keyboardController?.hide()
                    onSave(keyText)
                }
            ),
            trailingIcon = {
                Row {
                    IconButton(onClick = { keyVisible = !keyVisible }) {
                        Icon(
                            imageVector = if (keyVisible) Icons.Default.VisibilityOff
                                          else Icons.Default.Visibility,
                            contentDescription = if (keyVisible) "Ocultar clave" else "Mostrar clave"
                        )
                    }
                    IconButton(onClick = {
                        keyboardController?.hide()
                        onSave(keyText)
                    }) {
                        Icon(
                            imageVector = Icons.Default.Save,
                            contentDescription = "Guardar clave",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        )

        Text(
            text = "Obtén la clave del archivo .env en el servidor.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp, start = 40.dp)
        )
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(12.dp))
            content()
        }
    }
}

@Composable
private fun SettingsSwitch(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.bodyLarge)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
    }
}

@Composable
private fun SettingsSlider(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    onValueChange: (Float) -> Unit,
    valueLabel: String
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(text = title, style = MaterialTheme.typography.bodyLarge)
            Spacer(modifier = Modifier.weight(1f))
            Text(
                text = valueLabel,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            modifier = Modifier.padding(start = 40.dp)
        )
    }
}

@Composable
private fun SettingsItem(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.bodyLarge)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun SettingsButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit,
    isDanger: Boolean = false
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (isDanger)
                MaterialTheme.colorScheme.error
            else
                MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge,
                color = if (isDanger) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        IconButton(onClick = onClick) {
            Icon(
                imageVector = Icons.Default.ChevronRight,
                contentDescription = "Ir",
                tint = if (isDanger)
                    MaterialTheme.colorScheme.error
                else
                    MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
