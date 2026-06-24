package com.lsec.translator.ui.screens.capture

import android.Manifest
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.lsec.translator.ui.common.KeepScreenOn
import com.google.accompanist.permissions.shouldShowRationale
import java.util.concurrent.Executors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DataCaptureScreen(
    onNavigateBack: () -> Unit,
    viewModel: DataCaptureViewModel = hiltViewModel()
) {
    KeepScreenOn()
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    DataCaptureContent(
        uiState = uiState,
        onNavigateBack = onNavigateBack,
        viewModel = viewModel
    )
}

@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun DataCaptureContent(
    uiState: DataCaptureUiState,
    onNavigateBack: () -> Unit,
    viewModel: DataCaptureViewModel
) {
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Captura de Datos") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Volver")
                    }
                },
                actions = {
                    // Contadores separados: estáticas + secuencias
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        if (uiState.totalStatic > 0) {
                            Badge(containerColor = MaterialTheme.colorScheme.primary) {
                                Text("E:${uiState.totalStatic}")
                            }
                        }
                        if (uiState.totalSequences > 0) {
                            Badge(containerColor = MaterialTheme.colorScheme.tertiary) {
                                Text("S:${uiState.totalSequences}")
                            }
                        }
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Toggle modo captura
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                FilterChip(
                    selected = uiState.captureMode == CaptureMode.STATIC,
                    onClick = { viewModel.setCaptureMode(CaptureMode.STATIC) },
                    label = { Text("Estatico") },
                    leadingIcon = {
                        Icon(Icons.Default.CameraAlt, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                )
                Spacer(modifier = Modifier.width(8.dp))
                FilterChip(
                    selected = uiState.captureMode == CaptureMode.SEQUENCE,
                    onClick = { viewModel.setCaptureMode(CaptureMode.SEQUENCE) },
                    label = { Text("Secuencia") },
                    leadingIcon = {
                        Icon(Icons.Default.Videocam, contentDescription = null, modifier = Modifier.size(18.dp))
                    }
                )
            }

            // Selector de label
            Text(
                text = "Sena seleccionada: ${uiState.selectedLabel}",
                style = MaterialTheme.typography.titleMedium
            )

            val labels = if (uiState.captureMode == CaptureMode.STATIC)
                viewModel.staticLabels else viewModel.dynamicLabels
            val perLabelCounts = if (uiState.captureMode == CaptureMode.STATIC)
                uiState.capturedStatic else uiState.capturedSequences

            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                items(labels) { label ->
                    FilterChip(
                        selected = uiState.selectedLabel == label,
                        onClick = { viewModel.selectLabel(label) },
                        label = {
                            val count = perLabelCounts[label] ?: 0
                            val suffix = if (uiState.captureMode == CaptureMode.SEQUENCE && count > 0) " ($count ${if (count == 1) "vídeo" else "vídeos"})" else if (count > 0) " ($count)" else ""
                            Text("$label$suffix")
                        }
                    )
                }
            }

            // Area de preview de camara (placeholder)
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                when {
                    cameraPermissionState.status.isGranted -> {
                        DataCaptureCameraPreview(
                            onFrameAnalyzed = { viewModel.processFrame(it) },
                            handsDetected = uiState.handsDetected,
                            isRecording = uiState.isRecording,
                            recordedFrames = uiState.recordedFrames,
                            recordingTarget = uiState.recordingTargetFrames,
                            recordingProgress = uiState.recordingProgress,
                            canStopRecording = uiState.canStopRecording,
                            useFrontCamera = uiState.useFrontCamera,
                            onToggleCamera = { viewModel.toggleCamera() }
                        )
                    }

                    cameraPermissionState.status.shouldShowRationale -> {
                        PermissionStateContent(
                            title = "Permiso de cámara requerido",
                            description = "Necesitamos acceso a la cámara para capturar landmarks.",
                            buttonText = "Reintentar permiso",
                            onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                        )
                    }

                    else -> {
                        PermissionStateContent(
                            title = "Activa la cámara",
                            description = "Concede el permiso para abrir la vista de captura.",
                            buttonText = "Conceder permiso",
                            onRequestPermission = { cameraPermissionState.launchPermissionRequest() }
                        )
                    }
                }
            }

            // Botones de accion
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (uiState.captureMode == CaptureMode.STATIC) {
                    Button(
                        onClick = { viewModel.captureStaticSample() },
                        modifier = Modifier.weight(1f),
                        enabled = uiState.handsDetected && cameraPermissionState.status.isGranted
                    ) {
                        Icon(Icons.Default.Camera, contentDescription = null)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Capturar")
                    }
                } else {
                    val seqLabel = when {
                        uiState.isRecording && uiState.recordedFrames >= uiState.recordingTargetFrames -> "¡Listo! Detén"
                        uiState.isRecording -> "Detener (${uiState.recordedFrames})"
                        else -> "Grabar '${uiState.selectedLabel}'"
                    }
                    Button(
                        onClick = {
                            if (uiState.isRecording) viewModel.stopRecording()
                            else viewModel.startRecording()
                        },
                        modifier = Modifier.weight(1f),
                        colors = if (uiState.isRecording) ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.error
                        ) else ButtonDefaults.buttonColors(),
                        enabled = !uiState.isRecording || uiState.canStopRecording
                    ) {
                        Icon(
                            if (uiState.isRecording) Icons.Default.Stop else Icons.Default.FiberManualRecord,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(seqLabel)
                    }
                }

                // Boton de subir (cuenta ambas modalidades)
                Button(
                    onClick = { viewModel.uploadCapturedData() },
                    modifier = Modifier.weight(1f),
                    enabled = uiState.hasContent && !uiState.isUploading
                ) {
                    if (uiState.isUploading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                    } else {
                        Icon(Icons.Default.CloudUpload, contentDescription = null)
                    }
                    Spacer(modifier = Modifier.width(4.dp))
                    val uploadLabel = when {
                        uiState.totalStatic > 0 && uiState.totalSequences > 0 ->
                            "Subir (E:${uiState.totalStatic} S:${uiState.totalSequences})"
                        uiState.totalStatic > 0 -> "Subir (${uiState.totalStatic})"
                        uiState.totalSequences > 0 -> "Subir (${uiState.totalSequences} vídeos)"
                        else -> "Subir"
                    }
                    Text(uploadLabel)
                }
            }

            // Mensaje de resultado de upload
            uiState.uploadMessage?.let { message ->
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (message.startsWith("Error"))
                            MaterialTheme.colorScheme.errorContainer
                        else MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            message,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f)
                        )
                        IconButton(onClick = { viewModel.clearUploadMessage() }) {
                            Icon(Icons.Default.Close, contentDescription = "Cerrar", modifier = Modifier.size(16.dp))
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DataCaptureCameraPreview(
    onFrameAnalyzed: (android.media.Image?) -> Unit,
    handsDetected: Boolean,
    isRecording: Boolean,
    recordedFrames: Int,
    recordingTarget: Int = 45,
    recordingProgress: Float = 0f,
    canStopRecording: Boolean = false,
    useFrontCamera: Boolean = true,
    onToggleCamera: () -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    LaunchedEffect(useFrontCamera) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            onFrameAnalyzed(imageProxy.image)
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            val cameraSelector = if (useFrontCamera)
                CameraSelector.DEFAULT_FRONT_CAMERA
            else
                CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
            } catch (_: Exception) {
            }
        }, ContextCompat.getMainExecutor(context))
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        // Botón cambio de cámara (esquina superior derecha)
        IconButton(
            onClick = onToggleCamera,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(8.dp)
        ) {
            Icon(
                imageVector = if (useFrontCamera) Icons.Default.CameraFront else Icons.Default.CameraRear,
                contentDescription = "Cambiar cámara",
                tint = Color.White
            )
        }

        Card(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.6f))
        ) {
            Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
                Text(
                    if (handsDetected) "Mano detectada" else "Posiciona tu mano frente a la cámara",
                    color = Color.White,
                    style = MaterialTheme.typography.bodySmall
                )
                if (isRecording) {
                    Text(
                        "Grabando: $recordedFrames/$recordingTarget frames" +
                            if (canStopRecording) " · puedes detener" else "",
                        color = if (canStopRecording) Color(0xFFB9F6CA) else MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                    LinearProgressIndicator(
                        progress = { recordingProgress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 4.dp)
                            .height(3.dp),
                        color = if (canStopRecording) Color(0xFFB9F6CA) else MaterialTheme.colorScheme.error,
                        trackColor = Color.White.copy(alpha = 0.3f)
                    )
                }
            }
        }
    }
}

@Composable
private fun PermissionStateContent(
    title: String,
    description: String,
    buttonText: String,
    onRequestPermission: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Default.CameraAlt,
            contentDescription = null,
            modifier = Modifier.size(56.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(title, style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = description,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = onRequestPermission) {
            Text(buttonText)
        }
    }
}
