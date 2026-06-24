package com.lsec.translator.ui.screens.camera

import android.Manifest
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
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

/**
 * Pantalla de Calibración de 30 segundos.
 *
 * Flujo implementado:
 *   1. Usuario concede permiso de cámara.
 *   2. Pulsa "Iniciar". El [CalibrationViewModel] arranca un bucle de 30s
 *      que recorre las 12 señas del vocabulario dinámico.
 *   3. Mientras corre, la camera preview alimenta la state machine del
 *      SignTransformer (vía [CalibrationViewModel.processImage]). Al
 *      cierre de cada secuencia completa se extrae el embedding CLS de
 *      128-d y se acumula en `perSignEmbeddings[currentSign]`.
 *   4. Tras agotar los 30s se llama a
 *      `CentroidClassifier.setCentroids(perSignEmbeddings)`, que promedia
 *      los embeddings por seña y los persiste en
 *      `filesDir/calibration/default.bin`. A partir de ese momento
 *      `SignClassifier.processDynamicFrame` re-decide el label comparando
 *      contra los centroides por cosine similarity.
 *
 * Ver [CalibrationViewModel] para los detalles del bucle de captura.
 */
@OptIn(ExperimentalPermissionsApi::class, ExperimentalMaterial3Api::class)
@Composable
fun CalibrationScreen(
    onNavigateBack: () -> Unit,
    viewModel: CalibrationViewModel = hiltViewModel()
) {
    KeepScreenOn()
    val cameraPermissionState = rememberPermissionState(Manifest.permission.CAMERA)
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        if (!cameraPermissionState.status.isGranted) {
            cameraPermissionState.launchPermissionRequest()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Calibración (30s)") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Volver")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Black.copy(alpha = 0.7f),
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White,
                )
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .background(Color.Black)
        ) {
            when {
                !cameraPermissionState.status.isGranted ->
                    PermissionGate(
                        shouldShowRationale = cameraPermissionState.status.shouldShowRationale,
                        onRequest = { cameraPermissionState.launchPermissionRequest() }
                    )
                uiState.isCompleted -> CompletedSummary(
                    uiState = uiState,
                    onDone = onNavigateBack,
                    onRepeat = { viewModel.reset() }
                )
                else -> ActiveSession(
                    uiState = uiState,
                    onImage = viewModel::processImage,
                    onStart = viewModel::start,
                    onCancel = viewModel::cancel,
                )
            }
        }
    }
}

// ----------------------------------------------------------------------
// Active session UI
// ----------------------------------------------------------------------

@Composable
private fun ActiveSession(
    uiState: CalibrationUiState,
    onImage: (android.media.Image?) -> Unit,
    onStart: () -> Unit,
    onCancel: () -> Unit,
) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview al fondo.
        CalibrationCameraPreview(onFrame = onImage)

        // Overlay oscuro en la mitad inferior para que se lea el HUD.
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            Spacer(modifier = Modifier.height(120.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Color.Black.copy(alpha = 0.78f),
                        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
                    )
                    .padding(16.dp)
            ) {
                Column {
                    CountdownBar(uiState)
                    Spacer(Modifier.height(12.dp))
                    SignsGrid(uiState)
                    Spacer(Modifier.height(12.dp))
                    ControlButtons(
                        isCapturing = uiState.isCapturing,
                        onStart = onStart,
                        onCancel = onCancel,
                    )
                }
            }
        }

        // Banner "muestre esta seña" flotante cuando está capturando.
        if (uiState.isCapturing) {
            CurrentSignBanner(
                sign = uiState.currentSign,
                index = uiState.currentSignIndex,
                total = uiState.signs.size,
                samples = uiState.samplesForCurrentSign,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 16.dp, start = 16.dp, end = 16.dp)
            )
        }

        // Mensaje de cancelación si el usuario paró sin completar.
        if (uiState.wasCancelled && !uiState.isCompleted) {
            CancelledBanner(
                onRetry = onStart,
                onDismiss = { onCancel() /* re-arm */ },
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(24.dp)
            )
        }
    }
}

@Composable
private fun CountdownBar(uiState: CalibrationUiState) {
    val animated by animateFloatAsState(
        targetValue = uiState.progressFraction,
        animationSpec = tween(durationMillis = 120),
        label = "calib_progress",
    )
    val secsLeft = (uiState.remainingMs / 1000L).toInt().coerceAtLeast(0)
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                "Calibración",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.weight(1f),
            )
            Text(
                "${secsLeft}s restantes",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
        }
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { animated },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp)),
            color = when {
                uiState.progressFraction >= 1f -> MaterialTheme.colorScheme.primary
                uiState.progressFraction >= 0.5f -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.secondary
            },
            trackColor = Color.White.copy(alpha = 0.15f),
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "${uiState.totalCaptured} muestras capturadas en total",
            color = Color.White.copy(alpha = 0.7f),
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
private fun SignsGrid(uiState: CalibrationUiState) {
    val columns = 4
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        uiState.signs.chunked(columns).forEach { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                row.forEach { sign ->
                    val samples = uiState.samplesPerSign[sign] ?: 0
                    val isCurrent = uiState.isCapturing && sign == uiState.currentSign
                    SignChip(
                        name = sign,
                        samples = samples,
                        isCurrent = isCurrent,
                        modifier = Modifier.weight(1f)
                    )
                }
                // Rellena la última fila si quedó incompleta.
                repeat(columns - row.size) {
                    Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun SignChip(
    name: String,
    samples: Int,
    isCurrent: Boolean,
    modifier: Modifier = Modifier
) {
    val (bg, border) = when {
        samples > 0 -> MaterialTheme.colorScheme.primary.copy(alpha = 0.85f) to Color.Transparent
        isCurrent -> MaterialTheme.colorScheme.tertiary.copy(alpha = 0.85f) to Color.White
        else -> Color.White.copy(alpha = 0.10f) to Color.White.copy(alpha = 0.3f)
    }
    Surface(
        modifier = modifier,
        color = bg,
        shape = RoundedCornerShape(10.dp),
        border = if (isCurrent) androidx.compose.foundation.BorderStroke(2.dp, border) else null,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                name.replace("_", " "),
                color = Color.White,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
                textAlign = TextAlign.Center,
                fontWeight = if (isCurrent) FontWeight.Bold else FontWeight.Normal,
            )
            Text(
                when {
                    samples >= CalibrationViewModel.MAX_SAMPLES_PER_SIGN -> "✓"
                    samples > 0 -> "$samples/${CalibrationViewModel.MAX_SAMPLES_PER_SIGN}"
                    else -> "—"
                },
                color = Color.White.copy(alpha = if (samples > 0) 1f else 0.6f),
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
private fun ControlButtons(
    isCapturing: Boolean,
    onStart: () -> Unit,
    onCancel: () -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (isCapturing) {
            Button(
                onClick = onCancel,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = Color.White,
                ),
            ) {
                Icon(Icons.Default.Cancel, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Detener")
            }
        } else {
            Button(
                onClick = onStart,
                modifier = Modifier.weight(1f),
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Iniciar")
            }
        }
    }
}

@Composable
private fun CurrentSignBanner(
    sign: String,
    index: Int,
    total: Int,
    samples: Int,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        color = Color.Black.copy(alpha = 0.75f),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "Seña ${index + 1} / $total",
                color = Color.White.copy(alpha = 0.7f),
                style = MaterialTheme.typography.labelMedium,
            )
            Text(
                sign.replace("_", " ").uppercase(),
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
            Text(
                "$samples / ${CalibrationViewModel.MAX_SAMPLES_PER_SIGN} muestras",
                color = MaterialTheme.colorScheme.tertiary,
                style = MaterialTheme.typography.labelMedium,
            )
        }
    }
}

@Composable
private fun CancelledBanner(
    onRetry: () -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier,
        color = Color.Black.copy(alpha = 0.85f),
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Default.Warning,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.error,
                modifier = Modifier.size(40.dp),
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "No se capturaron muestras",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "Asegúrate de que la cámara vea tu mano y vuelve a iniciar.",
                color = Color.White.copy(alpha = 0.8f),
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = onDismiss) { Text("Cerrar") }
                Button(onClick = onRetry) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Reintentar")
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// Completed summary UI
// ----------------------------------------------------------------------

@Composable
private fun CompletedSummary(
    uiState: CalibrationUiState,
    onDone: () -> Unit,
    onRepeat: () -> Unit,
) {
    val totalSigns = uiState.samplesPerSign.size
    val totalSamples = uiState.samplesPerSign.values.sum()
    val maxPerSign = CalibrationViewModel.MAX_SAMPLES_PER_SIGN
    val perfectSigns = uiState.samplesPerSign.values.count { it >= maxPerSign }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            "Calibración completa",
            color = Color.White,
            style = MaterialTheme.typography.headlineMedium,
        )
        Text(
            "Capturadas $totalSamples muestras en 30s. El reconocedor ahora usará cosine similarity contra tus centroides.",
            color = Color.White.copy(alpha = 0.8f),
            style = MaterialTheme.typography.bodyMedium,
        )

        // Resumen de cada seña.
        Surface(
            color = Color.White.copy(alpha = 0.08f),
            shape = RoundedCornerShape(12.dp),
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Check,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "Resumen",
                        color = Color.White,
                        style = MaterialTheme.typography.titleSmall,
                    )
                }
                Spacer(Modifier.height(8.dp))
                Text(
                    "• $totalSigns / ${uiState.signs.size} señas con muestras",
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "• $perfectSigns señas con $maxPerSign muestras (máximo)",
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "• $totalSamples muestras totales (objetivo: ${uiState.signs.size * maxPerSign})",
                    color = Color.White.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // Lista de señas con estado.
        Text(
            "Detalle por seña",
            color = Color.White,
            style = MaterialTheme.typography.titleSmall,
        )
        Surface(
            color = Color.White.copy(alpha = 0.05f),
            shape = RoundedCornerShape(12.dp),
        ) {
            Column(modifier = Modifier.padding(8.dp)) {
                uiState.signs.forEach { sign ->
                    val samples = uiState.samplesPerSign[sign] ?: 0
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 8.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            if (samples > 0) Icons.Default.Check else Icons.Default.RadioButtonUnchecked,
                            contentDescription = null,
                            tint = if (samples > 0) MaterialTheme.colorScheme.primary
                            else Color.White.copy(alpha = 0.4f),
                            modifier = Modifier.size(16.dp),
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            sign.replace("_", " "),
                            color = Color.White,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            "$samples / $maxPerSign",
                            color = Color.White.copy(alpha = 0.7f),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onRepeat, modifier = Modifier.weight(1f)) {
                Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Repetir")
            }
            Button(onClick = onDone, modifier = Modifier.weight(1f)) {
                Text("Listo")
            }
        }
    }
}

// ----------------------------------------------------------------------
// Permission gate
// ----------------------------------------------------------------------

@Composable
private fun PermissionGate(
    shouldShowRationale: Boolean,
    onRequest: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(
            Icons.Default.Warning,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.error,
            modifier = Modifier.size(64.dp),
        )
        Spacer(Modifier.height(16.dp))
        Text(
            "Permiso de Cámara Requerido",
            color = Color.White,
            style = MaterialTheme.typography.titleLarge,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            if (shouldShowRationale)
                "Sin acceso a la cámara no podemos personalizar el reconocedor. Concede el permiso para continuar."
            else
                "Para la calibración de señas necesitamos acceso a tu cámara.",
            color = Color.White.copy(alpha = 0.8f),
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(20.dp))
        Button(onClick = onRequest) { Text(if (shouldShowRationale) "Reintentar" else "Conceder Permiso") }
    }
}

// ----------------------------------------------------------------------
// Camera preview (local copy; CameraScreen's version is private)
// ----------------------------------------------------------------------

@Composable
private fun CalibrationCameraPreview(
    onFrame: (android.media.Image?) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = remember { PreviewView(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            onFrame(imageProxy.image)
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                android.util.Log.e("CalibrationScreen", "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}