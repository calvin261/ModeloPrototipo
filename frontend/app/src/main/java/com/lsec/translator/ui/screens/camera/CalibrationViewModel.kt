package com.lsec.translator.ui.screens.camera

import android.media.Image
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lsec.translator.domain.usecase.DetectHandsUseCase
import com.lsec.translator.ml.SignClassifier
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import javax.inject.Inject

/**
 * Estado de la UI de la pantalla de Calibración (30s).
 *
 * La calibración recorre las 12 señas del vocabulario dinámico de forma
 * secuencial; durante la ventana de tiempo asignada a cada seña (~2.5s)
 * se capturan hasta [maxSamplesPerSign] embeddings de 128-d usando el
 * `SignTransformerClassifier.extractEmbedding()`. Al final, los embeddings
 * se promedian en `CentroidClassifier.setCentroids()` para personalizar
 * el reconocedor.
 */
data class CalibrationUiState(
    val isCapturing: Boolean = false,
    val isCompleted: Boolean = false,
    val wasCancelled: Boolean = false,
    val totalDurationMs: Long = 30_000L,
    val remainingMs: Long = 30_000L,
    val signs: List<String> = CalibrationViewModel.SIGNS,
    val currentSignIndex: Int = 0,
    val currentSign: String = CalibrationViewModel.SIGNS.first(),
    val samplesForCurrentSign: Int = 0,
    val samplesPerSign: Map<String, Int> = emptyMap(),
    val totalCaptured: Int = 0,
    val error: String? = null,
) {
    val progressFraction: Float
        get() = if (totalDurationMs == 0L) 0f
        else 1f - (remainingMs.toFloat() / totalDurationMs.toFloat()).coerceIn(0f, 1f)
}

/**
 * ViewModel para [CalibrationScreen]. Implementa el flujo completo de
 * captura de centroides a 30s:
 *
 *   1. Usuario pulsa "Iniciar".
 *   2. Bucle por las 12 señas; por cada una se abre una ventana de
 *      captura de ~2.5s donde se alimenta la state machine del
 *      SignTransformer y se extrae el embedding CLS de cada secuencia
 *      completa (max 5 embeddings por seña).
 *   3. Al cerrar la ventana se guarda la lista de embeddings de esa
 *      seña y se avanza a la siguiente.
 *   4. Al agotar los 30s se llama a CentroidClassifier.setCentroids().
 *
 * La cámara (CameraPreviewContent) llama a [processImage] en cada frame,
 * igual que en CameraScreen, para que la state machine del facade tenga
 * datos con los que detectar el fin de cada seña.
 */
@HiltViewModel
class CalibrationViewModel @Inject constructor(
    private val detectHandsUseCase: DetectHandsUseCase,
    private val signClassifier: SignClassifier,
) : ViewModel() {

    private val _uiState = MutableStateFlow(CalibrationUiState())
    val uiState: StateFlow<CalibrationUiState> = _uiState.asStateFlow()

    private val perSignEmbeddings = mutableMapOf<String, MutableList<FloatArray>>()
    private var captureJob: Job? = null

    /**
     * Inicia la captura. Idempotente: si ya está corriendo, no hace nada.
     */
    fun start() {
        if (_uiState.value.isCapturing) return
        perSignEmbeddings.clear()
        signClassifier.clearDynamicBuffer()
        _uiState.update {
            CalibrationUiState(
                isCapturing = true,
                isCompleted = false,
                wasCancelled = false,
                signs = it.signs,
                currentSignIndex = 0,
                currentSign = it.signs.first(),
                totalDurationMs = it.totalDurationMs,
                remainingMs = it.totalDurationMs,
                samplesPerSign = emptyMap(),
                totalCaptured = 0,
                samplesForCurrentSign = 0,
            )
        }
        captureJob = viewModelScope.launch { runCaptureLoop() }
    }

    /**
     * Cancela la captura en curso (botón "Detener").
     */
    fun cancel() {
        captureJob?.cancel()
        captureJob = null
        signClassifier.clearDynamicBuffer()
        _uiState.update {
            it.copy(
                isCapturing = false,
                wasCancelled = !it.isCompleted,
                error = null,
            )
        }
    }

    /**
     * Resetea el estado (vuelve a la pantalla inicial pre-Iniciar).
     */
    fun reset() {
        captureJob?.cancel()
        captureJob = null
        perSignEmbeddings.clear()
        signClassifier.clearDynamicBuffer()
        _uiState.value = CalibrationUiState()
    }

    /**
     * Llamado por la cámara en cada frame de preview. Si la captura está
     * activa, alimenta la state machine del SignTransformer para que
     * `lastClassifiedSequence()` se mantenga actualizada y
     * `extractEmbedding(...)` devuelva vectores válidos al finalizar la
     * seña.
     *
     * Importante: se ejecuta dentro del callback del `ImageAnalysis`
     * analyzer, que cierra el `ImageProxy` (y por tanto el `Image`) justo
     * después de que este método retorna. Por eso NO podemos lanzar una
     * coroutine con `viewModelScope.launch` y consumir el `Image` dentro:
     * llegaría cerrado. Usamos `runBlocking` (mismo patrón que
     * `CameraViewModel.processFrame`) para resolver `Image -> landmarks`
     * en el mismo thread antes de que el analyzer cierre el buffer.
     */
    fun processImage(image: Image?) {
        if (!_uiState.value.isCapturing) return
        try {
            val landmarks = runBlocking { detectHandsUseCase(image) }
            // Resultado se descarta aquí; sólo nos interesa poblar el
            // buffer interno del transformer para que extractEmbedding
            // tenga algo al cierre de la secuencia.
            signClassifier.processDynamicFrame(landmarks)
        } catch (e: Throwable) {
            Log.e(TAG, "processImage failed", e)
            _uiState.update { it.copy(error = e.message) }
        }
    }

    /**
     * Bucle principal: itera por las 12 señas y, dentro de la ventana
     * asignada a cada una, intenta capturar hasta [maxSamplesPerSign]
     * embeddings.
     */
    private suspend fun runCaptureLoop() {
        val totalMs = _uiState.value.totalDurationMs
        val startedAt = System.currentTimeMillis()
        val windowMs = totalMs / SIGNS.size

        // Loop principal de captura
        for ((idx, sign) in SIGNS.withIndex()) {
            if (!isActive(startedAt, totalMs)) break

            _uiState.update {
                it.copy(
                    currentSignIndex = idx,
                    currentSign = sign,
                    samplesForCurrentSign = 0,
                )
            }
            perSignEmbeddings[sign] = mutableListOf()
            val windowEnd = System.currentTimeMillis() + windowMs

            while (isActive(startedAt, totalMs) && System.currentTimeMillis() < windowEnd) {
                delay(POLL_INTERVAL_MS)
                val sample = tryCaptureSample()
                if (sample != null) {
                    perSignEmbeddings[sign]!!.add(sample)
                    _uiState.update { state ->
                        val newMap = state.samplesPerSign.toMutableMap()
                        newMap[sign] = (newMap[sign] ?: 0) + 1
                        state.copy(
                            samplesForCurrentSign = perSignEmbeddings[sign]!!.size,
                            samplesPerSign = newMap,
                            totalCaptured = state.totalCaptured + 1,
                        )
                    }
                    if (perSignEmbeddings[sign]!!.size >= MAX_SAMPLES_PER_SIGN) break
                }
            }

            // Pequeña pausa entre señas para que la state machine se asiente
            // (cooldown implícito del SignTransformer).
            delay(SIGN_TRANSITION_MS)
        }

        // Tick de countdown hasta agotar los 30s (para que el progress bar
        // se vea fluido aunque no haya más señas que procesar).
        while (isActive(startedAt, totalMs)) {
            val elapsed = System.currentTimeMillis() - startedAt
            _uiState.update { it.copy(remainingMs = (totalMs - elapsed).coerceAtLeast(0L)) }
            delay(COUNTDOWN_TICK_MS)
        }

        // Aplica los centroides si capturamos al menos 1 muestra por
        // al menos 1 seña. Si no, marca como cancelado.
        val totalSamples = perSignEmbeddings.values.sumOf { it.size }
        if (totalSamples == 0) {
            _uiState.update { it.copy(isCapturing = false, wasCancelled = true) }
            return
        }

        try {
            signClassifier.centroids()?.setCentroids(perSignEmbeddings.toMap())
            _uiState.update { it.copy(isCapturing = false, isCompleted = true, remainingMs = 0L) }
        } catch (e: Throwable) {
            Log.e(TAG, "setCentroids failed", e)
            _uiState.update { it.copy(isCapturing = false, error = e.message) }
        }
    }

    private fun isActive(startedAt: Long, totalMs: Long): Boolean {
        // Important: do NOT consult captureJob?.isActive here. When this
        // function is called from inside runCaptureLoop() (which itself is
        // running on Main.immediate inside the very same launch {...}
        // that creates captureJob), the field captureJob has not been
        // assigned yet — Main.immediate dispatches the lambda before the
        // outer assignment statement completes — and we'd read null,
        // causing the loop to exit immediately. Trust the wall clock.
        return System.currentTimeMillis() - startedAt < totalMs
    }

    /**
     * Intenta extraer el embedding CLS de la última secuencia completa
     * clasificada por el SignTransformer. Devuelve null si el modelo no
     * está disponible, si no hay secuencia reciente, o si se intentó dos
     * veces muy rápido (evita capturar el mismo gesto 5 veces en 50ms).
     */
    private fun tryCaptureSample(): FloatArray? {
        val tx = signClassifier.transformer() ?: return null
        val seq = tx.lastClassifiedSequence() ?: return null
        return try {
            tx.extractEmbedding(seq)
        } catch (e: Throwable) {
            Log.w(TAG, "extractEmbedding failed", e)
            null
        }
    }

    override fun onCleared() {
        super.onCleared()
        captureJob?.cancel()
        signClassifier.clearDynamicBuffer()
    }

    companion object {
        private const val TAG = "CalibrationViewModel"

        /** Duración total del flujo de calibración. */
        const val TOTAL_DURATION_MS: Long = 30_000L

        /** Máximo de muestras (embeddings) a capturar por seña. */
        const val MAX_SAMPLES_PER_SIGN: Int = 5

        /** Signos del vocabulario dinámico que se calibran. */
        val SIGNS: List<String> = listOf(
            "hola", "gracias", "por_favor", "si", "no",
            "ayuda", "bien", "mal", "perdon", "adios",
            "como_estas", "bano"
        )

        /** Cada cuántos ms revisamos si hay un embedding nuevo. */
        private const val POLL_INTERVAL_MS: Long = 120L

        /** Cada cuántos ms refrescamos el countdown en pantalla. */
        private const val COUNTDOWN_TICK_MS: Long = 100L

        /** Pausa entre señas consecutivas (ms). */
        private const val SIGN_TRANSITION_MS: Long = 150L
    }
}