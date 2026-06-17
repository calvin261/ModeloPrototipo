package com.lsec.translator.test

import android.content.Context
import android.os.Environment
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileWriter
import java.util.Locale

/**
 * F7 user-validation helper.
 *
 * Logs every prompted/predicted sign with confidence and end-to-end latency
 * to a CSV in the app's external cache directory. The CSV is the canonical
 * artifact consumed by `backend/scripts/analyze_f7_results.py`.
 *
 * CSV schema (one row per trial, header is always written first):
 *
 *     tester_id,round,rep,index,prompted_sign,predicted_sign,confidence,
 *     latency_ms,timestamp_ms,seed,dominance,note
 *
 * Latency is measured from the moment the "AHORA" prompt is shown
 * (gesture start) to the moment the predicted sign is rendered on
 * screen (end-to-end, including MediaPipe + TFLite + post-processing).
 * Timestamps come from `System.currentTimeMillis()`.
 */
class SignTestLogger(
    private val context: Context,
    private val testerId: String,
    private val dominance: String,
) {
    private val tag = "SignTestLogger"
    private val csvLock = Any()

    fun outputFile(round: Int): File {
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        if (!dir.exists()) dir.mkdirs()
        return File(dir, "${testerId}_round${round}.csv")
    }

    fun writeHeader(round: Int, seed: Int) {
        synchronized(csvLock) {
            val f = outputFile(round)
            if (!f.exists() || f.length() == 0L) {
                FileWriter(f, false).use { w ->
                    w.append("tester_id,round,rep,index,prompted_sign,")
                        .append("predicted_sign,confidence,latency_ms,")
                        .append("timestamp_ms,seed,dominance,note\n")
                    Log.i(tag, "Header written to ${f.absolutePath} (seed=$seed)")
                }
            }
        }
    }

    /**
     * Append one trial result. Safe to call from the main thread;
     * the actual write is fast enough (~tens of µs) not to need
     * a dedicated IO dispatcher, but we still serialize to keep
     * row order.
     */
    fun append(
        round: Int,
        rep: Int,
        index: Int,
        prompted: String,
        predicted: String,
        confidence: Float,
        latencyMs: Long,
        timestampMs: Long,
        seed: Int,
        note: String = "",
    ) {
        synchronized(csvLock) {
            FileWriter(outputFile(round), true).use { w ->
                w.append(testerId).append(',')
                    .append(round.toString()).append(',')
                    .append(rep.toString()).append(',')
                    .append(index.toString()).append(',')
                    .append(prompted).append(',')
                    .append(predicted).append(',')
                    .append(String.format(Locale.US, "%.4f", confidence)).append(',')
                    .append(latencyMs.toString()).append(',')
                    .append(timestampMs.toString()).append(',')
                    .append(seed.toString()).append(',')
                    .append(dominance).append(',')
                    .append(if (note.isEmpty()) "-" else note)
                    .append('\n')
            }
        }
    }

    fun deleteAllRounds() {
        synchronized(csvLock) {
            for (r in listOf(1, 2)) {
                val f = outputFile(r)
                if (f.exists()) f.delete()
            }
        }
    }
}

/**
 * Pure-Kotlin, deterministic Fisher-Yates shuffle with a fixed seed.
 * Used to build the prompted-sign order for a given tester/round so
 * the order is reproducible and differs between round 1 and round 2.
 */
internal fun seededShuffle(items: List<String>, seed: Int): List<String> {
    val rng = java.util.Random(seed.toLong())
    val out = items.toMutableList()
    for (i in out.size - 1 downTo 1) {
        val j = rng.nextInt(i + 1)
        val tmp = out[i]
        out[i] = out[j]
        out[j] = tmp
    }
    return out
}

/**
 * State machine backing the F7 test session UI. Owned by the
 * `TestSessionScreen` composable; coroutines are launched from
 * `rememberCoroutineScope()`.
 */
sealed interface TestPhase {
    data class Idle(val round: Int) : TestPhase
    data class Prompting(
        val round: Int,
        val index: Int,
        val sign: String,
        val countdownMsLeft: Long,
    ) : TestPhase
    data class AwaitingResult(
        val round: Int,
        val index: Int,
        val sign: String,
        val tStartMs: Long,
    ) : TestPhase
    data class ShowingResult(
        val round: Int,
        val index: Int,
        val sign: String,
        val predicted: String,
        val confidence: Float,
        val latencyMs: Long,
    ) : TestPhase
    data class Done(val round: Int, val nTrials: Int) : TestPhase
}

class TestSessionViewModel(
    private val logger: SignTestLogger,
    private val testerId: String,
    private val dominance: String,
    private val vocabulary: List<String>,
    private val reps: Int = 3,
    private val predictFn: suspend (String) -> Pair<String, Float>,
) {
    private val _phase = MutableStateFlow<TestPhase>(TestPhase.Idle(1))
    val phase: StateFlow<TestPhase> = _phase.asStateFlow()

    private val orders: Map<Int, List<String>> = mapOf(
        1 to seededShuffle(buildList { repeat(reps) { addAll(vocabulary) } }, testerId.hashCode()),
        2 to seededShuffle(
            buildList { repeat(reps) { addAll(vocabulary) } },
            testerId.hashCode() xor 0xCAFE,
        ),
    )

    fun startRound(round: Int) {
        logger.writeHeader(round, seed = orders[round]!!.hashCode())
        _phase.value = TestPhase.Prompting(
            round = round,
            index = 0,
            sign = orders[round]!![0],
            countdownMsLeft = 1500L,
        )
    }

    /**
     * Drive the round forward. The caller launches this in a
     * coroutine; it yields the UI between phases with `delay()`.
     * Never blocks the caller.
     */
    suspend fun runRound(round: Int) {
        val order = orders[round] ?: return
        for ((idx, sign) in order.withIndex()) {
            if (!kotlinx.coroutines.currentCoroutineContext().isActive) return
            _phase.value = TestPhase.Prompting(
                round, idx, sign, countdownMsLeft = 1500L,
            )
            delay(1500L)
            val tStart = System.currentTimeMillis()
            _phase.value = TestPhase.AwaitingResult(round, idx, sign, tStart)
            val (predicted, conf) = withContext(Dispatchers.Default) { predictFn(sign) }
            val tEnd = System.currentTimeMillis()
            val latency = tEnd - tStart
            logger.append(
                round = round,
                rep = (idx % reps) + 1,
                index = idx,
                prompted = sign,
                predicted = predicted,
                confidence = conf,
                latencyMs = latency,
                timestampMs = tStart,
                seed = order.hashCode(),
            )
            _phase.value = TestPhase.ShowingResult(
                round, idx, sign, predicted, conf, latency,
            )
            delay(1200L)
        }
        _phase.value = TestPhase.Done(round, order.size)
    }
}

/**
 * Compose screen that drives one F7 test session. The caller
 * wires `predictFn` to the real ML pipeline
 * (`DynamicSignClassifier` + `CentroidClassifier` for round 2).
 */
@Composable
fun TestSessionScreen(
    testerId: String,
    dominance: String,
    vocabulary: List<String>,
    predictFn: suspend (String) -> Pair<String, Float>,
    onSessionFinished: (round: Int) -> Unit = {},
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val logger = remember { SignTestLogger(context, testerId, dominance) }
    val vm = remember {
        TestSessionViewModel(
            logger = logger,
            testerId = testerId,
            dominance = dominance,
            vocabulary = vocabulary,
            predictFn = predictFn,
        )
    }
    val phase by vm.phase.collectAsState()

    Box(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "F7 Test — $testerId",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Vocabulario: ${vocabulary.size} señas, ${vocabulary.size * 3} trials por ronda",
                style = MaterialTheme.typography.bodySmall,
            )

            when (val p = phase) {
                is TestPhase.Idle -> {
                    Button(onClick = { vm.startRound(p.round); scope.launch { vm.runRound(p.round) } }) {
                        Text("Iniciar Ronda ${p.round}")
                    }
                }
                is TestPhase.Prompting -> {
                    Text("Prepárate…", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = p.sign,
                        style = MaterialTheme.typography.displayMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text("en ${(p.countdownMsLeft / 100).coerceAtLeast(0) * 100} ms")
                }
                is TestPhase.AwaitingResult -> {
                    Text("AHORA", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
                    Text(p.sign, style = MaterialTheme.typography.displaySmall)
                    CircularProgressIndicator()
                }
                is TestPhase.ShowingResult -> {
                    Text("Predicho:", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = p.predicted,
                        style = MaterialTheme.typography.displayMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Card { Text("confianza ${"%.2f".format(p.confidence)}") }
                    Text("latencia ${p.latencyMs} ms", style = MaterialTheme.typography.bodySmall)
                }
                is TestPhase.Done -> {
                    Text("Ronda ${p.round} completa: ${p.nTrials} trials")
                    Button(onClick = { onSessionFinished(p.round) }) {
                        Text("Volver")
                    }
                }
            }
        }
    }
}

/**
 * Helper: collect every CSV written so far. Used by the investigator
 * to pull the data off the device with `adb pull`.
 */
fun listTestCsvs(context: Context): List<File> {
    val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
    if (!dir.exists()) return emptyList()
    return dir.listFiles { f -> f.name.endsWith(".csv") }?.toList() ?: emptyList()
}
