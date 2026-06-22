package com.lsec.translator.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Dynamic-sign classifier backed by the trained SignTransformer
 * (ONNX exported from `backend/scripts/export_transformer_onnx.py`).
 *
 * The public API mirrors [DynamicSignClassifier] so the rest of the app
 * (`SignClassifier` facade, `CameraViewModel`) can swap implementations
 * without churn. Internally we:
 *   - Feed the model a (1, T, 21, 3) tensor padded with zeros at the tail
 *   - Build a padding mask `(seq.abs().sum(dim=(2,3)) < 1e-6)` for
 *     state-machine bookkeeping (and for the model, if the export was
 *     made with a `padding_mask` second input)
 *   - Use the CLS-pooled embedding for centroid-based few-shot
 *     personalization (see [CentroidClassifier])
 *
 * Model resolution order:
 *   1. `filesDir/models/sign_transformer.onnx` (downloaded from server)
 *   2. Bundled TFLite in assets (`sign_transformer.tflite`) as fallback
 *
 * Per-frame target latency on Infinix GT30 Pro (Dimensity 8350): ≤50ms.
 * ONNX Runtime 1.17.3 supports opset 18 which is what the export uses.
 */
class SignTransformerClassifier(
    private val context: Context,
    private val vocabulary: Map<String, Int> = emptyMap(),
) {
    private val tag = "SignTransformer"

    private val maxSeqLen = MAX_SEQ_LEN
    private val featuresPerFrame = FEATURES_PER_FRAME
    private val maxBufferFrames = MAX_BUFFER_FRAMES
    private val nJoints = 21
    private val nChannels = 3
    private val embeddingDim = EMBEDDING_DIM

    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null
    private var tfliteInterpreter: Interpreter? = null
    private var inverseVocabulary: Map<Int, String> = emptyMap()
    private var numClasses: Int = 12

    private var hasPaddingMaskInput: Boolean = false
    private var hasEmbeddingOutput: Boolean = false
    private var inputName: String = "sequence"
    private var paddingMaskInputName: String = "padding_mask"
    private var logitsOutputName: String = "logits"
    private var embeddingOutputName: String = "embedding"

    private val buffer = ArrayDeque<FloatArray>()

    // === Last classified sequence (for centroid re-decision + embeddings) ===
    @Volatile
    private var lastClassifiedSequence: List<FloatArray>? = null

    // === State machine (mirrors DynamicSignClassifier) ===
    private var state = State.IDLE
    private var lastLandmarks: FloatArray? = null
    private var stableFrameCount = 0
    private var cooldownCount = 0
    private var noHandCount = 0
    private var lastMotionLevel = 0f
    private var lastResultSign: String? = null
    private var lastResultConfidence: Float = 0f
    private var lastResultTimestamp: Long = 0L

    // === Adaptive thresholds (rebuilt by collector) ===
    @Volatile
    private var thresholds: AdaptiveThresholds = AdaptiveThresholds.DEFAULT
    private val thresholdCollector = AdaptiveThresholdCollector(context)

    enum class State { IDLE, BUFFERING, COOLDOWN }

    data class StateInfo(
        val state: State,
        val bufferSize: Int,
        val motionLevel: Float,
        val lastSign: String?,
        val lastConfidence: Float,
        val msSinceLastResult: Long,
    )

    init {
        try {
            initialize()
        } catch (e: Throwable) {
            Log.e(tag, "Init failed (${e.javaClass.simpleName}: ${e.message})", e)
        }
    }

    // ----------------------------------------------------------------------
    // Initialization
    // ----------------------------------------------------------------------

    private fun initialize() {
        inverseVocabulary = vocabulary.entries.associate { (k, v) -> v to k }
        numClasses = if (vocabulary.isNotEmpty()) vocabulary.size else 12

        val onnxFile = File(context.filesDir, "models/sign_transformer.onnx")
        if (onnxFile.exists()) {
            try {
                val env = OrtEnvironment.getEnvironment()
                ortEnv = env
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
                }
                val session = env.createSession(onnxFile.absolutePath, opts)
                ortSession = session

                val inputs = session.inputInfo
                val outputs = session.outputInfo
                inputs.keys.firstOrNull()?.let { inputName = it }
                // Detect optional second input for the padding mask.
                hasPaddingMaskInput = inputs.keys.any { it == "padding_mask" || it == "src_key_padding_mask" }
                if (hasPaddingMaskInput) {
                    paddingMaskInputName = inputs.keys.first { it == "padding_mask" || it == "src_key_padding_mask" }
                }
                // Detect optional second output for the CLS embedding.
                hasEmbeddingOutput = outputs.keys.any { it == "embedding" || it == "cls_embedding" }
                if (hasEmbeddingOutput) {
                    embeddingOutputName = outputs.keys.first { it == "embedding" || it == "cls_embedding" }
                }
                outputs.keys.firstOrNull()?.let { logitsOutputName = it }

                Log.i(
                    tag,
                    "Loaded sign_transformer ONNX (classes=$numClasses, " +
                        "padding_mask_in=$hasPaddingMaskInput, " +
                        "embedding_out=$hasEmbeddingOutput, " +
                        "inputs=${inputs.keys}, outputs=${outputs.keys})"
                )
                return
            } catch (e: Throwable) {
                Log.w(tag, "ONNX load failed, falling back to TFLite", e)
                runCatching { ortSession?.close() }
                runCatching { ortEnv?.close() }
                ortSession = null
                ortEnv = null
            }
        }

        // Fallback: bundled TFLite (the legacy GRU; same input shape).
        try {
            val modelBuffer = loadTfliteModel()
            if (modelBuffer != null) {
                tfliteInterpreter = Interpreter(
                    modelBuffer,
                    Interpreter.Options().apply { setNumThreads(2) }
                )
                Log.i(tag, "Loaded sign_transformer fallback TFLite")
            }
        } catch (e: Throwable) {
            Log.w(tag, "No transformer model available", e)
        }
    }

    // ----------------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------------

    /**
     * Apply a new set of adaptive thresholds (e.g. produced by
     * [AdaptiveThresholdCollector.rebuild]).
     */
    fun applyThresholds(newThresholds: AdaptiveThresholds) {
        thresholds = newThresholds
        Log.i(tag, "Applied thresholds: $newThresholds")
    }

    fun getThresholds(): AdaptiveThresholds = thresholds

    /**
     * Process one frame of hand landmarks. Returns a [SignClassificationResult]
     * only when the state machine decides a sign has finished.
     */
    fun processFrame(landmarks: FloatArray?): SignClassificationResult? {
        if (landmarks == null) {
            noHandCount++
            lastLandmarks = null
            lastMotionLevel = 0f

            when (state) {
                State.BUFFERING -> {
                    if (buffer.size >= thresholds.minFrames) {
                        val result = classifyBuffer()
                        if (result != null) {
                            emitResult(result)
                            state = State.COOLDOWN
                            cooldownCount = 0
                            return result
                        }
                    }
                    buffer.clear()
                    state = State.IDLE
                }
                State.IDLE -> noHandCount = 0
                State.COOLDOWN -> {
                    cooldownCount++
                    if (cooldownCount >= thresholds.cooldownFrames) {
                        state = State.IDLE
                        cooldownCount = 0
                    }
                }
            }
            return null
        }

        noHandCount = 0
        val motion = if (lastLandmarks != null) computeMotion(lastLandmarks!!, landmarks) else 0f
        lastLandmarks = landmarks.copyOf()
        lastMotionLevel = lastMotionLevel * 0.5f + motion * 0.5f
        val isMoving = lastMotionLevel > thresholds.motionThreshold

        when (state) {
            State.IDLE -> {
                if (isMoving) {
                    buffer.clear()
                    buffer.addLast(landmarks.copyOf())
                    state = State.BUFFERING
                    stableFrameCount = 0
                }
            }
            State.BUFFERING -> {
                if (isMoving) {
                    stableFrameCount = 0
                    if (buffer.size < maxBufferFrames) {
                        buffer.addLast(landmarks.copyOf())
                    } else {
                        val result = classifyBuffer()
                        if (result != null) {
                            emitResult(result)
                            state = State.COOLDOWN
                            cooldownCount = 0
                            return result
                        } else {
                            state = State.COOLDOWN
                            cooldownCount = 0
                        }
                    }
                } else {
                    if (buffer.size < thresholds.minFrames) {
                        buffer.clear()
                        state = State.IDLE
                    } else {
                        buffer.addLast(landmarks.copyOf())
                        stableFrameCount++
                        if (stableFrameCount >= thresholds.stableFramesRequired) {
                            val result = classifyBuffer()
                            if (result != null && result.confidence >= thresholds.confidenceThreshold) {
                                emitResult(result)
                                state = State.COOLDOWN
                                cooldownCount = 0
                                return result
                            } else {
                                buffer.clear()
                                state = State.IDLE
                                stableFrameCount = 0
                            }
                        }
                    }
                }
            }
            State.COOLDOWN -> {
                cooldownCount++
                if (cooldownCount >= thresholds.cooldownFrames) {
                    state = State.IDLE
                    cooldownCount = 0
                }
            }
        }
        return null
    }

    fun getStateInfo(): StateInfo {
        val now = System.currentTimeMillis()
        return StateInfo(
            state = state,
            bufferSize = buffer.size,
            motionLevel = lastMotionLevel.coerceIn(0f, 1f),
            lastSign = lastResultSign,
            lastConfidence = lastResultConfidence,
            msSinceLastResult = if (lastResultTimestamp == 0L) -1L else (now - lastResultTimestamp),
        )
    }

    fun clearBuffer() {
        buffer.clear()
        lastClassifiedSequence = null
        noHandCount = 0
        state = State.IDLE
        stableFrameCount = 0
        cooldownCount = 0
    }

    fun getBufferSize(): Int = buffer.size

    /**
     * Snapshot of the sequence that produced the most recent successful
     * classification. The facade uses it to apply the centroid override
     * after the state machine clears its internal buffer. Returns null
     * if no classification has happened yet in this session.
     */
    fun lastClassifiedSequence(): List<FloatArray>? = lastClassifiedSequence

    /**
     * Extract the 128-dim CLS-pooled embedding for a full sequence of
     * landmarks. Used by the calibration flow to compute centroids and by
     * the [CentroidClassifier] for few-shot personalization.
     *
     * Returns `null` if the model was not exported with an `embedding`
     * output (the current FP32 export only emits `logits`; re-export
     * with `output_names=["logits", "embedding"]` to enable this).
     *
     * @param sequence list of 63-dim landmark frames (length T)
     * @return FloatArray(128) or null
     */
    fun extractEmbedding(sequence: List<FloatArray>): FloatArray? {
        if (sequence.isEmpty()) return null
        val session = ortSession ?: return null
        if (!hasEmbeddingOutput) {
            Log.w(tag, "extractEmbedding called but ONNX has no 'embedding' output; re-export the model to enable")
            return null
        }
        val env = ortEnv ?: return null
        return try {
            val (tensorInput, mask) = buildTransformerInputs(sequence)
            val shape = longArrayOf(1L, maxSeqLen.toLong(), nJoints.toLong(), nChannels.toLong())
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorInput), shape)

            val inputs = HashMap<String, OnnxTensor>(2)
            inputs[inputName] = inputTensor
            if (hasPaddingMaskInput) {
                val maskShape = longArrayOf(1L, maxSeqLen.toLong())
                // Bool padding mask via ByteBuffer + OnnxJavaType.BOOL.
                val maskBytes = ByteArray(mask.size) { if (mask[it]) 1.toByte() else 0.toByte() }
                val maskTensor = OnnxTensor.createTensor(
                    env,
                    ByteBuffer.wrap(maskBytes),
                    maskShape,
                    OnnxJavaType.BOOL
                )
                inputs[paddingMaskInputName] = maskTensor
            }

            val outputs = session.run(inputs)
            // session.run(...) returns OrtSession.Result, not a Map. The
            // indexer delegates to Result.get(String) which is Optional<OnnxValue>.
            val embRaw: Any? = outputs[embeddingOutputName]?.get()?.getValue()
            // The CLS token output is (1, embed_dim) — pull the first row.
            val emb: FloatArray = when (val v = embRaw) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    val arr = v as Array<FloatArray>
                    if (arr.isEmpty()) FloatArray(0) else arr[0]
                }
                is FloatArray -> v
                else -> FloatArray(0)
            }
            outputs.close()
            inputTensor.close()
            if (hasPaddingMaskInput) inputs[paddingMaskInputName]?.close()
            emb
        } catch (e: Exception) {
            Log.e(tag, "extractEmbedding failed", e)
            null
        }
    }

    fun reload(newVocabulary: Map<String, Int>? = null) {
        close()
        if (newVocabulary != null) {
            inverseVocabulary = newVocabulary.entries.associate { (k, v) -> v to k }
            numClasses = newVocabulary.size
        }
        initialize()
        state = State.IDLE
        buffer.clear()
        lastClassifiedSequence = null
        stableFrameCount = 0
        cooldownCount = 0
    }

    fun close() {
        ortSession?.close(); ortSession = null
        ortEnv?.close(); ortEnv = null
        tfliteInterpreter?.close(); tfliteInterpreter = null
        buffer.clear()
    }

    // ----------------------------------------------------------------------
    // Internals
    // ----------------------------------------------------------------------

    private fun emitResult(result: SignClassificationResult) {
        lastResultSign = result.signName
        lastResultConfidence = result.confidence
        lastResultTimestamp = System.currentTimeMillis()

        // Hand the sample to the adaptive-threshold collector; it decides
        // when enough samples have accumulated to rebuild thresholds.
        thresholdCollector.record(
            highMotionLevel = lastMotionLevel,
            bufferSize = buffer.size,
        )
        if (thresholdCollector.shouldUpdate()) {
            applyThresholds(thresholdCollector.rebuild())
        }
    }

    private fun computeMotion(prev: FloatArray, curr: FloatArray): Float {
        require(prev.size == curr.size)
        var sum = 0.0
        var i = 0
        while (i < prev.size) {
            val dx = (curr[i] - prev[i]).toDouble()
            val dy = (curr[i + 1] - prev[i + 1]).toDouble()
            val dz = (curr[i + 2] - prev[i + 2]).toDouble()
            sum += sqrt(dx * dx + dy * dy + dz * dz)
            i += 3
        }
        return (sum / nJoints.toDouble()).toFloat()
    }

    private fun classifyBuffer(): SignClassificationResult? {
        if (buffer.isEmpty()) return null
        val asList = buffer.toList()
        // Capture snapshot for centroid override / extractEmbedding BEFORE
        // the state machine clears the buffer.
        lastClassifiedSequence = asList.map { it.copyOf() }
        val result = when {
            ortSession != null -> classifyOnnx(asList)
            tfliteInterpreter != null -> classifyTflite(asList)
            else -> null
        }
        buffer.clear()
        return result
    }

    /**
     * Build a (1, T, 21, 3) zero-padded input tensor plus the (T,) bool
     * padding mask `(seq.abs().sum(dim=(2,3)) < 1e-6)`. The mask is
     * returned alongside the tensor so callers that have a model with
     * a `padding_mask` input can pass it; the state machine also uses
     * it to count real (non-padded) frames in the buffer.
     */
    private fun buildTransformerInputs(sequence: List<FloatArray>): Pair<FloatArray, BooleanArray> {
        val tensor = FloatArray(maxSeqLen * nJoints * nChannels)
        val mask = BooleanArray(maxSeqLen)
        val framesToCopy = minOf(sequence.size, maxSeqLen)
        for (i in 0 until framesToCopy) {
            val frame = sequence[i]
            val dst = i * featuresPerFrame
            val n = minOf(frame.size, featuresPerFrame)
            System.arraycopy(frame, 0, tensor, dst, n)
            // Mask = True when the frame is "all zero" (padded/empty).
            var sumAbs = 0f
            for (j in 0 until n) sumAbs += abs(frame[j])
            mask[i] = sumAbs < 1e-6f
        }
        // Tail frames (i >= framesToCopy) are zero by construction; mark them padded.
        for (i in framesToCopy until maxSeqLen) mask[i] = true
        return tensor to mask
    }

    private fun classifyOnnx(sequence: List<FloatArray>): SignClassificationResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null
        return try {
            val (tensor, mask) = buildTransformerInputs(sequence)
            val shape = longArrayOf(1L, maxSeqLen.toLong(), nJoints.toLong(), nChannels.toLong())
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensor), shape)

            val inputs = HashMap<String, OnnxTensor>(2)
            inputs[inputName] = inputTensor
            if (hasPaddingMaskInput) {
                val maskShape = longArrayOf(1L, maxSeqLen.toLong())
                val maskBytes = ByteArray(mask.size) { if (mask[it]) 1.toByte() else 0.toByte() }
                val maskTensor = OnnxTensor.createTensor(
                    env,
                    ByteBuffer.wrap(maskBytes),
                    maskShape,
                    OnnxJavaType.BOOL
                )
                inputs[paddingMaskInputName] = maskTensor
            }

            val outputs = session.run(inputs)
            val raw: Any? = outputs[logitsOutputName]?.get()?.getValue()
            val logits: FloatArray = when (val v = raw) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    (v as Array<FloatArray>)[0]
                }
                is FloatArray -> v
                else -> FloatArray(numClasses)
            }
            outputs.close()
            inputTensor.close()
            if (hasPaddingMaskInput) inputs[paddingMaskInputName]?.close()

            val probs = softmax(logits)
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val word = inverseVocabulary[maxIdx] ?: "???"
            SignClassificationResult(word, word, probs[maxIdx], maxIdx)
        } catch (e: Exception) {
            Log.e(tag, "ONNX inference failed", e)
            null
        }
    }

    private fun classifyTflite(sequence: List<FloatArray>): SignClassificationResult? {
        val interp = tfliteInterpreter ?: return null
        val flat = FloatArray(maxSeqLen * featuresPerFrame)
        val framesToCopy = minOf(sequence.size, maxSeqLen)
        for (i in 0 until framesToCopy) {
            val frame = sequence[i]
            val n = minOf(frame.size, featuresPerFrame)
            System.arraycopy(frame, 0, flat, i * featuresPerFrame, n)
        }
        val inputBuffer = ByteBuffer.allocateDirect(flat.size * 4).apply {
            order(ByteOrder.nativeOrder())
            for (v in flat) putFloat(v)
            rewind()
        }
        val outputBuffer = Array(1) { FloatArray(numClasses) }
        interp.run(inputBuffer, outputBuffer)
        val probs = softmax(outputBuffer[0])
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val word = inverseVocabulary[maxIdx] ?: "???"
        return SignClassificationResult(word, word, probs[maxIdx], maxIdx)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size) {
            kotlin.math.exp((logits[it] - max).toDouble()).toFloat()
        }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }

    private fun loadTfliteModel(): ByteBuffer? {
        val downloaded = File(context.filesDir, "models/sign_transformer.tflite")
        if (downloaded.exists()) return loadFromFile(downloaded)
        return try { loadFromAssets("sign_transformer.tflite") } catch (e: Exception) { null }
    }

    private fun loadFromFile(file: File): MappedByteBuffer {
        return FileInputStream(file).channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun loadFromAssets(name: String): MappedByteBuffer {
        val afd = context.assets.openFd(name)
        return FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
        )
    }

    companion object {
        // === Tunables (start with the legacy GRU constants; AdaptiveThresholdCollector
        //     will rebuild them from per-user motion statistics at runtime).
        const val MAX_SEQ_LEN = 60
        const val FEATURES_PER_FRAME = 63
        const val MAX_BUFFER_FRAMES = 90
        const val EMBEDDING_DIM = 128
    }
}
