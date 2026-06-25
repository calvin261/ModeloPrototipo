package com.lsec.translator.ml

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
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
import kotlin.math.sqrt

class DynamicSignClassifier(
    private val context: Context,
    private val vocabulary: Map<String, Int> = emptyMap()
) {
    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null
    private var tfliteInterpreter: Interpreter? = null
    private var inverseVocabulary: Map<Int, String> = emptyMap()
    private var numClasses: Int = 50

    private val maxSeqLen = 60
    private val featuresPerFrame = 147
    private val maxBufferFrames = MAX_BUFFER_FRAMES
    private val buffer = ArrayDeque<FloatArray>()

    private var state = State.IDLE
    private var lastLandmarks: FloatArray? = null
    private var stableFrameCount = 0
    private var cooldownCount = 0
    private var noHandCount = 0
    private var lastMotionLevel = 0f
    private var lastResultSign: String? = null
    private var lastResultConfidence: Float = 0f
    private var lastResultTimestamp: Long = 0L

    init {
        initialize()
    }

    private fun initialize() {
        inverseVocabulary = vocabulary.entries.associate { (k, v) -> v to k }
        numClasses = if (vocabulary.isNotEmpty()) vocabulary.size else 50

        // Try ONNX first (downloaded from server)
        val onnxFile = File(context.filesDir, "models/sign_transformer.onnx")
        if (onnxFile.exists()) {
            try {
                val env = OrtEnvironment.getEnvironment()
                ortEnv = env
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.NO_OPT)
                }
                ortSession = env.createSession(onnxFile.absolutePath, opts)
                Log.i(TAG, "Dynamic classifier loaded via ONNX Runtime ($numClasses classes)")
                return
            } catch (e: Throwable) {
                Log.w(TAG, "ONNX load failed (${e.javaClass.simpleName}: ${e.message}), falling back to TFLite", e)
                runCatching { ortSession?.close() }
                runCatching { ortEnv?.close() }
                ortSession = null
                ortEnv = null
            }
        }

        try {
            val modelBuffer = loadTfliteModel()
            if (modelBuffer != null) {
                tfliteInterpreter = Interpreter(
                    modelBuffer,
                    Interpreter.Options().apply { setNumThreads(2) }
                )
                Log.i(TAG, "Dynamic classifier loaded via TFLite ($numClasses classes)")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "No dynamic classifier available (${e.javaClass.simpleName}: ${e.message})", e)
        }
    }

    enum class State { IDLE, BUFFERING, COOLDOWN }

    data class StateInfo(
        val state: State,
        val bufferSize: Int,
        val motionLevel: Float,
        val lastSign: String?,
        val lastConfidence: Float,
        val msSinceLastResult: Long,
    )

    fun processFrame(landmarks: FloatArray?): SignClassificationResult? {
        if (landmarks == null) {
            noHandCount++
            lastLandmarks = null
            lastMotionLevel = 0f

            when (state) {
                State.BUFFERING -> {
                    if (buffer.size >= MIN_FRAMES) {
                        val result = classifyBuffer()
                        if (result != null) emitResult(result)
                        state = State.COOLDOWN
                        cooldownCount = 0
                        return result
                    }
                    buffer.clear()
                    state = State.IDLE
                }
                State.IDLE -> {
                    noHandCount = 0
                }
                State.COOLDOWN -> {
                    cooldownCount++
                    if (cooldownCount >= COOLDOWN_FRAMES) {
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
        val isMoving = lastMotionLevel > MOTION_THRESHOLD

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
                    if (buffer.size < MIN_FRAMES) {
                        buffer.clear()
                        state = State.IDLE
                    } else {
                        buffer.addLast(landmarks.copyOf())
                        stableFrameCount++
                        if (stableFrameCount >= STABLE_FRAMES_REQUIRED) {
                            val result = classifyBuffer()
                            if (result != null && result.confidence >= CONFIDENCE_THRESHOLD) {
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
                if (cooldownCount >= COOLDOWN_FRAMES) {
                    state = State.IDLE
                    cooldownCount = 0
                }
            }
        }
        return null
    }

    private fun emitResult(result: SignClassificationResult) {
        lastResultSign = result.signName
        lastResultConfidence = result.confidence
        lastResultTimestamp = System.currentTimeMillis()
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

    private fun computeMotion(prev: FloatArray, curr: FloatArray): Float {
        require(prev.size == curr.size)
        var sum = 0.0
        var i = 0
        val nKeypoints = prev.size / 3
        while (i < prev.size) {
            val dx = (curr[i] - prev[i]).toDouble()
            val dy = (curr[i + 1] - prev[i + 1]).toDouble()
            val dz = (curr[i + 2] - prev[i + 2]).toDouble()
            sum += sqrt(dx * dx + dy * dy + dz * dz)
            i += 3
        }
        return (sum / nKeypoints).toFloat()
    }

    private fun classifyBuffer(): SignClassificationResult? {
        if (buffer.isEmpty()) return null

        val sequence = FloatArray(maxSeqLen * featuresPerFrame)
        val framesToCopy = minOf(buffer.size, maxSeqLen)
        var i = 0
        for (frame in buffer) {
            if (i >= framesToCopy) break
            System.arraycopy(frame, 0, sequence, i * featuresPerFrame, featuresPerFrame)
            i++
        }
        buffer.clear()

        return when {
            ortSession != null -> classifySequenceOnnx(sequence)
            tfliteInterpreter != null -> classifySequenceTflite(sequence)
            else -> null
        }
    }

    private fun classifySequenceOnnx(sequence: FloatArray): SignClassificationResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null
        return try {
            val shape = longArrayOf(1L, maxSeqLen.toLong(), featuresPerFrame.toLong())
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sequence), shape)
            val result = session.run(mapOf(session.inputNames.first() to tensor))
            @Suppress("UNCHECKED_CAST")
            val logits = (result[0].value as Array<FloatArray>)[0]
            result.close()
            tensor.close()
            val probs = softmax(logits)
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val word = inverseVocabulary[maxIdx] ?: "???"
            SignClassificationResult(word, word, probs[maxIdx], maxIdx)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX sequence inference error", e)
            null
        }
    }

    private fun classifySequenceTflite(sequence: FloatArray): SignClassificationResult? {
        val interp = tfliteInterpreter ?: return null
        val inputBuffer = ByteBuffer.allocateDirect(sequence.size * 4).apply {
            order(ByteOrder.nativeOrder())
            sequence.forEach { putFloat(it) }
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
        val exps = FloatArray(logits.size) { kotlin.math.exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }

    private fun loadTfliteModel(): ByteBuffer? {
        val downloaded = File(context.filesDir, "models/sign_gru.tflite")
        if (downloaded.exists()) return loadFromFile(downloaded)
        return try { loadFromAssets("sign_gru.tflite") } catch (e: Exception) { null }
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

    fun clearBuffer() {
        buffer.clear()
        noHandCount = 0
        state = State.IDLE
        stableFrameCount = 0
        cooldownCount = 0
    }

    fun getBufferSize(): Int = buffer.size

    fun reload(newVocabulary: Map<String, Int>? = null) {
        close()
        if (newVocabulary != null) {
            inverseVocabulary = newVocabulary.entries.associate { (k, v) -> v to k }
            numClasses = newVocabulary.size
        }
        initialize()
        state = State.IDLE
        buffer.clear()
        stableFrameCount = 0
        cooldownCount = 0
    }

    fun close() {
        ortSession?.close(); ortSession = null
        ortEnv?.close(); ortEnv = null
        tfliteInterpreter?.close(); tfliteInterpreter = null
        buffer.clear()
    }

    companion object {
        private const val TAG = "DynamicSignClassifier"

        const val MOTION_THRESHOLD = 0.04f
        const val STABLE_FRAMES_REQUIRED = 6
        const val MIN_FRAMES = 15
        const val MAX_BUFFER_FRAMES = 90
        const val COOLDOWN_FRAMES = 12
        const val CONFIDENCE_THRESHOLD = 0.35f
    }
}