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

/**
 * Clasificador de señas dinámicas (GRU).
 *
 * Busca el modelo en este orden:
 * 1. filesDir/models/sign_gru.onnx  (descargado del servidor - preferido)
 * 2. filesDir/models/sign_gru.tflite (descargado, formato legacy)
 * 3. assets/sign_gru.tflite          (bundled en el APK)
 *
 * Mantiene un buffer de frames y clasifica cuando el buffer se llena
 * o cuando la mano desaparece tras NO_HAND_THRESHOLD frames.
 */
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
    private val featuresPerFrame = 63
    private val buffer = mutableListOf<FloatArray>()
    private var noHandCount = 0

    init {
        initialize()
    }

    private fun initialize() {
        inverseVocabulary = vocabulary.entries.associate { (k, v) -> v to k }
        numClasses = if (vocabulary.isNotEmpty()) vocabulary.size else 50

        // Try ONNX first (downloaded from server)
        val onnxFile = File(context.filesDir, "models/sign_gru.onnx")
        if (onnxFile.exists()) {
            try {
                val env = OrtEnvironment.getEnvironment()
                ortEnv = env
                ortSession = env.createSession(
                    onnxFile.absolutePath,
                    OrtSession.SessionOptions()
                )
                Log.i(TAG, "Dynamic classifier loaded via ONNX Runtime ($numClasses classes)")
                return
            } catch (e: Exception) {
                Log.w(TAG, "ONNX load failed, falling back to TFLite", e)
                ortSession?.close(); ortSession = null
                ortEnv?.close(); ortEnv = null
            }
        }

        // Try TFLite (downloaded or bundled in assets)
        try {
            val modelBuffer = loadTfliteModel()
            if (modelBuffer != null) {
                tfliteInterpreter = Interpreter(
                    modelBuffer,
                    Interpreter.Options().apply { setNumThreads(2) }
                )
                Log.i(TAG, "Dynamic classifier loaded via TFLite ($numClasses classes)")
            }
        } catch (e: Exception) {
            Log.w(TAG, "No dynamic classifier available", e)
        }
    }

    fun processFrame(landmarks: FloatArray?): SignClassificationResult? {
        if (landmarks != null) {
            noHandCount = 0
            buffer.add(landmarks.copyOf())
            if (buffer.size >= maxSeqLen) return classifyBuffer()
        } else {
            noHandCount++
            if (noHandCount >= NO_HAND_THRESHOLD && buffer.size > MIN_FRAMES) {
                return classifyBuffer()
            }
        }
        return null
    }

    private fun classifyBuffer(): SignClassificationResult? {
        if (buffer.isEmpty()) return null

        val sequence = FloatArray(maxSeqLen * featuresPerFrame)
        val framesToCopy = minOf(buffer.size, maxSeqLen)
        for (i in 0 until framesToCopy) {
            System.arraycopy(buffer[i], 0, sequence, i * featuresPerFrame, featuresPerFrame)
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
            // Shape: (1, maxSeqLen, featuresPerFrame)
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
    }

    fun getBufferSize(): Int = buffer.size

    fun reload(newVocabulary: Map<String, Int>? = null) {
        close()
        if (newVocabulary != null) {
            inverseVocabulary = newVocabulary.entries.associate { (k, v) -> v to k }
            numClasses = newVocabulary.size
        }
        initialize()
    }

    fun close() {
        ortSession?.close(); ortSession = null
        ortEnv?.close(); ortEnv = null
        tfliteInterpreter?.close(); tfliteInterpreter = null
        buffer.clear()
    }

    companion object {
        private const val TAG = "DynamicSignClassifier"
        private const val NO_HAND_THRESHOLD = 15
        private const val MIN_FRAMES = 10
    }
}
