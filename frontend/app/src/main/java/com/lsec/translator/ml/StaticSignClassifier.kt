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
 * Clasificador de señas estáticas.
 *
 * Busca el modelo en este orden:
 * 1. filesDir/models/sign_mlp.onnx  (descargado del servidor - preferido)
 * 2. filesDir/models/sign_mlp.tflite (descargado, formato legacy)
 * 3. assets/sign_mlp.tflite          (bundled en el APK)
 *
 * Input: FloatArray(63) — landmarks normalizados
 * Output: (label, confidence)
 */
class StaticSignClassifier(
    private val context: Context,
    private val labelMap: Map<String, Int> = emptyMap()
) {
    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null
    private var tfliteInterpreter: Interpreter? = null
    private var inverseLabelMap: Map<Int, String> = emptyMap()
    private var numClasses: Int = 27

    init {
        initialize()
    }

    private fun initialize(overrideLabelMap: Map<String, Int>? = null) {
        val effectiveMap = overrideLabelMap ?: labelMap
        inverseLabelMap = effectiveMap.entries.associate { (k, v) -> v to k }
        numClasses = if (effectiveMap.isNotEmpty()) effectiveMap.size else 27

        // Try ONNX first (downloaded from server)
        val onnxFile = File(context.filesDir, "models/sign_mlp.onnx")
        if (onnxFile.exists()) {
            try {
                val env = OrtEnvironment.getEnvironment()
                ortEnv = env
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.NO_OPT)
                }
                ortSession = env.createSession(onnxFile.absolutePath, opts)
                Log.i(TAG, "Static classifier loaded via ONNX Runtime ($numClasses classes)")
                return
            } catch (e: Throwable) {
                Log.w(TAG, "ONNX load failed (${e.javaClass.simpleName}: ${e.message}), falling back to TFLite", e)
                runCatching { ortSession?.close() }
                runCatching { ortEnv?.close() }
                ortSession = null
                ortEnv = null
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
                Log.i(TAG, "Static classifier loaded via TFLite ($numClasses classes)")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "No static classifier available (${e.javaClass.simpleName}: ${e.message})", e)
        }
    }

    fun classify(landmarks: FloatArray): SignClassificationResult? {
        return when {
            ortSession != null -> classifyOnnx(landmarks)
            tfliteInterpreter != null -> classifyTflite(landmarks)
            else -> null
        }
    }

    private fun classifyOnnx(landmarks: FloatArray): SignClassificationResult? {
        val env = ortEnv ?: return null
        val session = ortSession ?: return null
        return try {
            val shape = longArrayOf(1L, landmarks.size.toLong())
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(landmarks), shape)
            val result = session.run(mapOf(session.inputNames.first() to tensor))
            @Suppress("UNCHECKED_CAST")
            val logits = (result[0].value as Array<FloatArray>)[0]
            result.close()
            tensor.close()
            val probs = softmax(logits)
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val label = inverseLabelMap[maxIdx] ?: "Desconocido"
            SignClassificationResult(label, label, probs[maxIdx], maxIdx)
        } catch (e: Exception) {
            Log.e(TAG, "ONNX inference error", e)
            null
        }
    }

    private fun classifyTflite(landmarks: FloatArray): SignClassificationResult? {
        val interp = tfliteInterpreter ?: return null
        val inputBuffer = ByteBuffer.allocateDirect(landmarks.size * 4).apply {
            order(ByteOrder.nativeOrder())
            landmarks.forEach { putFloat(it) }
            rewind()
        }
        val outputBuffer = Array(1) { FloatArray(numClasses) }
        interp.run(inputBuffer, outputBuffer)
        val probs = softmax(outputBuffer[0])
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val label = inverseLabelMap[maxIdx] ?: "Desconocido"
        return SignClassificationResult(label, label, probs[maxIdx], maxIdx)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size) { kotlin.math.exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }

    private fun loadTfliteModel(): ByteBuffer? {
        val downloaded = File(context.filesDir, "models/sign_mlp.tflite")
        if (downloaded.exists()) return loadFromFile(downloaded)
        return try { loadFromAssets("sign_mlp.tflite") } catch (e: Exception) { null }
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

    fun reload(newLabelMap: Map<String, Int>? = null) {
        close()
        initialize(newLabelMap)
    }

    fun close() {
        ortSession?.close(); ortSession = null
        ortEnv?.close(); ortEnv = null
        tfliteInterpreter?.close(); tfliteInterpreter = null
    }

    companion object {
        private const val TAG = "StaticSignClassifier"
    }
}
