package com.lsec.translator.ml

import android.content.Context
import android.util.Log
import java.io.File

/**
 * On-device centroid-based few-shot classifier for sign personalization.
 *
 * This class implements a simple but effective personalization approach:
 *   1. During a 30-60s calibration flow, the user records 3-5 examples
 *      of each sign.
 *   2. We extract the model's penultimate-layer embedding for each example
 *      (128-d for the SignTransformer, 256-d for the legacy GRU).
 *   3. We average the embeddings into a single "centroid" per sign.
 *   4. At inference time, we extract the embedding of the live gesture
 *      and pick the sign with the highest cosine similarity to any
 *      centroid.
 *
 * The advantage over a full retrain is that this requires ZERO gradient
 * updates — only forward passes — so it works with ONNX Runtime on the
 * phone without any training subgraph.
 *
 * The interface mirrors `DynamicSignClassifier` so the rest of the app
 * can switch between base-model classification and personalized
 * classification transparently.
 *
 * Persistence: centroids are stored as a float32 binary file in
 * `filesDir/calibration/<userId>.bin` so multiple users can have
 * separate calibrations on the same device.
 */
class CentroidClassifier(
    private val context: Context,
    private val vocabulary: Map<String, Int>,
    private val embeddingDim: Int,
    private val userId: String = "default",
) {
    private val tag = "CentroidClassifier"

    @Volatile
    private var centroids: Map<String, FloatArray>? = null
    @Volatile
    private var nSamplesPerClass: Map<String, Int> = emptyMap()

    init {
        load()
    }

    /**
     * Set the centroids from a recorded calibration session.
     * Each map entry is (sign_name -> list_of_N_embeddings).
     */
    fun setCentroids(perSignEmbeddings: Map<String, List<FloatArray>>) {
        val newCentroids = HashMap<String, FloatArray>(perSignEmbeddings.size)
        val counts = HashMap<String, Int>(perSignEmbeddings.size)
        for ((sign, embs) in perSignEmbeddings) {
            if (embs.isEmpty()) continue
            val centroid = FloatArray(embeddingDim)
            for (emb in embs) {
                for (i in 0 until embeddingDim) {
                    centroid[i] += emb[i]
                }
            }
            for (i in 0 until embeddingDim) centroid[i] /= embs.size
            newCentroids[sign] = centroid
            counts[sign] = embs.size
        }
        centroids = newCentroids
        nSamplesPerClass = counts
        save()
        Log.i(tag, "Calibrated with ${newCentroids.size} signs: $counts")
    }

    /**
     * Classify an embedding by cosine similarity to the nearest centroid.
     * Returns null if no centroids are loaded or similarity is below threshold.
     */
    fun classify(embedding: FloatArray, threshold: Float = 0.5f): Pair<String, Float>? {
        val c = centroids ?: return null
        var bestSign: String? = null
        var bestSim = -1f
        for ((sign, centroid) in c) {
            val sim = cosine(embedding, centroid)
            if (sim > bestSim) {
                bestSim = sim
                bestSign = sign
            }
        }
        if (bestSign == null || bestSim < threshold) return null
        return bestSign to bestSim
    }

    fun isCalibrated(): Boolean = centroids?.isNotEmpty() == true
    fun getCentroidCount(): Int = centroids?.size ?: 0
    fun getSamplesPerClass(): Map<String, Int> = nSamplesPerClass

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size && a.size == embeddingDim) {
            "Embedding dim mismatch: ${a.size} vs ${b.size}, expected $embeddingDim"
        }
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in 0 until embeddingDim) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        return (dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9)).toFloat()
    }

    // ---------- persistence ----------

    private fun storageFile(): File {
        val dir = File(context.filesDir, "calibration")
        dir.mkdirs()
        return File(dir, "${userId}.bin")
    }

    private fun save() {
        val c = centroids ?: return
        try {
            val file = storageFile()
            // Format: [n_classes (4B)] [for each class: name_len (4B) UTF-8 bytes, n_samples (4B), dim (4B), dim * 4B floats]
            val baos = java.io.ByteArrayOutputStream()
            val dos = java.io.DataOutputStream(baos)
            dos.writeInt(c.size)
            for ((sign, emb) in c) {
                val bytes = sign.toByteArray(Charsets.UTF_8)
                dos.writeInt(bytes.size)
                dos.write(bytes)
                dos.writeInt(nSamplesPerClass[sign] ?: 0)
                dos.writeInt(embeddingDim)
                for (v in emb) dos.writeFloat(v)
            }
            file.writeBytes(baos.toByteArray())
            Log.i(tag, "Saved centroids (${c.size} signs) to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(tag, "Failed to save centroids", e)
        }
    }

    private fun load() {
        val file = storageFile()
        if (!file.exists()) return
        try {
            val data = file.readBytes()
            val dis = java.io.DataInputStream(java.io.ByteArrayInputStream(data))
            val nClasses = dis.readInt()
            val newCentroids = HashMap<String, FloatArray>(nClasses)
            val newCounts = HashMap<String, Int>(nClasses)
            repeat(nClasses) {
                val nameLen = dis.readInt()
                val nameBytes = ByteArray(nameLen)
                dis.readFully(nameBytes)
                val name = String(nameBytes, Charsets.UTF_8)
                val n = dis.readInt()
                val dim = dis.readInt()
                require(dim == embeddingDim) { "Embedding dim mismatch in saved file: $dim vs $embeddingDim" }
                val emb = FloatArray(dim)
                for (i in 0 until dim) emb[i] = dis.readFloat()
                newCentroids[name] = emb
                newCounts[name] = n
            }
            centroids = newCentroids
            nSamplesPerClass = newCounts
            Log.i(tag, "Loaded ${newCentroids.size} centroids from ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(tag, "Failed to load centroids, ignoring", e)
        }
    }

    fun clearCalibration() {
        centroids = null
        nSamplesPerClass = emptyMap()
        storageFile().delete()
    }
}
