package com.lsec.translator.ml

import android.content.Context
import android.util.Log
import kotlin.math.max
import kotlin.math.min

/**
 * Adaptive thresholds for the dynamic-sign state machine.
 *
 * The legacy implementation (DynamicSignClassifier.kt) hard-codes constants
 * like MOTION_THRESHOLD=0.04, STABLE_FRAMES_REQUIRED=6, MIN_FRAMES=15,
 * MAX_BUFFER_FRAMES=90. These were tuned for the LSC50 dataset signers but
 * fail on real users with different signing speeds and hand sizes.
 *
 * This class stores per-user statistics gathered during a brief
 * calibration run (or passively during normal use) and converts them
 * into per-user thresholds via simple heuristics.
 *
 * Persistence: state is encoded as JSON in SharedPreferences (DataStore
 * would also work) and reloaded on the next launch.
 *
 * Heuristics (chosen empirically from the LSC50 dataset):
 *   MOTION_THRESHOLD        = 0.5 * median(high_motion_level)
 *   STABLE_FRAMES_REQUIRED  = clamp(0.1 * median(buffer_size), 3, 12)
 *   MIN_FRAMES              = clamp(0.5 * median(buffer_size), 10, 25)
 *   MAX_BUFFER_FRAMES       = clamp(2.0 * median(buffer_size), 60, 120)
 *   COOLDOWN_FRAMES         = 12 (kept fixed; tied to TTS lockout, not user style)
 *   CONFIDENCE_THRESHOLD    = 0.35 (kept fixed; can be raised with calibration)
 */
data class AdaptiveThresholds(
    val motionThreshold: Float,
    val stableFramesRequired: Int,
    val minFrames: Int,
    val maxBufferFrames: Int,
    val cooldownFrames: Int = 12,
    val confidenceThreshold: Float = 0.35f,
    val sampleCount: Int = 0,
    val collectedAtMs: Long = System.currentTimeMillis(),
) {
    /**
     * Persist as a single string in SharedPreferences. Format is
     * semicolon-separated `key=value` pairs for easy forward compatibility.
     */
    fun serialize(): String = listOf(
        "motionThreshold" to motionThreshold,
        "stableFramesRequired" to stableFramesRequired,
        "minFrames" to minFrames,
        "maxBufferFrames" to maxBufferFrames,
        "cooldownFrames" to cooldownFrames,
        "confidenceThreshold" to confidenceThreshold,
        "sampleCount" to sampleCount,
        "collectedAtMs" to collectedAtMs,
    ).joinToString(";") { "${it.first}=${it.second}" }

    companion object {
        private const val TAG = "AdaptiveThresholds"

        /** Conservative defaults (the legacy values). */
        val DEFAULT = AdaptiveThresholds(
            motionThreshold = 0.04f,
            stableFramesRequired = 6,
            minFrames = 15,
            maxBufferFrames = 90,
            sampleCount = 0,
        )

        /**
         * Build adaptive thresholds from a history of per-classification
         * samples. If `samples` is empty, returns DEFAULT.
         *
         * Each sample is a (high_motion_level, buffer_size) pair measured
         * during one successful classification.
         */
        fun fromSamples(
            highMotionLevels: List<Float>,
            bufferSizes: List<Int>,
        ): AdaptiveThresholds {
            if (highMotionLevels.isEmpty() || bufferSizes.isEmpty()) {
                return DEFAULT
            }
            val sortedMotion = highMotionLevels.sorted()
            val sortedBuffers = bufferSizes.sorted()
            fun percentile(p: Double): Float {
                val idx = ((sortedMotion.size - 1) * p).toInt().coerceIn(0, sortedMotion.size - 1)
                return sortedMotion[idx]
            }
            val medMotion = percentile(0.5)
            val medBuffer = sortedBuffers[sortedBuffers.size / 2]

            return AdaptiveThresholds(
                motionThreshold = (medMotion * 0.5f).coerceIn(0.02f, 0.15f),
                stableFramesRequired = (medBuffer * 0.1f).toInt().coerceIn(3, 12),
                minFrames = (medBuffer * 0.5f).toInt().coerceIn(10, 25),
                maxBufferFrames = (medBuffer * 2.0f).toInt().coerceIn(60, 120),
                sampleCount = min(highMotionLevels.size, bufferSizes.size),
            )
        }

        @JvmStatic
        fun deserialize(raw: String?): AdaptiveThresholds {
            if (raw.isNullOrBlank()) return DEFAULT
            return try {
                val map = raw.split(";")
                    .mapNotNull {
                        val parts = it.split("=")
                        if (parts.size != 2) null else parts[0] to parts[1]
                    }
                    .toMap()
                AdaptiveThresholds(
                    motionThreshold = map["motionThreshold"]?.toFloatOrNull() ?: DEFAULT.motionThreshold,
                    stableFramesRequired = map["stableFramesRequired"]?.toIntOrNull() ?: DEFAULT.stableFramesRequired,
                    minFrames = map["minFrames"]?.toIntOrNull() ?: DEFAULT.minFrames,
                    maxBufferFrames = map["maxBufferFrames"]?.toIntOrNull() ?: DEFAULT.maxBufferFrames,
                    cooldownFrames = map["cooldownFrames"]?.toIntOrNull() ?: DEFAULT.cooldownFrames,
                    confidenceThreshold = map["confidenceThreshold"]?.toFloatOrNull() ?: DEFAULT.confidenceThreshold,
                    sampleCount = map["sampleCount"]?.toIntOrNull() ?: 0,
                    collectedAtMs = map["collectedAtMs"]?.toLongOrNull() ?: System.currentTimeMillis(),
                )
            } catch (e: Exception) {
                Log.w(TAG, "Failed to deserialize adaptive thresholds, using default", e)
                DEFAULT
            }
        }
    }
}

/**
 * Helper that records per-classification statistics and rebuilds
 * `AdaptiveThresholds` once enough samples have been collected.
 *
 * Typical usage from a ViewModel:
 *   val collector = AdaptiveThresholdCollector(context)
 *   // after each successful classification:
 *   collector.record(highMotionLevel = stateInfo.motionLevel, bufferSize = stateInfo.bufferSize)
 *   if (collector.shouldUpdate()) {
 *     val thresholds = collector.rebuild()
 *     classifier.applyThresholds(thresholds)
 *   }
 */
class AdaptiveThresholdCollector(context: Context) {
    private val prefs = context.getSharedPreferences("adaptive_thresholds", Context.MODE_PRIVATE)
    private val motionSamples = mutableListOf<Float>()
    private val bufferSamples = mutableListOf<Int>()

    @Volatile
    private var currentThresholds: AdaptiveThresholds =
        AdaptiveThresholds.deserialize(prefs.getString("thresholds", null))

    fun getCurrent(): AdaptiveThresholds = currentThresholds

    fun record(highMotionLevel: Float, bufferSize: Int) {
        synchronized(this) {
            motionSamples.add(highMotionLevel)
            bufferSamples.add(bufferSize)
            // Keep only the last 200 samples to bound memory.
            if (motionSamples.size > 200) {
                motionSamples.removeAt(0)
                bufferSamples.removeAt(0)
            }
        }
    }

    /** Returns true if we have enough new samples to justify rebuilding. */
    fun shouldUpdate(): Boolean = synchronized(this) {
        motionSamples.size >= 10 && motionSamples.size % 5 == 0
    }

    fun rebuild(): AdaptiveThresholds {
        val new = synchronized(this) {
            AdaptiveThresholds.fromSamples(motionSamples.toList(), bufferSamples.toList())
        }
        currentThresholds = new
        prefs.edit().putString("thresholds", new.serialize()).apply()
        Log.i("AdaptiveThresholdCollector", "Updated thresholds: $new")
        return new
    }

    fun reset() {
        synchronized(this) {
            motionSamples.clear()
            bufferSamples.clear()
        }
        currentThresholds = AdaptiveThresholds.DEFAULT
        prefs.edit().remove("thresholds").apply()
    }
}
