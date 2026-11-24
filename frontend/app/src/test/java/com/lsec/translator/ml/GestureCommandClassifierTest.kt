package com.lsec.translator.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GestureCommandClassifierTest {
    @Test
    fun rejectsFramesWithoutFace() {
        val classifier = GestureCommandClassifier()
        val update = classifier.process(LandmarkOverlayData(null, hand(), null, 640, 480, 1))
        assertNull(update.result)
        assertEquals(GestureCommandClassifier.State.WAITING_FOR_SUBJECT, update.state)
    }

    @Test
    fun recognizesSingleHorizontalSwipeAsNo() {
        val classifier = GestureCommandClassifier()
        var now = 0L
        repeat(6) { now += 33; classifier.process(frame(0.50f, 0.50f, now), now) }
        // Base, desplazamiento y cinco frames quietos para cerrar la ventana.
        now += 33; classifier.process(frame(0.50f, 0.50f, now), now)
        now += 33; classifier.process(frame(0.78f, 0.50f, now), now)
        var result: SignClassificationResult? = null
        repeat(8) {
            now += 33
            val update = classifier.process(frame(0.80f, 0.50f, now), now)
            result = update.result ?: result
        }
        assertEquals("no", result?.signName)
    }

    @Test
    fun requiresAnOpenHandBeforeCapture() {
        val classifier = GestureCommandClassifier()
        var now = 0L
        repeat(8) {
            now += 33
            classifier.process(LandmarkOverlayData(null, closedHand(), face(), 640, 480, now), now)
        }
        val update = classifier.process(LandmarkOverlayData(null, closedHand(), face(), 640, 480, now + 33), now + 33)
        assertEquals(GestureCommandClassifier.State.WAITING_FOR_SUBJECT, update.state)
        assertNull(update.result)
    }

    private fun frame(wristX: Float, wristY: Float, timestamp: Long): LandmarkOverlayData =
        LandmarkOverlayData(null, hand(wristX, wristY), face(), 640, 480, timestamp)

    private fun hand(x: Float = 0.5f, y: Float = 0.5f): FloatArray = FloatArray(63).also {
        fun set(index: Int, px: Float, py: Float) { it[index * 3] = px; it[index * 3 + 1] = py }
        set(0, x, y)
        set(2, x - 0.04f, y - 0.04f); set(4, x - 0.16f, y - 0.12f)
        set(5, x - 0.05f, y - 0.05f); set(8, x - 0.07f, y - 0.25f)
        set(9, x, y - 0.05f); set(12, x, y - 0.29f)
        set(13, x + 0.05f, y - 0.05f); set(16, x + 0.07f, y - 0.25f)
        set(17, x + 0.09f, y - 0.03f); set(20, x + 0.14f, y - 0.19f)
    }

    private fun closedHand(): FloatArray = FloatArray(63).also {
        it[0] = 0.5f; it[1] = 0.5f
    }

    private fun face(): FloatArray = FloatArray(478 * 3).also {
        fun set(index: Int, x: Float, y: Float) { it[index * 3] = x; it[index * 3 + 1] = y }
        set(1, 0.5f, 0.45f) // nose
        set(13, 0.5f, 0.56f) // mouth
        set(33, 0.4f, 0.4f); set(263, 0.6f, 0.4f) // eye reference
    }
}
