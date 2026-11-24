package com.lsec.translator.ml

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Reconocedor geométrico temporal para el MVP. No interpreta LSEC: reconoce
 * comandos de demostración a partir de una mano abierta y la muñeca normalizada
 * contra el rostro. No sustituye un modelo de LSC entrenado.
 * No conserva imágenes ni landmarks fuera de la ventana activa.
 */
class GestureCommandClassifier {
    enum class State { WAITING_FOR_SUBJECT, READY, CAPTURING, COOLDOWN }

    data class Point(val x: Float, val y: Float)
    data class Update(
        val state: State,
        val status: String,
        val trail: List<Point>,
        val result: SignClassificationResult? = null,
    )

    private data class Sample(
        val point: Point,
        val mouthDistance: Float,
        val isOpenHand: Boolean,
        val elapsedMs: Long,
    )

    private val samples = ArrayDeque<Sample>()
    private var stableFrames = 0
    private var state = State.WAITING_FOR_SUBJECT
    private var startedAtMs = 0L
    private var cooldownUntilMs = 0L
    private var missingFrames = 0

    fun process(data: LandmarkOverlayData, nowMs: Long = System.currentTimeMillis()): Update {
        if (nowMs < cooldownUntilMs) {
            state = State.COOLDOWN
            return update("Resultado mencionado. Prepárate para otro gesto.")
        }
        if (!data.hasFace() || (!data.hasLeftHand() && !data.hasRightHand())) {
            missingFrames++
            if (missingFrames > MAX_MISSING_FRAMES) reset()
            state = State.WAITING_FOR_SUBJECT
            val message = when {
                !data.hasFace() -> "Ubica tu rostro completo en cámara"
                else -> "Muestra una mano junto a tu rostro"
            }
            return update(message)
        }

        val face = data.face ?: return update("Ubica tu rostro completo en cámara")
        val hand = data.rightHand ?: data.leftHand ?: return update("Muestra una mano junto a tu rostro")
        val reference = faceReference(face) ?: run {
            reset(); return update("No se pudieron ubicar los puntos del rostro")
        }
        missingFrames = 0
        val wrist = Point(hand[0], hand[1])
        val relative = Point((wrist.x - reference.nose.x) / reference.scale, (wrist.y - reference.nose.y) / reference.scale)
        val mouthDistance = hypot((wrist.x - reference.mouth.x) / reference.scale, (wrist.y - reference.mouth.y) / reference.scale)
        val isOpenHand = isOpenHand(hand)

        if (state == State.WAITING_FOR_SUBJECT) {
            if (!isOpenHand) {
                stableFrames = 0
                return update("Abre la mano y mantenla junto al rostro")
            }
            stableFrames++
            if (stableFrames < STABLE_FRAMES_REQUIRED) return update("Mantén rostro y mano visibles (${stableFrames}/$STABLE_FRAMES_REQUIRED)")
            state = State.READY
            samples.clear()
            return update("Listo: realiza uno de los gestos básicos")
        }

        if (state == State.READY) {
            if (samples.isEmpty()) {
                samples.addLast(Sample(relative, mouthDistance, isOpenHand, nowMs))
                return update("Listo: mueve la mano para iniciar")
            }
            val initial = samples.first().point
            if (distance(initial, relative) < START_DISTANCE) return update("Listo: mueve la mano para iniciar")
            state = State.CAPTURING
            startedAtMs = nowMs
        }

        samples.addLast(Sample(relative, mouthDistance, isOpenHand, nowMs))
        while (samples.size > MAX_SAMPLES) samples.removeFirst()
        val elapsed = nowMs - startedAtMs
        val shouldFinish = samples.size >= MIN_SAMPLES && elapsed >= MIN_DURATION_MS &&
            (elapsed >= MAX_DURATION_MS || isStill())
        if (!shouldFinish) return update("Capturando gesto…")

        val result = classify()
        reset()
        if (result == null) {
            state = State.READY
            return update("Gesto no reconocido. Intenta un movimiento más claro")
        }
        cooldownUntilMs = nowMs + COOLDOWN_MS
        state = State.COOLDOWN
        return Update(state, "Detectado: ${result.signName}", emptyList(), result)
    }

    fun clear() = reset()

    private fun classify(): SignClassificationResult? {
        val s = samples.toList()
        if (s.size < MIN_SAMPLES) return null
        val durationMs = s.last().elapsedMs - s.first().elapsedMs
        val openHandRatio = s.count { it.isOpenHand }.toFloat() / s.size
        if (durationMs !in MIN_DURATION_MS..MAX_DURATION_MS || openHandRatio < MIN_OPEN_HAND_RATIO) {
            return null
        }
        val first = s.first().point
        val last = s.last().point
        val dx = last.x - first.x
        val dy = last.y - first.y
        val rangeX = s.maxOf { it.point.x } - s.minOf { it.point.x }
        val rangeY = s.maxOf { it.point.y } - s.minOf { it.point.y }
        val directionChangesX = directionChanges(s.map { it.point.x })
        val directionChangesY = directionChanges(s.map { it.point.y })
        val nearMouthAtStart = s.first().mouthDistance < MOUTH_DISTANCE
        val totalAngle = accumulatedAngle(s.map { it.point })
        val pathLength = s.zipWithNext { a, b -> distance(a.point, b.point) }.sum()
        val netDistance = distance(first, last)
        val confidence = quality(max(rangeX, rangeY))

        val label = when {
            // Círculo frente al rostro/pecho.
            abs(totalAngle) > 5.0f && rangeX > 0.55f && rangeY > 0.55f && pathLength > 2.3f -> "por favor"
            // Dos oscilaciones laterales: arriba de ojos = adiós; a la altura de cara = hola.
            directionChangesX >= 2 && rangeX > 0.9f && rangeX > rangeY * 1.5f -> {
                val averageY = s.map { it.point.y }.sum() / s.size
                if (averageY < -0.75f) "adiós" else "hola"
            }
            // Mano parte de la boca y baja al centro.
            nearMouthAtStart && dy > 1.0f && abs(dx) < abs(dy) * 0.55f -> "perdón"
            // Mano parte de la boca y se aleja en el plano.
            nearMouthAtStart && netDistance > 1.1f && abs(dx) > 0.7f && abs(dy) > 0.7f -> "gracias"
            // Subida amplia hasta sobrepasar frente.
            dy < -1.5f && last.y < -1.1f && abs(dx) < abs(dy) * 0.55f -> "ayuda"
            // Oscilación vertical abajo-arriba.
            directionChangesY >= 1 && rangeY > 1.0f && rangeY > rangeX * 1.5f -> "sí"
            // Swipe lateral único.
            abs(dx) > 1.2f && abs(dx) > abs(dy) * 2f && pathLength / netDistance < 1.25f -> "no"
            else -> return null
        }
        return SignClassificationResult(label, label, confidence, MVP_LABELS.indexOf(label))
    }

    private fun isStill(): Boolean {
        if (samples.size < 7) return false
        val tail = samples.takeLast(5).map { it.point }
        return distance(tail.first(), tail.last()) < 0.12f
    }

    private fun update(status: String) = Update(state, status, samples.map { it.point })

    private fun reset() {
        samples.clear(); stableFrames = 0; missingFrames = 0; startedAtMs = 0L
        if (state != State.COOLDOWN) state = State.WAITING_FOR_SUBJECT
    }

    private data class FaceReference(val nose: Point, val mouth: Point, val scale: Float)
    private fun faceReference(face: FloatArray): FaceReference? {
        fun p(index: Int): Point? = if (face.size >= (index + 1) * 3) Point(face[index * 3], face[index * 3 + 1]) else null
        val nose = p(1) ?: return null
        val mouth = p(13) ?: return null
        val leftEye = p(33) ?: return null
        val rightEye = p(263) ?: return null
        val scale = distance(leftEye, rightEye)
        return if (scale > 0.01f) FaceReference(nose, mouth, scale) else null
    }

    private fun directionChanges(values: List<Float>): Int {
        var previous = 0
        var changes = 0
        for (i in 1 until values.size) {
            val delta = values[i] - values[i - 1]
            if (abs(delta) < 0.025f) continue
            val sign = if (delta > 0) 1 else -1
            if (previous != 0 && sign != previous) changes++
            previous = sign
        }
        return changes
    }

    private fun accumulatedAngle(points: List<Point>): Float {
        if (points.size < 3) return 0f
        var total = 0f
        for (i in 2 until points.size) {
            val a = atan2(points[i - 1].y - points[i - 2].y, points[i - 1].x - points[i - 2].x)
            val b = atan2(points[i].y - points[i - 1].y, points[i].x - points[i - 1].x)
            var d = b - a
            while (d > Math.PI) d -= (2 * Math.PI).toFloat()
            while (d < -Math.PI) d += (2 * Math.PI).toFloat()
            total += d
        }
        return total
    }

    private fun distance(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

    private fun isOpenHand(hand: FloatArray): Boolean {
        if (hand.size < 63) return false
        fun point(index: Int) = Point(hand[index * 3], hand[index * 3 + 1])
        val wrist = point(0)
        val tips = intArrayOf(4, 8, 12, 16, 20)
        val bases = intArrayOf(2, 5, 9, 13, 17)
        val extended = tips.indices.count { index ->
            distance(wrist, point(tips[index])) > distance(wrist, point(bases[index])) * OPEN_FINGER_RATIO
        }
        return extended >= MIN_EXTENDED_FINGERS
    }

    private fun quality(amplitude: Float) = min(0.98f, max(0.70f, 0.65f + amplitude * 0.2f))

    companion object {
        val MVP_LABELS = listOf("hola", "adiós", "sí", "no", "ayuda", "gracias", "por favor", "perdón")
        private const val STABLE_FRAMES_REQUIRED = 5
        private const val MAX_MISSING_FRAMES = 3
        private const val MIN_SAMPLES = 9
        const val MAX_SAMPLES = 54
        private const val START_DISTANCE = 0.20f
        private const val MIN_DURATION_MS = 250L
        private const val MAX_DURATION_MS = 1_800L
        private const val COOLDOWN_MS = 1_000L
        private const val MOUTH_DISTANCE = 1.2f
        private const val OPEN_FINGER_RATIO = 1.35f
        private const val MIN_EXTENDED_FINGERS = 3
        private const val MIN_OPEN_HAND_RATIO = 0.8f
    }
}
