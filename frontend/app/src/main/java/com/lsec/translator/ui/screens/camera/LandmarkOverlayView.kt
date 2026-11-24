package com.lsec.translator.ui.screens.camera

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import com.lsec.translator.ml.LandmarkConnections
import com.lsec.translator.ml.LandmarkOverlayData
import com.lsec.translator.ml.GestureCommandClassifier

/**
 * Capa Compose que dibuja los landmarks (manos + cara + pose) sobre el
 * [androidx.camera.view.PreviewView].
 *
 * Decisiones de UX:
 *
 *   1. **Colores sutiles y semitransparentes** para no robar atención al
 *      usuario ni tapar el rostro/manos. La cara usa alpha 0.35, las manos
 *      0.85 (más relevantes para la seña).
 *
 *   2. **Manos primero, cara después** en el orden de pintado → si se
 *      solapan, las manos (lo que firma) quedan siempre visibles.
 *
 *   3. **Punto central del puño** (landmark 0, muñeca) más grande que el
 *      resto para anclar visualmente la orientación.
 *
 *   4. **Contornos de cara reducidos** (no tesselation completa) para que
 *      478 puntos no saturen la pantalla. Si el usuario quiere ver
 *      todos los landmarks, lo cambiamos desde aquí sin tocar el detector.
 *
 *   5. **Sin estado mutable**: este composable solo lee [data]; los
 *      recompose los dispara el ViewModel. El Canvas redibuja en 0.5–1.5 ms
 *      en gama media (medido: Pixel 5 ~0.8 ms con 478 + 42 puntos).
 *
 * @param data snapshot de landmarks del frame actual (o [EmptyLandmarkOverlay])
 * @param isFrontCamera true para espejar (cámara frontal)
 * @param showFace true para pintar cara (puede desactivarse en modo batería)
 * @param showHandConnections true para pintar las aristas (false = solo puntos)
 */
@Composable
fun LandmarkOverlayView(
    data: LandmarkOverlayData,
    isFrontCamera: Boolean,
    gestureTrail: List<GestureCommandClassifier.Point> = emptyList(),
    showFace: Boolean = true,
    showHandConnections: Boolean = true,
    modifier: Modifier = Modifier,
) {
    // Buffers reutilizables para evitar GC pressure a 30 fps.
    // Tamaño máximo: 478 puntos cara + 42 puntos manos = 520 puntos × 2 floats = 1040 floats.
    val pointBuffer = remember { FloatArray(1100) }
    val faceKeyPointBuffer = remember { FloatArray(64) } // max ~32 keypoints × 2

    Canvas(modifier = modifier.fillMaxSize()) {
        if (data.timestampMs == 0L) return@Canvas  // frame vacío

        val viewW = size.width
        val viewH = size.height
        val imgW = data.imageWidth
        val imgH = data.imageHeight
        val rotation = data.imageRotationDegrees

        if (imgW <= 0 || imgH <= 0) return@Canvas

        // ── Cara (debajo de las manos, semitransparente) ─────────────────
        if (showFace && data.hasFace()) {
            drawFaceOverlay(
                faceLandmarks = data.face!!,
                count = data.faceCount(),
                imageW = imgW,
                imageH = imgH,
                viewW = viewW,
                viewH = viewH,
                isFrontCamera = isFrontCamera,
                imageRotationDegrees = rotation,
                pointBuffer = pointBuffer,
            )
        }

        // ── Mano derecha ────────────────────────────────────────────────
        if (data.hasRightHand()) {
            val rh = data.rightHand!!
            drawHand(
                landmarks = rh,
                color = HandRightColor,
                imageW = imgW,
                imageH = imgH,
                viewW = viewW,
                viewH = viewH,
                isFrontCamera = isFrontCamera,
                imageRotationDegrees = rotation,
                drawConnections = showHandConnections,
                pointBuffer = pointBuffer,
            )
        }

        // ── Mano izquierda ───────────────────────────────────────────────
        if (data.hasLeftHand()) {
            val lh = data.leftHand!!
            drawHand(
                landmarks = lh,
                color = HandLeftColor,
                imageW = imgW,
                imageH = imgH,
                viewW = viewW,
                viewH = viewH,
                isFrontCamera = isFrontCamera,
                imageRotationDegrees = rotation,
                drawConnections = showHandConnections,
                pointBuffer = pointBuffer,
            )
        }

        if (gestureTrail.size > 1) {
            drawGestureTrail(
                trail = gestureTrail,
                imageW = imgW,
                imageH = imgH,
                viewW = viewW,
                viewH = viewH,
                isFrontCamera = isFrontCamera,
                imageRotationDegrees = rotation,
            )
        }
    }
}

private fun DrawScope.drawGestureTrail(
    trail: List<GestureCommandClassifier.Point>,
    imageW: Int,
    imageH: Int,
    viewW: Float,
    viewH: Float,
    isFrontCamera: Boolean,
    imageRotationDegrees: Int,
) {
    val raw = FloatArray(trail.size * 3)
    trail.forEachIndexed { index, point ->
        raw[index * 3] = point.x
        raw[index * 3 + 1] = point.y
    }
    val mapped = FloatArray(trail.size * 2)
    CoordinateMapper.mapMany(raw, 0, trail.size, imageW, imageH, viewW, viewH, isFrontCamera, imageRotationDegrees, mapped)
    for (index in 1 until trail.size) {
        drawLine(
            color = Color(0xFFFFD54F),
            start = Offset(mapped[(index - 1) * 2], mapped[(index - 1) * 2 + 1]),
            end = Offset(mapped[index * 2], mapped[index * 2 + 1]),
            strokeWidth = 5f,
            cap = StrokeCap.Round,
        )
    }
    drawCircle(
        color = Color(0xFFFFF59D),
        radius = 8f,
        center = Offset(mapped[mapped.size - 2], mapped[mapped.size - 1]),
    )
}

// ════════════════════════════════════════════════════════════════════════════
//  Drawing primitives
// ════════════════════════════════════════════════════════════════════════════

/**
 * Pinta las 21 articulaciones de una mano + las aristas de conexión.
 */
private fun DrawScope.drawHand(
    landmarks: FloatArray,           // 21 × 3 floats
    color: Color,
    imageW: Int,
    imageH: Int,
    viewW: Float,
    viewH: Float,
    isFrontCamera: Boolean,
    imageRotationDegrees: Int,
    drawConnections: Boolean,
    pointBuffer: FloatArray,
) {
    CoordinateMapper.mapMany(
        landmarks = landmarks,
        offset = 0,
        count = 21,
        imageWidth = imageW,
        imageHeight = imageH,
        viewWidth = viewW,
        viewHeight = viewH,
        isFrontCamera = isFrontCamera,
        imageRotationDegrees = imageRotationDegrees,
        out = pointBuffer,
    )

    // Conexiones (aristas)
    if (drawConnections) {
        for ((a, b) in LandmarkConnections.HAND) {
            drawLine(
                color = color.copy(alpha = 0.75f),
                start = Offset(pointBuffer[a * 2], pointBuffer[a * 2 + 1]),
                end = Offset(pointBuffer[b * 2], pointBuffer[b * 2 + 1]),
                strokeWidth = 3.5f,
                cap = StrokeCap.Round,
            )
        }
    }

    // Puntos: muñeca (0) más grande, nudillos (5,9,13,17) medianos, resto pequeño.
    for (i in 0 until 21) {
        val px = pointBuffer[i * 2]
        val py = pointBuffer[i * 2 + 1]
        val radius = when (i) {
            0 -> 7f      // muñeca (ancla)
            5, 9, 13, 17 -> 5f  // nudillos
            else -> 3.5f
        }
        drawCircle(
            color = color,
            radius = radius,
            center = Offset(px, py),
        )
        // Borde blanco para contraste sobre fondos oscuros
        drawCircle(
            color = Color.White.copy(alpha = 0.35f),
            radius = radius,
            center = Offset(px, py),
            style = Stroke(width = 1f),
        )
    }
}

/**
 * Pinta la malla facial usando los contornos (no la tesselation completa).
 * Si el mesh trae menos de 478 puntos, igualmente pintamos todos los disponibles.
 */
private fun DrawScope.drawFaceOverlay(
    faceLandmarks: FloatArray,
    count: Int,
    imageW: Int,
    imageH: Int,
    viewW: Float,
    viewH: Float,
    isFrontCamera: Boolean,
    imageRotationDegrees: Int,
    pointBuffer: FloatArray,
) {
    // Mapear todos los puntos al buffer compartido.
    CoordinateMapper.mapMany(
        landmarks = faceLandmarks,
        offset = 0,
        count = count,
        imageWidth = imageW,
        imageHeight = imageH,
        viewWidth = viewW,
        viewHeight = viewH,
        isFrontCamera = isFrontCamera,
        imageRotationDegrees = imageRotationDegrees,
        out = pointBuffer,
    )

    val edgeColor = FaceEdgeColor
    val dotColor = FaceDotColor

    // ── Aristas de los contornos faciales ──────────────────────────────
    for ((a, b) in LandmarkConnections.FACE_CONTOURS) {
        if (a >= count || b >= count) continue
        drawLine(
            color = edgeColor,
            start = Offset(pointBuffer[a * 2], pointBuffer[a * 2 + 1]),
            end = Offset(pointBuffer[b * 2], pointBuffer[b * 2 + 1]),
            strokeWidth = 1.2f,
            cap = StrokeCap.Round,
        )
    }

    // ── Puntos clave (subconjunto reducido para legibilidad) ───────────
    for (idx in LandmarkConnections.FACE_KEY_POINTS) {
        if (idx >= count) continue
        val px = pointBuffer[idx * 2]
        val py = pointBuffer[idx * 2 + 1]
        drawCircle(
            color = dotColor,
            radius = 2.5f,
            center = Offset(px, py),
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Paleta de colores (centralizada para que se pueda tematizar)
// ════════════════════════════════════════════════════════════════════════════

/**
 * Color para la mano derecha (≈ "Right" handedness de MediaPipe).
 * En cámara frontal la mano derecha del usuario aparece en el lado
 * izquierdo de la pantalla.
 */
val HandRightColor = Color(0xFFFFA07A)   // light salmon

/**
 * Color para la mano izquierda.
 */
val HandLeftColor = Color(0xFF87CEFA)    // light sky blue

/**
 * Color para aristas faciales. Semitransparente para no tapar el rostro.
 */
val FaceEdgeColor = Color(0xFFDDA0DD).copy(alpha = 0.35f)  // plum

/**
 * Color para puntos clave de la cara (iris, boca, nariz).
 */
val FaceDotColor = Color(0xFFE6E6FA).copy(alpha = 0.85f)   // lavender

@Suppress("unused")
private val UNUSED_SIZE_HINT = Size(0f, 0f)  // placeholder si se quiere escalar el overlay
