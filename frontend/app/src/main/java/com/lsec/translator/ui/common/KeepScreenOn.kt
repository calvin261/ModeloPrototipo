package com.lsec.translator.ui.common

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.platform.LocalView

/**
 * Mantiene la pantalla encendida mientras este composable está en composición.
 * Se desactiva automáticamente al salir (DisposableEffect).
 *
 * No requiere permiso WAKE_LOCK; usa WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
 * que solo aplica mientras esta ventana está en primer plano. El usuario puede
 * seguir bloqueando manualmente desde el botón de encendido.
 *
 * Uso: envolver el contenido de pantallas donde el usuario mira fijamente
 * (cámara en vivo, entrenamiento, captura de datos, calibración).
 */
@Composable
fun KeepScreenOn() {
    val view = LocalView.current
    DisposableEffect(view) {
        view.keepScreenOn = true
        onDispose {
            view.keepScreenOn = false
        }
    }
}