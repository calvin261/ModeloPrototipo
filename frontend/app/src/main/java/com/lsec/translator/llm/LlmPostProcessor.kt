package com.lsec.translator.llm

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import com.lsec.translator.data.preferences.UserPreferences
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Contrato minimo para cualquier post-procesador LLM opcional.
 *
 * Es un REFRASEADOR DE ORACIONES, NO un clasificador. Recibe una secuencia
 * de glosas LSC (palabras sueltas producidas por el clasificador + GRU) y
 * devuelve una oracion natural en espanol. Si devuelve la entrada casi
 * sin cambios, sigue siendo valido (no es un fallo).
 *
 * Mantenlo minimo: una sola operacion suspendida + un par de flags. Asi
 * se puede cambiar el backend (llama.cpp, MLC, ORT-GenAI) sin tocar
 * el resto de la app.
 */
interface LlmPostProcessor {
    /**
     * Rephrasa la secuencia de glosas en una oracion en espanol.
     *
     * @param signSequence glosas en orden cronologico. Lista vacia = no-op.
     * @return oracion en espanol. Si el LLM no esta listo o falla,
     *         devuelve la concatenacion `joinToString(" ")` como fallback.
     */
    suspend fun rephrase(signSequence: List<String>): String

    /** True si el modelo esta cargado en memoria y listo para inferir. */
    val isReady: Boolean

    /**
     * True si el LLM debe participar en el pipeline. Falso por defecto
     * (feature opt-in, ver [UserPreferences.llmEnabled]). Tambien es
     * forzado a falso en dispositivos con <6 GB de RAM, independientemente
     * de la preferencia del usuario.
     */
    val isEnabled: Boolean

    /** Cambia el flag `isEnabled` en runtime. No toca la preferencia persistida. */
    fun setEnabled(enabled: Boolean)
}

/**
 * Callback de progreso de carga del modelo. Se invoca en el hilo de
 * carga (no en main). El primer valor es 0f, el ultimo es 1f. Errores
 * se reportan con [LlmLoadProgress.onError].
 */
interface LlmLoadProgress {
    /** @param fraction 0f..1f, no monotonico garantizado (puede saltar atras al validar). */
    fun onProgress(fraction: Float, stage: String)
    fun onLoaded()
    fun onError(throwable: Throwable)
}

/**
 * Implementacion concreta basada en llama.cpp via JNI.
 *
 * **Por que existe esta clase:** el clasificador (MLP) produce glosas
 * palabra-por-palabra; el usuario querria ver una oracion natural
 * ("Hola, ¿como estas?") en vez de la lista cruda ("hola, como, estas").
 * Phi-3-mini INT4 en GGUF hace ese rephrase en espanol a 2-4 tok/s
 * en el Infinix GT30 Pro.
 *
 * **No es un clasificador.** No se entrena, no se reentrena, no
 * reemplaza al MLP ni al GRU. Solo embellece la salida.
 */
@Singleton
class Phi3LlamaCppPostProcessor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val userPreferences: UserPreferences,
) : LlmPostProcessor {

    private val mutex = Mutex()
    private val modelFile: File = File(context.filesDir, "models/phi3-mini-int4.gguf")

    private val _isReady = MutableStateFlow(false)
    val isReadyState: StateFlow<Boolean> = _isReady.asStateFlow()

    @Volatile
    private var runtimeEnabled: Boolean = false

    /**
     * Habilitacion efectiva: preferencia persistida AND dispositivo capaz
     * (>= 6 GB RAM). Se re-evalua en [setEnabled] y al inicio.
     */
    override val isEnabled: Boolean
        get() = runtimeEnabled && hasEnoughRam()

    override val isReady: Boolean
        get() = _isReady.value

    init {
        // El modelo SIEMPRE arranca deshabilitado. La preferencia persistida
        // se consulta luego (async) para no bloquear la inicializacion.
        runtimeEnabled = false
    }

    override fun setEnabled(enabled: Boolean) {
        runtimeEnabled = enabled && hasEnoughRam()
        Log.i(TAG, "isEnabled -> $runtimeEnabled (requested=$enabled, ramOK=${hasEnoughRam()})")
    }

    /**
     * Rephrasa la secuencia. Si no esta listo, devuelve la concatenacion
     * cruda (fallback degradado pero funcional). Si falla la inferencia,
     * log + fallback.
     */
    override suspend fun rephrase(signSequence: List<String>): String {
        if (signSequence.isEmpty()) return ""
        if (!isEnabled) return signSequence.joinToString(" ")
        if (!isReady) {
            // Trigger lazy load; devolvemos el fallback para no bloquear
            ensureLoaded()
            return signSequence.joinToString(" ")
        }

        val prompt = "Seyas: " + signSequence.joinToString(", ") + " -> Oracion:"
        return mutex.withLock {
            try {
                withContext(Dispatchers.IO) {
                    Phi3LlamaCppBridge.complete(
                        systemPrompt = SYSTEM_PROMPT,
                        userPrompt = prompt,
                        maxTokens = 64,
                        temperature = 0.3f,
                        topP = 0.9f,
                    ).trim().ifEmpty { signSequence.joinToString(" ") }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Inference failed, returning raw fallback", e)
                signSequence.joinToString(" ")
            }
        }
    }

    /**
     * Carga perezosa del modelo. Llamar explicitamente desde la UI para
     * mostrar progreso; [rephrase] lo llama automaticamente en background
     * si se olvido.
     */
    suspend fun ensureLoaded(progress: LlmLoadProgress? = null) {
        if (_isReady.value) {
            progress?.onLoaded()
            return
        }
        if (!modelFile.exists()) {
            val err = IllegalStateException(
                "Phi-3 GGUF not found at ${modelFile.absolutePath}. " +
                "Call ModelDownloadManager.downloadPhi3Gguf() first."
            )
            progress?.onError(err)
            throw err
        }
        withContext(Dispatchers.IO) {
            try {
                progress?.onProgress(0.05f, "Inicializando llama.cpp")
                Phi3LlamaCppBridge.init(
                    modelPath = modelFile.absolutePath,
                    nCtx = 2048,
                    nThreads = 4,
                    useMmap = true,
                )
                progress?.onProgress(0.95f, "Calentando modelo")
                // Warm-up de 1 token para que el primer rephrase real no tarde 5s.
                Phi3LlamaCppBridge.complete(SYSTEM_PROMPT, "test", 1, 0.0f, 0.0f)
                _isReady.value = true
                progress?.onLoaded()
                Log.i(TAG, "Phi-3 model loaded: ${modelFile.length() / 1024 / 1024} MB")
            } catch (e: Throwable) {
                progress?.onError(e)
                throw e
            }
        }
    }

    /** Libera el modelo de la RAM. Llamar en onStop si se quiere. */
    fun release() {
        try {
            Phi3LlamaCppBridge.release()
        } catch (e: Throwable) {
            Log.w(TAG, "release() failed", e)
        }
        _isReady.value = false
    }

    /**
     * Chequea RAM total del dispositivo. 6 GB es el piso; en menos,
     * el LLM se desactiva automaticamente (OOM en 8 GB con todo
     * lo demas cargado).
     */
    private fun hasEnoughRam(): Boolean {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            ?: return false
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val totalGb = memInfo.totalMem.toDouble() / 1024.0 / 1024.0 / 1024.0
        return totalGb >= MIN_RAM_GB
    }

    companion object {
        private const val TAG = "Phi3PostProcessor"

        /**
         * Prompt del sistema. Importante: dice "convierte" + "manten el
         * significado" para que Phi-3 NO invente contenido. La salida
         * tiene que ser <= maxTokens (64) y en espanol.
         */
        const val SYSTEM_PROMPT: String =
            "Eres un asistente que convierte secuencias de seyas LSC a " +
            "oraciones naturales en espanol. Manten el significado original. " +
            "Se conciso."

        /** 6 GB piso. 8 GB en el GT30 Pro apenas alcanza. */
        const val MIN_RAM_GB: Double = 6.0

        /** Ventana para el auto-trigger: 3+ glosas en este tiempo. */
        const val AUTO_TRIGGER_WINDOW_MS: Long = 6_000L

        /** Minimo de glosas para auto-trigger. */
        const val AUTO_TRIGGER_MIN_SIGNS: Int = 3
    }
}
