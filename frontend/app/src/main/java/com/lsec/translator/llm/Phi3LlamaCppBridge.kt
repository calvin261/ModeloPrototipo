package com.lsec.translator.llm

/**
 * Bridge JNI hacia la libreria nativa `libllama.so` (o como se llame
 * tras compilar `examples/llama.android` de llama.cpp).
 *
 * Esta clase es INTENCIONALMENTE MINIMA. Mantenerla asi:
 *  - Cero estado mutable.
 *  - Cero logica de negocio.
 *  - Cero corrutinas (los nativos son bloqueantes; envolver con
 *    `withContext(Dispatchers.IO)` desde Kotlin).
 *  - Tipos de dato portables: `String`, `Int`, `Float`, `Boolean`.
 *
 * Asi, cuando los bindings JNI cambien (que van a cambiar), el
 * unico archivo a tocar es este. No hay pruebas unitarias
 * posibles (son nativas) y eso esta bien.
 *
 * ## Como llenar los `external fun`
 *
 * 1. Clonar llama.cpp: `git clone https://github.com/ggerganov/llama.cpp`
 * 2. Mirar `examples/llama.android/` — ya trae un MainActivity con JNI.
 *    Copiar los `.cpp`/`.h` a `frontend/app/src/main/cpp/`.
 * 3. En el `CMakeLists.txt` de la app, anadir la lib de llama.cpp
 *    como subdirectory o `add_subdirectory(llama.cpp/build)`.
 * 4. Implementar las funciones en C++ envueltas en `extern "C" JNIEXPORT`.
 * 5. El nombre Java de cada metodo es Fijo:
 *    `com.lsec.translator.llm.Phi3LlamaCppBridge.<nombreMetodo>`.
 *    C++ usa `Java_com_lsec_translator_llm_Phi3LlamaCppBridge_<nombre>`.
 *
 * Pasos detallados en `frontend/SETUP_LLM.md`.
 */
object Phi3LlamaCppBridge {

    /**
     * Inicializa el contexto de llama.cpp y carga el modelo GGUF en RAM.
     *
     * **Native method** — implementar en C++ con `llama_model_load_from_file`
     * y `llama_new_context_with_model`. Threading: SINGLETON, no llamar
     * desde multiples hilos simultaneamente (usar el Mutex del wrapper).
     *
     * @param modelPath ruta absoluta al `.gguf`. Ya validada por Kotlin.
     * @param nCtx tamano de ventana (2048 es seguro para Phi-3-mini-4k).
     * @param nThreads numero de threads CPU (4 = sweet spot para
     *                 Dimensity 8350, mas = thermal throttling).
     * @param useMmap true: usa mmap en vez de copiar a RAM. En Android
     *                con archivos en `filesDir` funciona bien y reduce
     *                el peak RAM ~1 GB.
     * @return handle opaco (long) o 0 si falla. Kotlin lo ignora; C++ lo
     *         guarda en una variable global.
     */
    @JvmStatic
    external fun init(
        modelPath: String,
        nCtx: Int,
        nThreads: Int,
        useMmap: Boolean,
    ): Long

    /**
     * Inferencia sincrona (prompt -> respuesta completa).
     *
     * **Native method** — implementar con `llama_tokenize`,
     * `llama_decode` en bucle + sampler. Para 2-4 tok/s no vale la pena
     * streaming con callback; el rephrase genera <30 tokens de salida.
     *
     * @param systemPrompt prompt de sistema (cache separado del user prompt).
     * @param userPrompt prompt del usuario (ya formateado por Kotlin).
     * @param maxTokens limite duro de tokens de salida (64 recomendado).
     * @param temperature 0.0f = determinista, 0.3f = ligera variacion,
     *                    >0.7f = creativa (NO recomendado para rephrase).
     * @param topP nucleus sampling. 0.9f es el default sensato.
     * @return texto generado, sin el prompt de entrada. Vacio si falla.
     */
    @JvmStatic
    external fun complete(
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
    ): String

    /**
     * Libera el modelo y el contexto de llama.cpp.
     *
     * **Native method** — llamar a `llama_free` y `llama_model_free`.
     * Despues de esto, hay que volver a llamar [init] antes de [complete].
     */
    @JvmStatic
    external fun release()

    /**
     * Intenta cargar la libreria nativa en cuanto se toca la clase.
     * Si falla, todos los `external fun` lanzaran `UnsatisfiedLinkError`
     * — atrapado por el wrapper que cae al fallback de concatenacion.
     */
    init {
        runCatching { System.loadLibrary("llama") }
            .onFailure { android.util.Log.w("Phi3Bridge", "libllama.so no cargada: ${it.message}") }
    }
}
