package com.lsec.translator.tts

import android.content.Context
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume

/**
 * Motor de síntesis de voz híbrido para LSEC Translator.
 * Combina Google TTS con banco de audio pregrabado para ecuatorianismos.
 */
@Singleton
class TTSEngine @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private var mediaPlayer: MediaPlayer? = null
    
    // Mapeo de ecuatorianismos a archivos de audio pregrabados
    private val ecuadorianAudioMap = mapOf(
        "¡Qué más!" to "audio_que_mas",
        "Pana" to "audio_pana",
        "Chévere" to "audio_chevere",
        "Full" to "audio_full",
        "Bacán" to "audio_bacan",
        "¡Qué fue, cómo vas?" to "audio_que_fue",
        "No cacho" to "audio_no_cacho",
        "Ahorita" to "audio_ahorita",
        "Al rato" to "audio_al_rato",
        "La vieja" to "audio_la_vieja",
        "El viejo" to "audio_el_viejo",
        "El guagua" to "audio_guagua",
        "El ñaño" to "audio_nanio",
        "¡Chao, nos vemos!" to "audio_chao",
        "¡Dale, vamos!" to "audio_dale_vamos",
        "¡Ojo!" to "audio_ojo",
        "Gracias, pana" to "audio_gracias_pana",
        "Encantado, pana" to "audio_encantado_pana"
    )
    
    // Configuración de voz
    private var speechRate = 1.0f
    private var pitch = 1.0f
    
    init {
        initializeTTS()
    }
    
    /**
     * Inicializa el motor TTS de Google.
     */
    private fun initializeTTS() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Configurar español latinoamericano (lo más cercano a Ecuador)
                val locale = Locale("es", "EC")
                val result = tts?.setLanguage(locale)
                
                if (result == TextToSpeech.LANG_MISSING_DATA || 
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    // Fallback a español genérico
                    tts?.setLanguage(Locale("es", "ES"))
                }
                
                // Configurar parámetros de voz
                tts?.setSpeechRate(speechRate)
                tts?.setPitch(pitch)
                
                isInitialized = true
            }
        }
    }
    
    /**
     * Sintetiza voz para el texto dado.
     * Usa audio pregrabado para ecuatorianismos o TTS como fallback.
     */
    suspend fun speak(text: String) = withContext(Dispatchers.Main) {
        if (text.isBlank()) return@withContext
        
        // Intentar usar audio pregrabado primero
        if (playCustomAudio(text)) {
            return@withContext
        }
        
        // Fallback a Google TTS
        speakWithTTS(text)
    }
    
    /**
     * Intenta reproducir audio pregrabado para ecuatorianismos.
     */
    private suspend fun playCustomAudio(text: String): Boolean {
        val audioFileName = ecuadorianAudioMap[text] ?: return false
        
        return try {
            val resourceId = context.resources.getIdentifier(
                audioFileName,
                "raw",
                context.packageName
            )
            
            if (resourceId == 0) return false
            
            suspendCancellableCoroutine { continuation ->
                mediaPlayer?.release()
                mediaPlayer = MediaPlayer.create(context, resourceId)
                
                mediaPlayer?.setOnCompletionListener {
                    it.release()
                    mediaPlayer = null
                    continuation.resume(true)
                }
                
                mediaPlayer?.setOnErrorListener { _, _, _ ->
                    mediaPlayer?.release()
                    mediaPlayer = null
                    continuation.resume(false)
                    true
                }
                
                mediaPlayer?.start()
                
                continuation.invokeOnCancellation {
                    mediaPlayer?.release()
                    mediaPlayer = null
                }
            }
        } catch (e: Exception) {
            android.util.Log.w("TTSEngine", "Failed to play custom audio for: $text", e)
            false
        }
    }
    
    /**
     * Sintetiza voz usando Google TTS.
     */
    private suspend fun speakWithTTS(text: String) {
        if (!isInitialized || tts == null) {
            android.util.Log.w("TTSEngine", "TTS not initialized")
            return
        }
        
        suspendCancellableCoroutine { continuation ->
            val utteranceId = UUID.randomUUID().toString()
            
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                
                override fun onDone(utteranceId: String?) {
                    continuation.resume(Unit)
                }
                
                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    continuation.resume(Unit)
                }
                
                override fun onError(utteranceId: String?, errorCode: Int) {
                    continuation.resume(Unit)
                }
            })
            
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
            
            continuation.invokeOnCancellation {
                tts?.stop()
            }
        }
    }
    
    /**
     * Detiene cualquier reproducción en curso.
     */
    fun stop() {
        tts?.stop()
        mediaPlayer?.stop()
        mediaPlayer?.release()
        mediaPlayer = null
    }
    
    /**
     * Configura la velocidad del habla.
     * @param rate Velocidad (0.5 = lento, 1.0 = normal, 2.0 = rápido)
     */
    fun setSpeechRate(rate: Float) {
        speechRate = rate.coerceIn(0.5f, 2.0f)
        tts?.setSpeechRate(speechRate)
    }
    
    /**
     * Configura el tono de voz.
     * @param pitchValue Tono (0.5 = grave, 1.0 = normal, 2.0 = agudo)
     */
    fun setPitch(pitchValue: Float) {
        pitch = pitchValue.coerceIn(0.5f, 2.0f)
        tts?.setPitch(pitch)
    }
    
    /**
     * Verifica si el TTS está listo para usar.
     */
    fun isReady(): Boolean = isInitialized
    
    /**
     * Libera recursos del motor TTS.
     */
    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
