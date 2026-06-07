package com.lsec.translator.domain.usecase

import com.lsec.translator.tts.TTSEngine
import javax.inject.Inject

/**
 * Caso de uso para síntesis de voz con ecuatorianismos.
 * Combina Google TTS con banco de audio personalizado.
 */
class SynthesizeSpeechUseCase @Inject constructor(
    private val ttsEngine: TTSEngine
) {
    /**
     * Sintetiza voz para el texto dado.
     * Usa audio pregrabado para ecuatorianismos o TTS estándar como fallback.
     * 
     * @param text Texto a sintetizar (puede ser ecuatorianismo o español estándar)
     */
    suspend operator fun invoke(text: String) {
        ttsEngine.speak(text)
    }
}
