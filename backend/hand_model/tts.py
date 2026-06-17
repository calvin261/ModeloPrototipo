import threading

import pyttsx3


class TTSEngine:
    """Offline text-to-speech engine using pyttsx3 with Spanish voice.

    On Windows, Spanish voices may need to be installed via:
        Settings -> Time & Language -> Speech -> Add voices
    """

    def __init__(self, rate: int = 150, lang: str = "es"):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.spanish_voice_found = False

        # Try to set a Spanish voice (check SAPI5 and OneCore tokens)
        voices = self.engine.getProperty("voices")
        for voice in voices:
            voice_id_lower = voice.id.lower()
            name_lower = voice.name.lower()
            if any(kw in name_lower for kw in ("spanish", "español")) or \
               any(kw in voice_id_lower for kw in ("es-", "es_", "spanish")):
                self.engine.setProperty("voice", voice.id)
                self.spanish_voice_found = True
                break

        if not self.spanish_voice_found:
            print(
                "WARNING: No Spanish TTS voice found. Using default voice.\n"
                "  To install Spanish voices on Windows:\n"
                "  Settings -> Time & Language -> Speech -> Add voices -> Espanol"
            )

    def speak(self, text: str):
        """Speak text synchronously (blocks until done)."""
        if text.strip():
            self.engine.say(text)
            self.engine.runAndWait()

    def speak_async(self, text: str):
        """Speak text in a background thread (non-blocking)."""
        if text.strip():
            thread = threading.Thread(
                target=self.speak, args=(text,), daemon=True
            )
            thread.start()
