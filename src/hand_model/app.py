import cv2
import numpy as np

from .classifier import StaticSignRecognizer
from .collect import HAND_CONNECTIONS, _draw_landmarks_on_frame
from .config import AppConfig
from .landmarks import LandmarkExtractor
from .tts import TTSEngine


class SignLanguageApp:
    """Real-time sign language recognition application.

    Controls:
        SPACE     — Speak accumulated text
        C         — Clear text buffer
        BACKSPACE — Delete last character
        M         — Toggle between static/dynamic mode
        Q / ESC   — Quit
    """

    def __init__(self, cfg: AppConfig = None):
        self.cfg = cfg or AppConfig()
        self.extractor = LandmarkExtractor(
            model_path=self.cfg.landmark_model,
            num_hands=1,
            min_detection_confidence=0.5,
        )

        self.mode = self.cfg.mode
        self.recognizer = None
        self.seq_recognizer = None

        if self.mode == "static":
            self._load_static_model()
        else:
            self._load_sequence_model()

        self.tts = TTSEngine(rate=self.cfg.tts_rate, lang=self.cfg.tts_lang)

        # State
        self.text_buffer = ""
        self.last_prediction = ""
        self.prediction_counter = 0
        self.speaking = False

        # Sequence recording buffer for dynamic mode
        self.sequence_buffer = []
        self.max_seq_len = 60

    def _load_static_model(self):
        self.recognizer = StaticSignRecognizer(
            self.cfg.classifier_model, self.extractor
        )

    def _load_sequence_model(self):
        from .sequence_model import DynamicSignRecognizer
        self.seq_recognizer = DynamicSignRecognizer(
            self.cfg.sequence_model, self.extractor
        )

    def run(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)

        print("Sign Language App started. Press Q or ESC to quit.")
        print(f"Mode: {self.mode}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Draw landmarks and get prediction
            raw = self.extractor.extract_raw(frame)
            result = None

            if raw is not None:
                _draw_landmarks_on_frame(frame, raw)

                if self.mode == "static" and self.recognizer is not None:
                    from .landmarks import normalize_landmarks
                    landmarks = normalize_landmarks(raw).astype(np.float32)
                    result = self.recognizer.predict(frame)
                elif self.mode == "dynamic" and self.seq_recognizer is not None:
                    result = self.seq_recognizer.predict(frame)

            if result is not None:
                label, confidence = result
                if confidence > self.cfg.confidence_threshold:
                    self._handle_prediction(label)
                self._draw_prediction(frame, label, confidence)
            else:
                # Reset counter when no hand is detected
                self.prediction_counter = 0

            self._draw_text_buffer(frame)
            self._draw_controls(frame)
            cv2.imshow("LSEC Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == ord(" "):  # Space = speak
                self.tts.speak_async(self.text_buffer)
            elif key == ord("c"):  # Clear
                self.text_buffer = ""
            elif key == 8:  # Backspace
                self.text_buffer = self.text_buffer[:-1]
            elif key == ord("m"):  # Toggle mode
                self._toggle_mode()

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _handle_prediction(self, label: str):
        """Stabilize predictions: require N consecutive same predictions."""
        if label == self.last_prediction:
            self.prediction_counter += 1
        else:
            self.last_prediction = label
            self.prediction_counter = 1

        if self.prediction_counter == self.cfg.confirm_frames:
            if self.mode == "static":
                if label == "SPACE":
                    self.text_buffer += " "
                else:
                    self.text_buffer += label
            else:
                self.text_buffer += label + " "
            # Reset to avoid repeating the same character
            self.prediction_counter = 0
            self.last_prediction = ""

    def _toggle_mode(self):
        if self.mode == "static":
            self.mode = "dynamic"
            if self.seq_recognizer is None:
                try:
                    self._load_sequence_model()
                except Exception as e:
                    print(f"Could not load sequence model: {e}")
                    self.mode = "static"
                    return
        else:
            self.mode = "static"
            if self.recognizer is None:
                self._load_static_model()
        print(f"Switched to mode: {self.mode}")

    def _draw_prediction(self, frame, label: str, confidence: float):
        h, w = frame.shape[:2]
        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        color = (0, 255, 0) if confidence > self.cfg.confidence_threshold else (0, 165, 255)
        cv2.putText(
            frame,
            f"Sign: {label} ({confidence:.0%})",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Stability progress bar
        progress = min(
            self.prediction_counter / self.cfg.confirm_frames, 1.0
        )
        bar_width = int(progress * 200)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 50), (210, 70), (200, 200, 200), 1)

    def _draw_text_buffer(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        display_text = self.text_buffer[-50:] if len(self.text_buffer) > 50 else self.text_buffer
        cv2.putText(
            frame,
            f"Text: {display_text}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _draw_controls(self, frame):
        h, w = frame.shape[:2]
        mode_text = f"Mode: {self.mode.upper()}"
        controls = "[SPACE]Speak [C]lear [BS]Delete [M]ode [Q]uit"
        cv2.putText(
            frame, mode_text,
            (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )
        cv2.putText(
            frame, controls,
            (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )
