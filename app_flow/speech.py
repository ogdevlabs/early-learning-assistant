import logging
import time

import pyttsx3


class SpeechManager:
    def __init__(self, cooldown_seconds=3):
        self.engine = pyttsx3.init()
        self.spoken_recently = {}
        self.cooldown = cooldown_seconds

    def speak(self, label: str):
        current_time = time.time()
        last_spoken = self.spoken_recently.get(label, 0)

        if current_time - last_spoken >= self.cooldown:
            try:
                self.engine.say(label)
                self.engine.runAndWait()
                self.spoken_recently[label] = current_time
            except Exception as e:
                logging.error(f"Speech error: {e}")