import threading
import queue
import pyttsx3
import time

class SpeechFlow:
    def __init__(self, cooldown_seconds=3):
        self._queue = queue.Queue()
        self._engine = pyttsx3.init()
        self._spoken_recently = {}
        self._cooldown = cooldown_seconds

        # Start background TTS thread
        self._thread = threading.Thread(target=self.speech_worker, daemon=True)
        self._thread.start()

    def speech_worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break

            label = item
            now = time.time()
            last_spoken = self._spoken_recently.get(label,0)
            if now - last_spoken >= self._cooldown:
                self._engine.say(label)
                self._engine.runAndWait()
                self._spoken_recently[label]= now
            self._queue.task_done()

    def speak(self, label):
        self._queue.put(label)

    def shutdown(self):
        self._queue.put(None)
        self._thread.join()
