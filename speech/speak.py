import pyttsx3

from config import VOICE_NAME


class Speak:

    @staticmethod
    def get_available_voices():
        """Lists available voices in the system."""
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        return {v.name.lower(): v.id for v in voices}
    def speak_instruction(self, instruction):
        """Speaks out the given instruction."""
        engine = pyttsx3.init()

        available_voices = self.get_available_voices()
        selected_voice = available_voices.get(VOICE_NAME.lower())
        if selected_voice:
            engine.setProperty("voice", selected_voice)
        else:
            print(f"⚠️ Voice '{VOICE_NAME}' not found! Using default.")
            engine.setProperty("voice", 'anna')

        engine.say(instruction)
        engine.runAndWait()



