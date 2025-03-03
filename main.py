import time
import random

from config import LANGUAGE, DELAY_SECONDS, VOICE_NAME
from speech.prompts import PROMPTS
from speech.speak import Speak

def main():
    """Runs the toddler learning session."""
    print(f"\nStarting in {LANGUAGE.upper()} mode. Delay: {DELAY_SECONDS} seconds.")
    print(f"Using voice: {VOICE_NAME}")
    print('\n')
    speak = Speak()

    while True:
        instruction = random.choice(PROMPTS[LANGUAGE])
        print(f"Prompt: {instruction}")
        speak.speak_instruction(instruction)
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    main()