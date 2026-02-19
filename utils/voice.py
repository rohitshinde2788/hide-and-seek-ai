import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty('rate', 160)

# GLOBAL LOCK (VERY IMPORTANT)
voice_lock = threading.Lock()

def speak(text):
    if voice_lock.locked():
        return

    with voice_lock:
        engine.say(text)
        engine.runAndWait()
