from pydub import AudioSegment
from pydub.playback import play

class Voice:
    def __init__(self):
        print("Voice features initialized")
    def play(self,name):
        song=AudioSegment.from_mp3(name)
        play(song)

