from gtts import gTTS
import pyttsx3 
import glob 

languge = 'en'
text="meow  meow meow meowww meow meow meow meowwwww "

speech = gTTS(text=text, lang=languge,slow=False, tld="com.au")
speech.save("textToSpeech.mp3")




