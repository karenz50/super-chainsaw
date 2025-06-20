from dotenv import load_dotenv
import os

load_dotenv()

import assemblyai as aai

aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")

config = aai.TranscriptionConfig(
    speaker_labels=True
)

transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe("data/Anthony_Filamor_plix5ba068ee-shift-20250618_144548.wav")

for utterance in transcript.utterances:
    print(f"{utterance.speaker}: {utterance.text}")