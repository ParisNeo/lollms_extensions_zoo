from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
from lollms.utilities import PackageManager
import subprocess
from pathlib import Path

if not PackageManager.check_package_installed("tortoise"):
    PackageManager.install_package("tortoise-tts")
    from tortoise import utils, api
else:
    from tortoise import utils, api

extension_name="TortoiseTTS"

import sys
import os

class TortoiseTTS(LOLLMSExtension):
    def __init__(self, app,
                    installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY) -> None:
        template = ConfigTemplate([
                {"name":"sample_path","type":"str","value":"","help":"The path to a folder containing samples to clone"},
                {"name":"active","type":"bool","value":False},
            ])
        config = BaseConfig.from_template(template)
        extension_config = TypedConfig(
            template,
            config
        )
        super().__init__("tortoise_tts", Path(__file__).parent, extension_config, app, installation_option=installation_option)


    def build_extension(self):
        reference_clips = [utils.audio.load_audio(p, 22050) for p in Path(self.config.sample_path).iterdir()]
        tts = api.TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
        pcm_audio = tts.tts_with_preset("your text here", voice_samples=reference_clips, preset='fast')
        return self
    
    def install(self):
        """
        Installation procedure (to be implemented)
        """
        super().install()
        parent_dir = self.script_path
        subprocess.run(["pip", "install", "--upgrade", "pyaudio"])
        subprocess.run(["pip", "install", "--upgrade", "tortoise-tts"])

    def pre_gen(self, previous_prompt:str, prompt:str):
        return previous_prompt, prompt

    def in_gen(self, chunk:str)->str:
        return chunk

    def post_gen(self, ai_output:str):
        from bark import SAMPLE_RATE, generate_audio, preload_models
        import pyaudio
        os.environ["SUNO_OFFLOAD_CPU"] = "True"
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"
        # Preload models for faster audio generation
        preload_models()

        audio_array = generate_audio(ai_output)
        # save audio to disk
        # write_wav(self.app.lollms_paths.personal_outputs_path/"audio.wav", SAMPLE_RATE, audio_array)
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Create an audio stream
        stream = p.open(format=pyaudio.paFloat32,  # You can adjust the format if needed
                        channels=1,               # Mono audio
                        rate=SAMPLE_RATE,         # Sample rate (samples per second)
                        output=True)

        # Play the audio
        stream.start_stream()
        stream.write(audio_array.tobytes())
        stream.stop_stream()

        # Close the audio stream and PyAudio
        stream.close()
        p.terminate()

    def get_ui():
        """
        Get user interface of the extension
        """
        return "<p>This is a ui extension template</p>"
    


if __name__=="__main__":
    from lollmsvectordb.text_document_loader import TextDocumentsLoader
    text=GenericDataLoader.read_file(Path(__file__).parent/"examples/rap2.txt")
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav
    from tqdm import tqdm
    import numpy as np
    # SAMPLE_RATE = 44100
    # Split the text into paragraphs
    paragraphs = text.split(".")

    # Preload models for faster audio generation
    preload_models()

    # Create an empty list to store the audio arrays
    audio_arrays = []
    voice_preset = "v2/en_speaker_2"
    # Generate audio for each paragraph
    i=0
    for paragraph in tqdm(paragraphs, desc="Generating audio"):
        audio_array = generate_audio(paragraph+'.', history_prompt=str(voice_preset), silent=True)
        audio_arrays.append(audio_array)
        write_wav(Path.home()/f"audio_out_chunk_{i}.wav", SAMPLE_RATE, audio_array)
        i += 1
    # Concatenate the audio arrays
    concatenated_audio = np.concatenate(audio_arrays)
    # save audio to disk
    write_wav(Path.home()/"audio_out.wav", SAMPLE_RATE, concatenated_audio)
