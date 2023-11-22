from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
import subprocess
from pathlib import Path
extension_name="Bark"

import sys
import os

sys.path.append(os.getcwd())
pth = Path(__file__).parent/"bark_core"
sys.path.append(str(pth))

class Bark(LOLLMSExtension):
    def __init__(self, app,
                    installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY) -> None:
        template = ConfigTemplate([
                {"name":"active","type":"bool","value":False},
            ])
        config = BaseConfig.from_template(template)
        extension_config = TypedConfig(
            template,
            config
        )
        super().__init__("bark", Path(__file__).parent, extension_config, app, installation_option=installation_option)


    def build_extension(self):
     
        import torch
        import torchaudio

        from tortoise.api import MODELS_DIR, TextToSpeech
        from tortoise.utils.audio import get_voices, load_voices, load_audio
        from tortoise.utils.text import split_and_recombine_text

        # download and load all models
        preload_models()
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
    text="""
♪ Yo, listen up, I got a story to tell,
'Bout a tool called lollms, it's a rap rebel,
It's a web interface, user-friendly and slick,
Helping you with tasks, it's the ultimate pick ♪

♪ You can choose your binding, model, and personality,
Enhance your writing, coding, and data clarity,
From emails to essays, it's got you covered,
With light and dark mode, your preferences discovered ♪

♪ Search, organize, generate, it's all in there,
Images, music, answers, it's beyond compare,
Integrated with GitHub, easy access at hand,
With ratings and discussions, it's a genius plan ♪

♪ But let's talk about ethics, the implications of AI,
Lollms encourages reflection, don't be shy,
Weigh the pros and cons, the impact on society,
As we dive into the world of this technology ♪

♪ And it's open source, anyone can contribute,
Making it better, that's the attribute,
Developed by ParisNeo, giving back to the community,
A tool that's free, enhancing unity ♪

♪ So if you're looking for a tool that's top-notch,
Lollms is here, ready to rock,
With ongoing development and a supportive crew,
It's the ultimate choice, that's true ♪

♪ So check out the documentation, get started today,
Unleash your creativity, let your words play,
Lollms, the rapper's best friend, a skilled companion,
Bringing the art of rap to a new dimension ♪
    
    """
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav
    from tqdm import tqdm
    import numpy as np
    # SAMPLE_RATE = 44100
    # Split the text into paragraphs
    paragraphs = text.split("\n\n")

    # Preload models for faster audio generation
    preload_models()

    # Create an empty list to store the audio arrays
    audio_arrays = []
    voice_preset = "v2/en_speaker_6"
    # Generate audio for each paragraph
    i=0
    for paragraph in tqdm(paragraphs, desc="Generating audio"):
        audio_array = generate_audio(paragraph, history_prompt=str(voice_preset), silent=True)
        audio_arrays.append(audio_array)
        write_wav(Path.home()/"audio_out_chunk_{i}.wav", SAMPLE_RATE, audio_array)
        i += 1
    # Concatenate the audio arrays
    concatenated_audio = np.concatenate(audio_arrays)
    # save audio to disk
    write_wav(Path.home()/"audio_out.wav", SAMPLE_RATE, concatenated_audio)
