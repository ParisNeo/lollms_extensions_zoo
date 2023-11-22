from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
import subprocess
extension_name="Bark"

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

        super().__init__("bark", extension_config, app, installation_option=installation_option)


    def build_extension(self):
        from bark_core import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav

        # download and load all models
        preload_models()
        return self
    
    def install(self):
        """
        Installation procedure (to be implemented)
        """
        super().install()
        parent_dir = self.script_path
        subfolder_path = parent_dir / "bark_core"
        repo_url="https://github.com/ParisNeo/bark.git"
        requirements_file = self.script_path/"bark_core"/ "requirements.txt"
        subprocess.run(["git", "clone", repo_url, str(subfolder_path)])
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])    
        subprocess.run(["pip", "install", "--upgrade", "pyaudio"])

    def pre_gen(self, previous_prompt:str, prompt:str):
        return previous_prompt, prompt

    def in_gen(self, chunk:str)->str:
        return chunk

    def post_gen(self, ai_output:str):
        from bark_core import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav
        import pyaudio
        SAMPLE_RATE = 44100

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