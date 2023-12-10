from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
from lollms.utilities import PackageManager
from lollms.media import AudioRecorder
import subprocess
from pathlib import Path
import time
if not PackageManager.check_package_installed("faster_whisper"):
    PackageManager.install_package("faster-whisper")
    from faster_whisper import WhisperModel
else:
    from faster_whisper import WhisperModel


if not PackageManager.check_package_installed("pyaudio"):
    PackageManager.install_package("pyaudio")
    PackageManager.install_package("wave")
    import pyaudio
    import wave
else:
    import pyaudio
    import wave
import threading

extension_name="Whisper"


class Whisper(LOLLMSExtension):
    def __init__(self, app,
                    installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY) -> None:
        template = ConfigTemplate([
                {"name":"active","type":"bool","value":False},
                {"name":"model","type":"str","value":"large-v3","options":["large-v3"]},
                {"name":"device","type":"str","value":"auto","options":["auto","cpu","cuda"]},
                {"name":"compute_type","type":"str","value":"float16","options":["float16"]},
            ])
        config = BaseConfig.from_template(template)
        extension_config = TypedConfig(
            template,
            config
        )

        super().__init__("whisper", Path(__file__).parent, extension_config, app, installation_option=installation_option)
        self.output_folder:Path = self.app.lollms_paths.personal_outputs_path/self.name
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.audioRecorder = AudioRecorder(self.output_folder/"chunk.wav")

    def build_extension(self):
        # Run on GPU with FP16
        self.model = WhisperModel(self.extension_config.model, device=self.extension_config.device, compute_type=self.extension_config.compute_type)
        return self
    
    def install(self):
        """
        Installation procedure (to be implemented)
        """
        super().install()

    def start(self):
        self.audioRecorder.start_recording()

    def start(self):
        self.audioRecorder.start_recording()

    def pre_gen(self, previous_prompt:str, prompt:str):
        return previous_prompt, prompt

    def in_gen(self, chunk:str)->str:
        return chunk

    def post_gen(self, ai_output:str):
        pass

    def get_ui():
        """
        Get user interface of the extension
        """
        return "<p>This is a ui extension template</p>"