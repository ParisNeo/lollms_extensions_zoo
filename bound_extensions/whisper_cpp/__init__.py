from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
from lollms.utilities import PackageManager
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


class AudioRecorder:
    def __init__(self, filename, channels=1, sample_rate=44100, chunk_size=1024, silence_threshold=0.01, silence_duration=2):
        self.filename = filename
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        self.audio_stream = None
        self.audio_frames = []
        self.is_recording = False
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.last_sound_time = time.time()

    def start_recording(self):
        self.is_recording = True
        self.audio_stream = pyaudio.PyAudio().open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print("Recording started...")

        threading.Thread(target=self._record).start()

    def _record(self):
        while self.is_recording:
            data = self.audio_stream.read(self.chunk_size)
            self.audio_frames.append(data)

            # Check for silence
            rms = self._calculate_rms(data)
            if rms < self.silence_threshold:
                current_time = time.time()
                if current_time - self.last_sound_time >= self.silence_duration:
                    self.stop_recording()
            else:
                self.last_sound_time = time.time()

    def _calculate_rms(self, data):
        squared_sum = sum([sample ** 2 for sample in data])
        rms = (squared_sum / len(data)) ** 0.5
        return rms

    def stop_recording(self):
        self.is_recording = False
        self.audio_stream.stop_stream()
        self.audio_stream.close()

        audio = wave.open(self.filename, 'wb')
        audio.setnchannels(self.channels)
        audio.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        audio.setframerate(self.sample_rate)
        audio.writeframes(b''.join(self.audio_frames))
        audio.close()

        print(f"Recording saved to {self.filename}")


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