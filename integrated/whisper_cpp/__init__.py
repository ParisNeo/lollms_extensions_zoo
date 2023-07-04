from lollms.extension import Extension

class Whisper_cpp(Extension):
    def __init__(self, metadata_file_path: str, app) -> None:
        super().__init__(metadata_file_path, app)