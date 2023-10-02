from lollms.extension import LOLLMSExtension
from lollms.helpers import ASCIIColors
from lollms.config import InstallOption, TypedConfig, BaseConfig, ConfigTemplate
import subprocess
extension_name="Whisper"

class Whisper(LOLLMSExtension):
    def __init__(self, app) -> None:
        template = ConfigTemplate([
                {"name":"active","type":"bool","value":False},
            ])
        config = BaseConfig.from_template(template)
        extension_config = TypedConfig(
            template,
            config
        )

        super().__init__("whisper", extension_config, app)


    def build_extension(self):
        return self
    
    def install(self):
        """
        Installation procedure (to be implemented)
        """
        super().install()

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