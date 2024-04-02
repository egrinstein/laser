from .openai import OpenAiCommandCreator
from .template import TemplateCommandCreator, random_template_command


class CommandCreator:
    def __init__(self, mode="template", use_corrector=True):
        if mode == "template":
            self.command_creator = TemplateCommandCreator(use_corrector=use_corrector)
        elif mode == "e2e":
            self.command_creator = OpenAiCommandCreator()
        else:
            raise ValueError(f"mode {mode} not supported")
        
    def __call__(self, target_caption, interferer_caption):
        return self.command_creator(target_caption, interferer_caption,
                                    return_type=True)
