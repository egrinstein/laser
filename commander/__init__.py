from .openai import OpenAiCommandCreator
from .template import TemplateCommandCreator, random_template_command


class CommandCreator:
    def __init__(self, mode="template", use_corrector=True):
        if mode == "template":
            self.command_creator = TemplateCommandCreator(use_corrector=use_corrector)
        elif mode == "e2e":
            self.command_creator = OpenAiCommandCreator()
        elif mode == "caption":
            self.command_creator = None
        else:
            raise ValueError(f"mode {mode} not supported")
        
        self.mode = mode

    def __call__(self, target_caption, interferer_caption):
        if self.mode == "caption":
            return target_caption, "positive"
        else:    
            return self.command_creator(target_caption, interferer_caption,
                                        return_type=True)
