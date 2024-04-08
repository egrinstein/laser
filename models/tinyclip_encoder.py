import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPTextModelWithProjection # CLIPProcessor, CLIPModel 


class TinyCLIP_Encoder(nn.Module):
    def __init__(
        self,
        sampling_rate=32000,
        caption_to_command = True,
        force_cpu=False
    ):
        super().__init__()

        self.device = torch.device('cpu') if force_cpu else _get_device()
        self.precision = 'fp32'
        self.sampling_rate = sampling_rate

        self.model = CLIPTextModelWithProjection.from_pretrained(
            'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M').to(self.device)

        # Load the processor, which is responsible for text and image preprocessing
        # We only use it for text preprocessing here, which converts a string to a tensor of tokens
        # self.processor = CLIPProcessor.from_pretrained('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M')
        self.processor = AutoTokenizer.from_pretrained('wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M')

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        self.caption_to_command = caption_to_command

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def _get_text_embed(self, batch):
        double_batch = False
        if len(batch) == 1:
            batch = batch * 2
            double_batch = True
        with torch.no_grad():
            text_input = self.tokenizer(batch)
            #embed = self.model.get_text_features(**text_input)
            embed = self.model(**text_input).text_embeds
        if double_batch:
            embed = embed[0].unsqueeze(0)
        
        return embed.detach()


    def __call__(self, text):
        embed = self._get_text_embed(text)
   
        return embed.float()

    def tokenizer(self, text_batch):
        text_input = self.processor(text=text_batch, return_tensors='pt', padding=True)

        # Move text input to the device
        text_input = {k: v.to(self.device) for k, v in text_input.items()}
        return text_input


def _get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device
