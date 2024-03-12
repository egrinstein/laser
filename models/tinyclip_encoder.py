import requests
import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from ontology.caption_to_ontology import caption_to_random_command


class TinyCLIP_Encoder(nn.Module):
    def __init__(
        self,
        sampling_rate=32000,
        caption_to_command = True,
    ):
        super().__init__()
        self.device = "cpu"
        self.precision = "fp32"
        self.sampling_rate = sampling_rate

        self.model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
        self.processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.encoder_type = 'CLAP'

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
            text_input = self.processor(text=batch, return_tensors="pt", padding=True)
            embed = self.model.get_text_features(**text_input)
        if double_batch:
            embed = embed[0].unsqueeze(0)
        
        return embed.detach()


    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if self.caption_to_command and text:
            text = [caption_to_random_command(t) for t in text]

        embed = self._get_text_embed(text)
   
        return embed.float()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
