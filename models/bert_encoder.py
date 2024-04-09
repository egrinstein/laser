import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings('ignore')
# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'prajjwal1/bert-mini': (BertModel, BertTokenizer),
}

class BertEncoder(nn.Module):
    def __init__(self, dim=256, add_start_token=True):
        super().__init__()
        self.base_model = 'prajjwal1/bert-mini'
        self.dropout = 0.1

        self.tokenizer = MODELS[self.base_model][1].from_pretrained(self.base_model)

        self.bert_layer =  MODELS[self.base_model][0].from_pretrained(self.base_model,
                                                    add_pooling_layer=False,
                                                    hidden_dropout_prob=self.dropout,
                                                    attention_probs_dropout_prob=self.dropout,
                                                    output_hidden_states=True)
        
        self.linear_layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.add_start_token = add_start_token

    def tokenize(self, caption):
        if self.add_start_token:
            caption = ['[CLS] ' + cap for cap in caption]
        tokenized = self.tokenizer(caption, add_special_tokens=False, padding=True, return_tensors='pt')
        input_ids = tokenized['input_ids']
        attns_mask = tokenized['attention_mask']

        input_ids = input_ids.to(self.device)
        attns_mask = attns_mask.to(self.device)
        return input_ids, attns_mask

    def forward(self, caption):
        input_ids, attns_mask = self.tokenize(caption)
        output = self.bert_layer(input_ids=input_ids, attention_mask=attns_mask)[0]
        cls_embed = output[:, 0, :]
        text_embed = self.linear_layer(cls_embed)

        return text_embed, output  # text_embed: (batch, hidden_size)