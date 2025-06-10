# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from contexttab.constants import embedding_model_to_dimension_and_pooling


class SentenceEmbedder:
    def __init__(self, sentence_embedding_model_name, batch_size=256, device=None):
        super().__init__()
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.model = AutoModel.from_pretrained(sentence_embedding_model_name)
        self.embedding_dimension, self.pooling_method = embedding_model_to_dimension_and_pooling[
            sentence_embedding_model_name]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device).eval()
        if torch.cuda.is_available():
            self.model = self.model.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def pooling(self, model_output, attention_mask):
        """
        model_output[1] claims to contain the "pooled value", which would
        already have the correct shape, but actually the manual says _not_
        to use it, and instead do the either average or first token [CLS]
        pooling, depending on the model.
        Either case, returns a tensor of shape (batch_size, embedding_dim)
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        if self.pooling_method == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).type(
                token_embeddings.dtype)
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)
        assert self.pooling_method == 'cls'
        return token_embeddings[:, 0].type(token_embeddings.dtype)

    @torch.no_grad()
    def embed_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        results = []
        for start_idx in range(0, len(input_ids), self.batch_size):
            these_ids = input_ids[start_idx:start_idx + self.batch_size].to(self.device)
            this_mask = attention_mask[start_idx:start_idx + self.batch_size].to(self.device)
            model_output = self.model(these_ids, this_mask)
            results.append(self.pooling(model_output, this_mask))
        res = torch.concat(results)
        if res.dtype != self.model.dtype or self.dtype != torch.float16:
            res = res.type(torch.float16)
        return res

    def embed(self, texts: List[str]):
        if not len(texts):
            return []
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        embeddings = self.embed_sentences(encoded.input_ids, encoded.attention_mask)
        embeddings = embeddings.cpu().numpy()
        if self.dtype != torch.float16:
            embeddings = embeddings.astype('float16')
        return [embedding.tobytes() for embedding in embeddings]
