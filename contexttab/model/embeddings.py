# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Dict, Literal
from torch import nn

from contexttab.data.tokenizer import Tokenizer


class DateEmbeddings(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.year_embeddings = nn.Embedding(52, hidden_size)
        self.month_embeddings = nn.Embedding(13, hidden_size)
        self.day_embeddings = nn.Embedding(32, hidden_size)
        self.weekday_embeddings = nn.Embedding(8, hidden_size)

    def forward(self, date_year_month_day_weekday):
        # date_year_month_day_weekday has shape (num_rows, num_cols, 4)
        year_embeds = self.year_embeddings(date_year_month_day_weekday[:, :, 0])
        month_embeds = self.month_embeddings(date_year_month_day_weekday[:, :, 1])
        day_embeds = self.day_embeddings(date_year_month_day_weekday[:, :, 2])
        weekday_embeds = self.weekday_embeddings(date_year_month_day_weekday[:, :, 3])

        return year_embeds + month_embeds + day_embeds + weekday_embeds


class CellEmbeddings(nn.Module):
    """
    Embedding module for self supervised learning.
    On the input side, it sums four contributions:
    - Numbers (itself coming from embedding three one-hot encoded values: sign, exponent, fraction)
    - Dates (itself coming from embedding four one-hot encoded values: year, month, day, weekday, plus one multi-hot encoded: holidays)
    - Column names (sentence embedding of the column name, adjusted to the hidden size)
    - (String) contents (sentence embedding of the column name, adjusted to the hidden size)
    For labels, it also computes and returns the sentence embedding of the string contents (not adjusted to the size).
    All string embeddings (column names, contents of both input and labels)
    """

    def __init__(self,
                 config,
                 regression_type: Literal['reg-as-classif', 'l2'] = 'reg-as-classif',
                 is_target_content_mapping: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        if regression_type == 'l2':
            self.number_embeddings = nn.Linear(1, config.hidden_size)
        else:
            self.number_embeddings = nn.Embedding(Tokenizer.QUANTILE_DIMENSION, config.hidden_size)

        self.regression_type = regression_type
        # for standard cross-entropy we use class indices otherwise we map content embeddings in target column with linear layer
        self.is_target_content_mapping = is_target_content_mapping

        if regression_type == 'l2':
            self.target_embedding_layer_reg = nn.Linear(1, config.hidden_size)
        else:
            self.target_embedding_layer_reg = nn.Embedding(Tokenizer.QUANTILE_DIMENSION, config.hidden_size)
        self.target_embedding_layer_classif = nn.Embedding(Tokenizer.QUANTILE_DIMENSION, config.hidden_size)

        self.date_embeddings = DateEmbeddings(config.hidden_size)

        self.column_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)
        self.content_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)
        if self.is_target_content_mapping:
            self.target_content_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def increase_by_one_and_map_negative_to_zero(self, tensor):
        """
        In dataset, "valid" labels are 0, 1, 2, ... and masked values are -100.
        We want to map them to 1, 2, 3, ... and 0.
        (Also, they might be float values, but we want to keep them as integers.)
        """
        tensor = tensor.int()
        return torch.where(tensor < 0, torch.zeros_like(tensor), tensor + 1)

    def forward(self, input_dict: Dict, is_regression: bool):
        num_rows, num_cols, _ = input_dict['text_embeddings'].shape
        is_classification = not is_regression

        if self.regression_type == 'l2':
            numbers_normalized = input_dict['number_normalized'].unsqueeze(-1).type(self.number_embeddings.weight.dtype)
            number_embeds = self.number_embeddings(numbers_normalized)
            number_embeds = torch.where(numbers_normalized <= -99, torch.zeros_like(number_embeds), number_embeds)
        else:
            number_perc_floor = input_dict['number_percentile_floor']
            number_embeds = torch.zeros((num_rows, num_cols, self.hidden_size),
                                        dtype=self.number_embeddings.weight.dtype,
                                        device=self.number_embeddings.weight.device)
            mask = number_perc_floor > -99
            number_embeds[mask] = self.number_embeddings(number_perc_floor[mask])

            next_perc = torch.minimum(number_perc_floor[mask] + 1, torch.tensor(Tokenizer.QUANTILE_DIMENSION - 1))
            number_embeds_plus_one = self.number_embeddings(next_perc)
            delta = input_dict['number_percentile_delta'][mask].type(number_embeds.dtype).unsqueeze(-1)
            number_embeds[mask] = number_embeds[mask] * (1 - delta) + number_embeds_plus_one * delta

        date_embeds = self.date_embeddings(input_dict['date_year_month_day_weekday'])  # (rows, cols, embed_dim)

        unsqueezed = input_dict['column_embeddings'].unsqueeze(0)  # (1, cols, embed_dim)
        column_embeds = self.column_remapping(unsqueezed.type(self.column_remapping.weight.dtype))

        target_text_embeddings = input_dict['text_embeddings'][:, -1].clone()
        # set to 0 for the case when is_target_content_mapping is False
        input_dict['text_embeddings'][:, -1] = 0

        content_embeds = self.content_remapping(input_dict['text_embeddings'].type(self.content_remapping.weight.dtype))
        if self.is_target_content_mapping:
            content_embeds[:, -1] = 0  # zero out to remove bias from `content_remapping` layer

        input_embeds = column_embeds + content_embeds + number_embeds + date_embeds

        if is_classification and self.is_target_content_mapping:
            # use the text embeddings, but pass them through a dedicated linear layer for the target column
            target_text_embeddings = target_text_embeddings.type(self.target_content_remapping.weight.dtype)
            target_content_embeds = self.target_content_remapping(target_text_embeddings)
            target_embeds = target_content_embeds.type(number_embeds.dtype)
        elif is_classification:
            target_values_classif = self.increase_by_one_and_map_negative_to_zero(input_dict['target'])
            target_embeds_classif = self.target_embedding_layer_classif(target_values_classif)  # (rows, embed_dim)
            target_embeds = target_embeds_classif.type(number_embeds.dtype)
        else:
            # regression
            if self.regression_type == 'l2':
                target_values_reg = input_dict['target'].unsqueeze(-1).type(
                    self.target_embedding_layer_reg.weight.dtype)
                target_embeds_reg = self.target_embedding_layer_reg(target_values_reg)
                target_embeds_reg = torch.where(target_values_reg <= -99, torch.zeros_like(target_embeds_reg),
                                                target_embeds_reg)
                target_embeds = target_embeds_reg.type(number_embeds.dtype)
            else:
                target_values_reg = self.increase_by_one_and_map_negative_to_zero(input_dict['target'])
                target_embeds_reg = self.target_embedding_layer_reg(target_values_reg)
                target_plus_one_embeds_reg = self.target_embedding_layer_reg(target_values_reg + 1)
                delta = input_dict['target_delta'].type(target_embeds_reg.dtype).unsqueeze(-1)
                target_embeds_reg = target_embeds_reg * (1 - delta) + target_plus_one_embeds_reg * delta
                target_embeds = target_embeds_reg

        padded_target_embeds = torch.zeros_like(number_embeds)
        padded_target_embeds[:, -1] = target_embeds
        input_embeds += padded_target_embeds

        input_embeds = self.layer_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)
        return input_embeds
