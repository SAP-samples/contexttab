import re
from abc import ABC
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torcheval.metrics.functional import r2_score
from transformers.activations import gelu
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaConfig

from contexttab.constants import ModelSize
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
                 add_target_embedding_to_input: bool,
                 regression_type: Literal['reg-as-classif', 'l2', 'l2-with-target-binning', 'clustering',
                                          'clustering-cosine'] = 'reg-as-classif',
                 is_target_content_mapping: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        if regression_type == 'l2':
            self.number_embeddings = nn.Linear(1, config.hidden_size)
        else:
            self.number_embeddings = nn.Embedding(Tokenizer.QUANTILE_DIMENSION, config.hidden_size)

        self.add_target_embedding_to_input = add_target_embedding_to_input
        self.regression_type = regression_type
        # for standard cross-entropy we use class indices otherwise we map content embeddings in target column with linear layer
        self.is_target_content_mapping = is_target_content_mapping

        if add_target_embedding_to_input:
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

        if self.add_target_embedding_to_input:
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


class AbstractModel(nn.Module, ABC, ModuleUtilsMixin):

    def __init__(self,
                 model_size: ModelSize,
                 regression_type: Literal['reg-as-classif', 'l2', 'l2-with-target-binning', 'clustering',
                                          'clustering-cosine'] = 'reg-as-classif',
                 classification_type: Literal['cross-entropy', 'clustering', 'triplet-l2', 'triplet-cosine',
                                              'clustering-cosine'] = 'cross-entropy'):
        super().__init__()
        self.model_size = model_size.value
        num_layers, hidden_size = model_size.value
        self.config = RobertaConfig(num_hidden_layers=num_layers,
                                    hidden_size=hidden_size,
                                    intermediate_size=hidden_size * 4,
                                    num_attention_heads=hidden_size // 64,
                                    layer_norm_eps=1e-5,
                                    type_vocab_size=1,
                                    hidden_dropout_prob=0.1)
        self.regression_type = regression_type
        self.classification_type = classification_type
        if classification_type == 'triplet-cosine':
            self.cosine_triplet_loss = CosineTripletLoss(margin=0.5)
        max_number_of_labels = Tokenizer.QUANTILE_DIMENSION

        if self.classification_type in ['clustering', 'clustering-cosine']:
            # adjacency matrix prediction head
            self.cluster_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.cluster_out_dim = self.config.hidden_size
            self.cluster_output_head = nn.Linear(self.config.hidden_size, self.cluster_out_dim)
        elif self.classification_type in ['triplet-l2', 'triplet-cosine']:
            # triplet head
            self.dense_classif = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.output_head_classif = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        else:
            # standard class prediction head
            self.dense_classif = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.output_head_classif = nn.Linear(self.config.hidden_size, max_number_of_labels)

        self.dense_reg = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        if self.regression_type in ['l2', 'l2-with-target-binning']:
            self.output_head_reg = nn.Linear(self.config.hidden_size, 1)
        elif self.regression_type in ['clustering', 'clustering-cosine']:
            self.cluster_out_dim = self.config.hidden_size
            self.output_head_reg = nn.Linear(self.config.hidden_size, self.cluster_out_dim)
        else:
            self.output_head_reg = nn.Linear(self.config.hidden_size, max_number_of_labels)

    def build_context_attention_mask(self, data, device):
        """
        Builds a context attention mask of shape (num_rows, num_rows)
        Everything can attend to context but query can only be attended by itself.
        This has the output shape of get_extended_attention_mask, which means:
        - 0 on the diagonal, as well as in any (i, j) position with j in context and any i;
        - -inf elsewhere
        """
        assert data['target'].ndim == 1, 'Expected target to be a 1D tensor, got shape: {}'.format(data['target'].shape)
        num_rows = int(data['target'].numel())
        context_attention_mask = torch.eye(num_rows)

        context_rows = data['target'] > -99
        context_attention_mask[:, context_rows] = 1

        # The following is equivalent to calling self.get_extended_attention_mask which however is a bit weird (needs batch,
        # needs a useless second parameter) so we avoid calling it
        context_attention_mask = context_attention_mask.to(device)
        return (1.0 - context_attention_mask) * torch.finfo(context_attention_mask.dtype).min

    def compute_classif_loss_and_metric(self, logits, labels, train_target):
        # logits has shape (num_rows, max_number_of_labels)
        # labels, is_test_mask, and train_target have shape (num_rows,)
        # labels is unmasked and has type int; is_test_mask should be used to mask labels.
        # train_target is the target value for the training set (used for dummy predictions).
        # the "real" target is round(target + target_delta)
        labels = labels.long()
        is_test_mask = train_target <= -99
        loss_labels = torch.where(is_test_mask, labels, -100 * torch.ones_like(labels))
        loss_classif = nn.functional.cross_entropy(logits.float(), loss_labels, ignore_index=-100).float()

        prediction = logits.argmax(dim=-1)[is_test_mask]
        accuracy = torch.mean((prediction == labels[is_test_mask]).float())

        dummy_prediction = train_target[train_target > -99].mode().values
        dummy_accuracy = torch.mean((dummy_prediction == labels[is_test_mask]).float())
        metric_classif = torch.clip((accuracy - dummy_accuracy) / (1 - dummy_accuracy + 1e-5), 0, 1)
        return loss_classif, metric_classif

    @staticmethod
    def memory_efficient_cosine_similarity(x, batch_size=1000):
        """
        Computes cosine similarity between all pairs of vectors in x efficiently.

        Args:
            x: Tensor of shape (n, d)
            batch_size: Number of vectors to process at once

        Returns:
            Cosine similarity matrix of shape (n, n)
        """
        n = x.size(0)

        x_normalized = F.normalize(x, p=2, dim=1)
        result = torch.zeros((n, n), device=x.device)
        for i in range(0, n, batch_size):
            batch_x = x_normalized[i:]
            similarity = torch.mm(batch_x, x_normalized.t())
            result[i:] = similarity

        return result  # shape (n, n)

    def forward_clustering_head(self,
                                encoder_outputs: torch.Tensor,
                                out_layer_1,
                                out_layer_2,
                                use_cosine_similarity=False):
        cluster_out = out_layer_1(encoder_outputs)
        cluster_out = gelu(cluster_out)
        cluster_out = out_layer_2(cluster_out)

        # cluster_out has shape (num_rows, cluster_out_dim)

        if use_cosine_similarity:
            # Don't use torch.nn.functional.cosine_similarity because it uses a huge amount of
            # memory via broadcasting
            out_clustering = self.memory_efficient_cosine_similarity(cluster_out)
            # values in [-1, 1]
        else:
            out_clustering = torch.matmul(cluster_out, cluster_out.T)
            # any real value; will be squashed to [0, 1] with sigmoid.
        # shape: (num_rows, num_rows)
        # values in [-1, 1]
        return out_clustering

    def compute_triplet_loss_and_metric(self, logits, labels, train_target, anchors_idx, positives_idx, negatives_idx):
        if anchors_idx is not None:
            is_test_mask = train_target <= -99
            query_logits = logits[is_test_mask]
            context_logits = logits[~is_test_mask]

            query_labels = labels[is_test_mask]
            context_labels = labels[~is_test_mask]

            anchors_tensor = query_logits[anchors_idx]
            positives_tensor = context_logits[positives_idx]
            negatives_tensor = context_logits[negatives_idx]

            if self.classification_type == 'triplet-cosine':
                triplet_loss = self.cosine_triplet_loss(anchors_tensor, positives_tensor, negatives_tensor)

                query_norm = F.normalize(query_logits, p=2, dim=1)
                context_norm = F.normalize(context_logits, p=2, dim=1)

                cosine_similarities = torch.mm(query_norm, context_norm.T)
                query_pred_context_idx = torch.argmax(cosine_similarities, dim=1)
            else:
                triplet_loss = torch.nn.functional.triplet_margin_loss(anchors_tensor,
                                                                       positives_tensor,
                                                                       negatives_tensor,
                                                                       margin=1.0,
                                                                       p=2.0)
                triplet_loss = torch.clip(triplet_loss, 0, 1.0)

                distances = torch.cdist(query_logits, context_logits, p=2.0)  # [num_queries, num_contexts]
                # get the index of the closest context for each query
                query_pred_context_idx = torch.argmin(distances, dim=1)

            # get the label idx corresponding to each closest context for each query
            query_predicted_classes = context_labels[query_pred_context_idx]
            # calculate accuracy
            metric_classif = (query_predicted_classes == query_labels).float().mean()
        else:
            triplet_loss = torch.tensor(0.0)
            metric_classif = torch.tensor(0.0)

        return triplet_loss, metric_classif

    @staticmethod
    def compute_clustering_output_loss_and_metric(logits,
                                                  labels,
                                                  train_target,
                                                  is_mask_out_context=False,
                                                  is_cosine_similarity=False):
        # logits has shape (num_rows, num_rows)
        # might be either cosine similarity or scalar product (for clustering-cosine and clustering respectively)
        # labels, is_test_mask, and train_target have shape (num_rows, )
        adjacency_matrices = (labels.unsqueeze(-1) == labels.unsqueeze(-2)).to(dtype=logits.dtype)

        if is_cosine_similarity:
            # cosine_similarity is between -1 and 1. We morally clip it at 0:
            # in this way, we push the same class vectors to be as aligned as possible (cosine similarity = 1)
            # and different classes to be orthogonal (cosine similarity = 0) or opposite (cosine similarity < 0)
            # but we don't push it to _always_ be opposite (cosine similarity = -1) because if there are more
            # than 2 classes that's impossible to achieve
            # We also clip at 1.0 because very rarely it seems to crash otherwise...
            loss_cluster = torch.nn.functional.binary_cross_entropy(torch.clip(logits, min=0.0, max=1.0),
                                                                    adjacency_matrices,
                                                                    reduction='none')
        else:
            loss_cluster = torch.nn.functional.binary_cross_entropy_with_logits(logits,
                                                                                adjacency_matrices,
                                                                                reduction='none')

        if is_mask_out_context:
            # only leave loss for context x query off-diagonal elements
            is_context = train_target > -99  # shape (num_rows, )
            off_diagonal_mask = (is_context.unsqueeze(-1) & ~is_context.unsqueeze(-2)).int()
            loss_cluster = torch.mul(loss_cluster, off_diagonal_mask)
            denominator = torch.sum(off_diagonal_mask).clip(min=1)
        else:
            denominator = loss_cluster.numel()
        loss_cluster = torch.sum(loss_cluster) / denominator
        loss_cluster *= 3  # arbitrarily increase clustering loss to keep its magnitude closer to the regression loss

        # This might need to be checked: since the model is free to predict any cosine similarity <= 0
        # when it things two rows are in different classes, it's not clear that we shouldn't clip the
        # values to 0 before averaging. However this only has an effect if one is positive and the other
        # is negative, which should happen rather rarely.
        if is_cosine_similarity:
            out_clustering = logits
        else:
            out_clustering = torch.sigmoid(logits)
        out_clustering = (out_clustering + out_clustering.transpose(-2, -1)) / 2

        metric_cluster = (out_clustering > 0.5).int()
        mask = (metric_cluster == 1) | (adjacency_matrices == 1)
        metric_cluster = (metric_cluster[mask] == adjacency_matrices[mask]).sum() / mask.sum()
        return out_clustering, loss_cluster, metric_cluster

    def compute_regression_output_loss_and_metric(self, logits, labels, train_target):
        if self.regression_type == 'reg-as-classif':
            loss_reg, metric_reg = self.compute_classif_loss_and_metric(logits, labels, train_target)
        else:
            logits = logits.squeeze(-1)  # shape (num_rows, )
            test_mask = train_target <= -99
            masked_labels = labels[test_mask]
            masked_logits = logits[test_mask]
            masked_logits = torch.nan_to_num(masked_logits)
            loss_reg = nn.functional.mse_loss(masked_logits.float(), masked_labels.float()).float()
            loss_reg = torch.clip(loss_reg, 0, 10)

            try:
                # although it is r2 on the normalized data
                metric_reg = r2_score(masked_logits, masked_labels)
                metric_reg = torch.nan_to_num(metric_reg)
                metric_reg = torch.clip(metric_reg, -1, 1)
            except:
                metric_reg = torch.tensor(0).float()
                print('error calculating r2 score in the training loop')
        return logits, loss_reg, metric_reg

    def forward_heads(self,
                      encoder_outputs: torch.Tensor,
                      is_regression: bool,
                      labels: Optional[torch.Tensor] = None,
                      target: Optional[torch.Tensor] = None,
                      target_delta: Optional[torch.Tensor] = None,
                      anchors_idx: Optional[torch.Tensor] = None,
                      positives_idx: Optional[torch.Tensor] = None,
                      negatives_idx: Optional[torch.Tensor] = None):
        """
        Last part of the "forward" method.
        It takes the encoder outputs (one token per row) and applies the heads and losses (if labels are provided).
        """
        is_classification = not is_regression

        if is_classification:
            if self.classification_type in ['clustering', 'clustering-cosine']:
                use_cosine_similarity = self.classification_type == 'clustering-cosine'
                out = self.forward_clustering_head(encoder_outputs,
                                                   self.cluster_dense,
                                                   self.cluster_output_head,
                                                   use_cosine_similarity=use_cosine_similarity)
            else:
                out = self.dense_classif(encoder_outputs)
                out = gelu(out)
                out = self.output_head_classif(out)
        else:
            if self.regression_type in ['clustering', 'clustering-cosine']:
                use_cosine_similarity = self.regression_type == 'clustering-cosine'
                out = self.forward_clustering_head(encoder_outputs,
                                                   self.dense_reg,
                                                   self.output_head_reg,
                                                   use_cosine_similarity=use_cosine_similarity)
            else:
                out = self.dense_reg(encoder_outputs)
                out = gelu(out)
                out = self.output_head_reg(out)

        if labels is None:
            if is_classification:
                if self.classification_type == 'clustering':
                    out = torch.sigmoid(out)
                if self.classification_type in ['clustering', 'clustering-cosine']:
                    out = (out + out.transpose(-2, -1)) / 2
            else:
                if self.regression_type == 'clustering':
                    out = torch.sigmoid(out)
                if self.regression_type in ['clustering', 'clustering-cosine']:
                    out = (out + out.transpose(-2, -1)) / 2
            return out

        assert target is not None

        if is_classification:
            if self.classification_type in ['clustering', 'clustering-cosine']:
                out, loss, metric = self.compute_clustering_output_loss_and_metric(
                    out, labels, target, is_cosine_similarity=self.classification_type == 'clustering-cosine')
            elif self.classification_type in ['triplet-l2', 'triplet-cosine']:
                loss, metric = self.compute_triplet_loss_and_metric(out, labels, target, anchors_idx,
                                                                    positives_idx, negatives_idx)
            else:
                loss, metric = self.compute_classif_loss_and_metric(out, labels, target)
        else:
            assert target_delta is not None
            real_target = torch.round(target + target_delta).int()
            if self.regression_type in ['clustering', 'clustering-cosine']:
                out, loss, metric = self.compute_clustering_output_loss_and_metric(
                    out, labels, real_target, is_cosine_similarity=self.regression_type == 'clustering-cosine')
            else:
                out, loss, metric = self.compute_regression_output_loss_and_metric(out, labels, real_target)

        return out, loss, metric

    def copy_last_layer_weights_to_all(self, state_dict):
        # Find encoder layers by filtering keys containing 'in_context_encoder'
        encoder_layers = [key for key in state_dict.keys() if 'in_context_encoder' in key]

        # Extract max layer number using regex (if they follow a pattern like in_context_encoder.X)
        layer_numbers = []
        for key in encoder_layers:
            match = re.search(r'in_context_encoder\.(\d+)', key)
            if match:
                layer_numbers.append(int(match.group(1)))
        last_layer_num = max(layer_numbers)

        for k in list(state_dict.keys()):
            if f'in_context_encoder.{last_layer_num}.' in k:
                for layer_idx in range(last_layer_num):
                    state_dict[k.replace(f'in_context_encoder.{last_layer_num}.',
                                         f'in_context_encoder.{layer_idx}.')] = state_dict[k]
        return state_dict

    def load_weights(self, checkpoint_path: Union[str, Path], device: torch.device, is_load_rnn=False):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        try:
            if is_load_rnn:
                state_dict = self.copy_last_layer_weights_to_all(state_dict)
            self.load_state_dict(state_dict)
        except RuntimeError:
            if is_load_rnn:
                state_dict = self.copy_last_layer_weights_to_all(state_dict)
            # Remove module. in front of all keys - maybe added by deepspeed?
            self.load_state_dict({k.removeprefix('module.'): v for k, v in state_dict.items()})

    def extract_prediction_classification(self, logits: torch.Tensor, targets: torch.Tensor, label_classes: np.ndarray):
        test_mask = (targets <= -99)

        if self.classification_type in ['clustering', 'clustering-cosine']:
            test_preds, test_logits = self._extract_prediction_clustering(logits, targets, test_mask, label_classes)
        elif self.classification_type in ['triplet-l2', 'triplet-cosine']:
            test_preds, test_logits, _ = self._extract_prediction_triplet(logits, targets, test_mask, label_classes)
        else:
            test_logits = logits[test_mask]
            test_logits = test_logits[:, :len(label_classes)].cpu().float()
            test_preds_indices = torch.argmax(test_logits, dim=-1).numpy()
            test_preds = label_classes[test_preds_indices]
        return test_preds, test_logits

    def extract_prediction_regression(self,
                                      logits: torch.Tensor,
                                      targets: torch.Tensor,
                                      label_classes: Union[np.ndarray, torch.Tensor],
                                      target_mean: Optional[torch.Tensor] = None,
                                      target_std: Optional[torch.Tensor] = None):
        test_mask = (targets <= -99)

        if isinstance(label_classes, torch.Tensor):
            label_classes = label_classes.cpu().numpy()

        if self.regression_type == 'reg-as-classif':
            test_logits = logits[test_mask]
            test_probas = torch.softmax(test_logits[:, :len(label_classes)], dim=1).cpu().float().numpy()
            test_preds = test_probas @ label_classes
        elif self.regression_type in ['clustering', 'clustering-cosine']:
            _, test_logits = self._extract_prediction_clustering(logits, targets, test_mask, label_classes)
            test_probas = torch.softmax(test_logits[:, :len(label_classes)], dim=1).cpu().float().numpy()
            test_preds = test_probas @ label_classes
        else:
            assert target_mean is not None and target_std is not None
            test_logits = logits[test_mask]
            # rescale prediction to the original scale
            test_preds = (test_logits * target_std + target_mean).cpu().float().numpy()
            test_probas = None
        return test_preds, test_probas

    def _extract_prediction_triplet(self, logits: torch.Tensor, targets: torch.Tensor, test_mask: torch.Tensor,
                                    label_classes: np.ndarray):
        """
        logits has shape  (context + queries, hidden_dim)
        targets has shape (context + queries, ) and contains the target values for each row.
        test_mask has shape (context + queries,) and indicates which rows are queries (True) and which are contexts (False).
        """
        context_mask = ~test_mask
        targets_for_context = targets[context_mask].cpu()

        query_logits = logits[test_mask]
        context_logits = logits[context_mask]

        if self.classification_type == 'triplet-l2':
            distances = torch.cdist(query_logits, context_logits,
                                    p=2.0).cpu().float().detach()  # [num_queries, num_contexts]
        else:
            query_norm = F.normalize(query_logits, p=2, dim=1)
            context_norm = F.normalize(context_logits, p=2, dim=1)
            cosine_similarity = torch.mm(query_norm, context_norm.T).cpu().float().detach()
            distances = 2 - (1 + cosine_similarity)  # map it to range [0, 2] where closest means best

        queries_num = distances.shape[0]

        test_logits = torch.zeros((queries_num, len(label_classes)))

        for query_idx, query_distances in enumerate(distances):
            query_best_context_idx = torch.argmin(query_distances)
            best_class = targets_for_context[query_best_context_idx]

            min_distance_from_best = query_distances[query_best_context_idx]
            context_with_best_class = targets_for_context == best_class
            max_distance_from_best = torch.max(query_distances[context_with_best_class])

            for c in targets_for_context.unique():
                context_from_class_c = targets_for_context == c
                min_distance_from_c = torch.min(query_distances[context_from_class_c])
                rescaled_distance = (min_distance_from_c - min_distance_from_best) / (max_distance_from_best
                                                                                      - min_distance_from_best + 1e-10)
                # best class has logit 0, while all the other classes have logits < 0
                # if the distance is greater than the maximum distance in the range then they have logits -inf
                test_logits[query_idx, c] = 1 - 1 / max(1e-5, 1 - rescaled_distance)

        probabilities = torch.softmax(test_logits, dim=1)
        test_preds = torch.argmax(probabilities, dim=1).numpy()
        test_preds = label_classes[test_preds]
        return test_preds, test_logits, distances

    def _extract_prediction_clustering(self, similarities: torch.Tensor, targets: torch.Tensor, test_mask: torch.Tensor,
                                       label_classes: np.ndarray):
        """
        similarities has hape (num_rows, num_rows) and contains similarities between all pairs of rows.
        targets has shape (num_rows, ) and contains the target values for each row.
        test_mask has shape (num_rows) and indicates which rows are queries (True) and which are contexts (False).
        """

        context_mask = ~test_mask
        targets_for_context = targets[context_mask].cpu()
        # get queries in rows and corresponding contexts in columns
        similarities_masked = similarities[test_mask][:, context_mask].cpu()  # [queries_num, contexts_num]

        queries_num = similarities_masked.shape[0]
        test_similarities = torch.full((queries_num, len(label_classes)),
                                       float('-inf'),
                                       dtype=similarities_masked.dtype)
        index = targets_for_context.unsqueeze(0).expand(queries_num, -1)

        test_similarities.scatter_reduce_(
            dim=1,
            index=index,
            src=similarities_masked,
            reduce='amax',
            include_self=False  # Changes nothing, it's -inf anyway
        )

        test_preds = torch.argmax(test_similarities, dim=1).numpy()
        test_preds = label_classes[test_preds]

        # Similarities above are already in [0, 1] and -inf
        # We go to logits here, because "logits" can be transformed to probabilities by usual softmax
        # However clip, because we need to avoid infinities in the logit space, otherwise softmax becomes NaN
        test_logits = torch.logit(test_similarities, eps=1e-6)
        test_logits = torch.clip(torch.nan_to_num(test_logits, -1e4), -1e4, 1e4)
        return test_preds, test_logits


class CosineTripletLoss(nn.Module):

    def __init__(self, margin=0.5):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings for cosine similarity
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)

        # Compute cosine similarities
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
        neg_sim = torch.sum(anchor_norm * negative_norm, dim=1)

        # Compute triplet loss
        # Note: For cosine similarity, larger values mean more similar
        # So we need to flip the inequality compared to distance-based triplet loss
        losses = F.relu(self.margin - (pos_sim - neg_sim))

        return losses.mean()
