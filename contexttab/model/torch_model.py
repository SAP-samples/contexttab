# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torcheval.metrics.functional import r2_score
from torch.utils.checkpoint import checkpoint_sequential
from transformers.activations import gelu
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.roberta.modeling_roberta import RobertaConfig

from contexttab.constants import ModelSize
from contexttab.data.tokenizer import Tokenizer
from contexttab.model.attention import TwoDimensionalAttentionLayer
from contexttab.model.embeddings import CellEmbeddings

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'


class ConTextTab(nn.Module, ModuleUtilsMixin):
    """ConTextTab model class.

    Args:
        model_size: size of the model e.g. ModelSize.mini or ModelSize.base
        regression_type: regression type that was used in the specified model
            - reg-as-classif - binned regression where bin is associated with the quantile of a given column
            - l2 - direct prediction of the target value with L2 loss during training
        classification_type: classification type that was used in the specified model
            - cross-entropy - class likelihood prediction using cross entropy loss during training
            - clustering - class prediction using similarity between context and query vectors
            - clustering-cosine - class prediction using cosine similarity between context and query vectors
        attention_implementation: backend for scaled dot product attention: math, efficient.
        checkpointing_segments: number of model's chunks/segments to checkpoint during training to save memory.
    """
    def __init__(self,
                 model_size: ModelSize,
                 regression_type: Literal['reg-as-classif', 'l2'] = 'reg-as-classif',
                 classification_type: Literal['cross-entropy', 'clustering', 'clustering-cosine'] = 'cross-entropy',
                 attention_implementation='efficient',
                 checkpointing_segments=1,
                 **kwargs):
        super().__init__()
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
        max_number_of_labels = Tokenizer.QUANTILE_DIMENSION

        if self.classification_type in ['clustering', 'clustering-cosine']:
            # adjacency matrix prediction head
            self.cluster_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.cluster_out_dim = self.config.hidden_size
            self.cluster_output_head = nn.Linear(self.config.hidden_size, self.cluster_out_dim)
        else:
            # standard class prediction head
            self.dense_classif = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.output_head_classif = nn.Linear(self.config.hidden_size, max_number_of_labels)

        self.dense_reg = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        if self.regression_type == 'l2':
            self.output_head_reg = nn.Linear(self.config.hidden_size, 1)
        else:
            self.output_head_reg = nn.Linear(self.config.hidden_size, max_number_of_labels)

        assert 0 <= checkpointing_segments <= self.config.num_hidden_layers
        self.checkpointing_segments = checkpointing_segments

        self.config.attention_implementation = attention_implementation

        self.embeddings = CellEmbeddings(self.config,
                                         regression_type=regression_type,
                                         is_target_content_mapping=(classification_type != 'cross-entropy'))
        self.in_context_encoder = nn.ModuleList(
            [TwoDimensionalAttentionLayer(self.config) for _ in range(self.config.num_hidden_layers)])

    @staticmethod
    def build_context_attention_mask(data, device):
        """
        Builds a context attention mask of shape (num_rows, num_rows)
        Everything can attend to context but query can only be attended by itself.
        This has the output shape of get_extended_attention_mask, which means:
        - 0 on the diagonal, as well as in any (i, j) position with j in context and any i;
        - -inf elsewhere
        """
        assert data['target'].ndim == 1, \
            'Expected target to be a 1D tensor, got shape: {}'.format(data['target'].shape)
        num_rows = int(data['target'].numel())
        context_attention_mask = torch.eye(num_rows)

        context_rows = data['target'] > -99
        context_attention_mask[:, context_rows] = 1

        # The following is equivalent to calling self.get_extended_attention_mask which however is a bit weird (needs batch,
        # needs a useless second parameter) so we avoid calling it
        context_attention_mask = context_attention_mask.to(device)
        return (1.0 - context_attention_mask) * torch.finfo(context_attention_mask.dtype).min

    @staticmethod
    def compute_classif_loss_and_metric(logits, labels, train_target):
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
                      target_delta: Optional[torch.Tensor] = None):
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
            out = self.dense_reg(encoder_outputs)
            out = gelu(out)
            out = self.output_head_reg(out)

        if labels is None:
            if is_classification:
                if self.classification_type == 'clustering':
                    out = torch.sigmoid(out)
                if self.classification_type in ['clustering', 'clustering-cosine']:
                    out = (out + out.transpose(-2, -1)) / 2
                if self.regression_type == 'l2':
                    out = out.squeeze(-1)
            return out

        assert target is not None

        if is_classification:
            if self.classification_type in ['clustering', 'clustering-cosine']:
                out, loss, metric = self.compute_clustering_output_loss_and_metric(
                    out, labels, target, is_cosine_similarity=self.classification_type == 'clustering-cosine')
            else:
                loss, metric = self.compute_classif_loss_and_metric(out, labels, target)
        else:
            assert target_delta is not None
            real_target = torch.round(target + target_delta).int()
            out, loss, metric = self.compute_regression_output_loss_and_metric(out, labels, real_target)

        return out, loss, metric

    @staticmethod
    def copy_last_layer_weights_to_all(state_dict):
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

    def load_weights(self, checkpoint_path: Union[str, Path], device: torch.device, is_copy_last_layer=True):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        try:
            if is_copy_last_layer:
                state_dict = self.copy_last_layer_weights_to_all(state_dict)
            # Remove module. in front of all keys - maybe added by deepspeed?
            self.load_state_dict({k.removeprefix('module.'): v for k, v in state_dict.items()})
        except:
            return self.load_weights(checkpoint_path, device, is_copy_last_layer=True)

    def extract_prediction_classification(self, logits: torch.Tensor, targets: torch.Tensor, label_classes: np.ndarray):
        test_mask = (targets <= -99)

        if self.classification_type in ['clustering', 'clustering-cosine']:
            test_preds, test_logits = self._extract_prediction_clustering(logits, targets, test_mask, label_classes)
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
        else:
            assert target_mean is not None and target_std is not None
            test_logits = logits[test_mask]
            # rescale prediction to the original scale
            test_preds = (test_logits * target_std + target_mean).cpu().float().numpy()
            test_probas = None
        return test_preds, test_probas

    @staticmethod
    def _extract_prediction_clustering(similarities: torch.Tensor, targets: torch.Tensor, test_mask: torch.Tensor,
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

    def forward(self, data: dict[str, torch.Tensor], is_regression: bool, labels=None, **kwargs):
        input_embeds = self.embeddings(data, is_regression)
        # (max_num_rows, max_num_columns, hidden_size)

        extended_attention_mask = self.build_context_attention_mask(data, input_embeds.device)
        extended_attention_mask = extended_attention_mask.type(input_embeds.dtype)

        if self.checkpointing_segments == 0:
            for layer in self.in_context_encoder:
                input_embeds = layer(input_embeds, extended_attention_mask)
            encoder_outputs = input_embeds
        else:
            # Remark: we need to bind `module` during creation time to avoid the issue that just doing
            #    lambda x: module(x, extended_attention_mask) for module in self.in_context_encoder
            # ends up always capturing the last value of the loop for all functions.
            functions_to_checkpoint = [
                lambda x, mod=module: mod(x, extended_attention_mask) for module in self.in_context_encoder
            ]

            # We checkpoint the forward pass of the encoder, to save memory.
            encoder_outputs = checkpoint_sequential(functions_to_checkpoint,
                                                    segments=self.checkpointing_segments,
                                                    input=input_embeds,
                                                    use_reentrant=False)

        # encoder_outputs has shape (num_rows, num_columns, hidden_size)

        target_column_output = encoder_outputs[:, -1]  # (num_rows, hidden_size)

        return self.forward_heads(target_column_output, is_regression, labels, data['target'], data['target_delta'])
