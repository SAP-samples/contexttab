# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from enum import Enum



embedding_model_to_dimension_and_pooling = {
    'sentence-transformers/all-MiniLM-L6-v2': (384, 'mean'),
    'intfloat/multilingual-e5-small': (384, 'mean'),
    'Alibaba-NLP/gte-multilingual-base': (768, 'cls'),
}

ZMQ_PORT_DEFAULT = 5655
QUANTILE_DIMENSION_DEFAULT = 64

class ModelSize(Enum):
    # The two values are the number of layers and the hidden size
    tiny = (2, 128)
    mini = (4, 256)
    small = (4, 512)
    medium = (8, 512)
    base = (12, 768)
    large = (24, 1024)
    xlarge = (24, 2048)


class ModelSizeAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values not in ModelSize.__members__ or not isinstance(values, str):
            raise ValueError(f'{values} is not a valid value for ModelSize: {ModelSize.__members__.keys()}')
        value = ModelSize[values]
        setattr(namespace, self.dest, value)
