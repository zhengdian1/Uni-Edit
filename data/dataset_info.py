# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .real_edit_dataset import SftEditJSONLIterableDataset

DATASET_REGISTRY = {
	'unified_edit': SftEditJSONLIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
}

DATASET_INFO = {
	'unified_edit': {
        '148k': {
			'data_dir': 'Uni-Edit-PATH',
			'jsonl_path': 'PATH_TO_data_148k.jsonl',
			'num_total_samples': 150000
		},
        '40k': {
			'data_dir': 'Uni-Edit-PATH',
			'jsonl_path': 'PATH_TO_data_40k.jsonl',
			'num_total_samples': 41000
		}
    },
}