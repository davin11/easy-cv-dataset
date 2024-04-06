# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
easy_cv_dataset.

A library for dataset loading.
"""

__version__ = "0.0.1"

from .image_dataset import image_dataframe_from_directory
from .image_dataset import image_classification_dataset_from_dataframe
from .image_dataset import image_segmentation_dataset_from_dataframe
from .image_dataset import image_objdetect_dataset_from_dataframe
