# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-Memory implementation of HopliteDBInterface for testing."""

import collections
import dataclasses
from typing import Any, Sequence

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import interface


@dataclasses.dataclass
class InMemoryGraphSearchDB(interface.HopliteDBInterface):
  """Graph search database backed by NumPy arrays.

  We add embeddings to a fixed-sized array, keeping track of which indices are
  occupied. This avoids copies and resizing of the embeddings table.
  """

  embedding_dim: int
  max_size: int
  embeddings: np.ndarray
  embedding_sources: dict[int, interface.EmbeddingSource] = dataclasses.field(
      default_factory=dict
  )
  embedding_ids: set[int] = dataclasses.field(default_factory=set)
  embedding_dtype: type[Any] = np.float16
  labels: dict[int, list[interface.Label]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(list)
  )
  kv_store: dict[str, config_dict.ConfigDict] = dataclasses.field(
      default_factory=dict
  )

  @classmethod
  def create(cls, **kwargs):
    """Connect to and, if needed, initialize the database."""
    if 'embeddings' in kwargs:
      db = cls(**kwargs)
      return db

    embeddings = np.zeros(
        [kwargs['max_size'], kwargs['embedding_dim']],
        dtype=kwargs.get('embedding_dtype', np.float16),
    )
    db = cls(embeddings=embeddings, **kwargs)
    return db

  def thread_split(self) -> interface.HopliteDBInterface:
    """Return a readable instance of the database."""
    # Since numpy arrays are in shared memory, we can reuse the same object.
    return self

  def count_embeddings(self) -> int:
    """Return a count of all embeddings in the database."""
    return len(self.embedding_ids)

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    self.kv_store[key] = value

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    if key is None:
      return config_dict.ConfigDict(self.kv_store)
    return self.kv_store[key]

  def get_dataset_names(self) -> Sequence[str]:
    """Get all dataset names in the database."""
    ds_names = set()
    for source in self.embedding_sources.values():
      ds_names.add(source.dataset_name)
    return tuple(ds_names)

  def embedding_dimension(self) -> int:
    return self.embedding_dim

  def insert_embedding(
      self, embedding: np.ndarray, source: interface.EmbeddingSource
  ) -> int:
    """Add an embedding to the database."""
    if len(self.embedding_ids) >= self.max_size:
      # TODO(tomdenton): Automatically resize instead of throwing an error.
      raise ValueError('No more space in predefined memory.')

    if len(self.embedding_ids) not in self.embedding_ids:
      idx = len(self.embedding_ids)
    else:
      # find an unused index...
      idx = None
      for candidate in range(self.max_size):
        if candidate not in self.embedding_ids:
          idx = candidate
      if not idx:
        raise ValueError('Could not find an unused index.')

    self.embedding_ids.add(idx)
    self.embeddings[idx] = embedding[np.newaxis, :]
    self.embedding_sources[idx] = source
    return idx

  def get_embedding(self, embedding_id: int) -> np.ndarray:
    """Retrieve an embedding from the database."""
    return self.embeddings[embedding_id]

  def get_embedding_source(
      self, embedding_id: int
  ) -> interface.EmbeddingSource:
    return self.embedding_sources[embedding_id]

  def get_embeddings(
      self, embedding_ids: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    return embedding_ids, self.embeddings[embedding_ids]

  def get_embedding_ids(self) -> np.ndarray:
    """Get all embedding IDs in the database."""
    return np.array(tuple(int(id_) for id_ in self.embedding_ids))

  def get_one_embedding_id(self) -> int:
    return next(iter(self.embedding_ids))

  def get_embeddings_by_source(
      self,
      dataset_name: str,
      source_id: str | None,
      offsets: np.ndarray | None = None,
  ) -> np.ndarray:
    found_idxes = set()
    for idx, embedding_source in self.embedding_sources.items():
      if dataset_name and dataset_name != embedding_source.dataset_name:
        continue
      if source_id is not None and source_id != embedding_source.source_id:
        continue
      if offsets is not None and not np.array_equal(
          offsets, embedding_source.offsets
      ):
        continue
      found_idxes.add(idx)
    return np.array(tuple(found_idxes), np.int64)

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    pass

  def insert_label(
      self, label: interface.Label, skip_duplicates: bool = False
  ) -> bool:
    if label.type is None:
      raise ValueError('label type must be set')
    if label.provenance is None:
      raise ValueError('label source must be set')
    if skip_duplicates and label in self.get_labels(label.embedding_id):
      return False

    self.labels[label.embedding_id].append(label)
    return True

  def get_embeddings_by_label(
      self,
      label: str,
      label_type: interface.LabelType | None = interface.LabelType.POSITIVE,
      provenance: str | None = None,
  ) -> np.ndarray:
    found_idxes = set()
    for idx, emb_labels in self.labels.items():
      for emb_label in emb_labels:
        if emb_label.label != label:
          continue
        if label_type is not None and emb_label.type.value != label_type.value:
          continue
        if provenance is not None and emb_label.provenance != provenance:
          continue
        found_idxes.add(idx)
    return np.array(tuple(found_idxes), np.int64)

  def get_labels(self, embedding_id: int) -> Sequence[interface.Label]:
    return self.labels[embedding_id]

  def get_classes(self) -> Sequence[str]:
    label_set = set()
    for labels in self.labels.values():
      for l in labels:
        label_set.add(l.label)
    return tuple(sorted(label_set))

  def get_class_counts(
      self, label_type: interface.LabelType = interface.LabelType.POSITIVE
  ) -> dict[str, int]:
    class_counts = collections.defaultdict(int)
    for labels in self.labels.values():
      counted_labels = set()
      for l in labels:
        # Avoid double-counting the same label on the same embedding because of
        # different provenances.
        if l.label in counted_labels:
          continue
        if l.type.value == label_type.value:
          class_counts[l.label] += 1
          counted_labels.add(l.label)
        else:
          # Creates a key in the dict for all labels, even if they have no
          # matching type counts.
          class_counts[l.label] += 0
    return class_counts
