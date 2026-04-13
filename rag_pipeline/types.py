"""
Shared pipeline data types.

Author: Gayatri Malladi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
        }

    @property
    def is_poisoned(self) -> bool:
        # New format: poisoned docs have _id starting with "IPI_"
        # Old format: had _poisoned in metadata
        return self.doc_id.startswith("IPI_") or bool(self.metadata.get("_poisoned", False))
