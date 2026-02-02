from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class User:
    id: int
    name: str
    embedding: np.ndarray
    is_authorized: bool
    created_at: datetime


@dataclass
class AccessLog:
    id: int
    user_id: Optional[int]
    confidence: float
    access_granted: bool
    timestamp: datetime