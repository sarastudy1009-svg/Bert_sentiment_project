"""재현성 설정과 실행 장치 선택 유틸리티입니다."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch의 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA 사용 가능 여부에 따라 GPU 또는 CPU 장치를 반환합니다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
