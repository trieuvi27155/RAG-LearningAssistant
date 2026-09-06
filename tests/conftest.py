"""Cấu hình dùng chung cho toàn bộ test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag import bo_nho_dem


@pytest.fixture(autouse=True, scope="session")
def khong_ghi_vao_thu_muc_du_an(tmp_path_factory):
    """Trỏ cache và thư mục ảnh sang thư mục tạm của phiên test."""
    goc = tmp_path_factory.mktemp("du_lieu_test")
    config.CACHE_DIR = goc / "cache"
    config.IMAGES_DIR = goc / "images"
    for thu_muc in (config.CACHE_DIR, config.IMAGES_DIR):
        thu_muc.mkdir(parents=True, exist_ok=True)

    bo_nho_dem.kho_tai_lieu = bo_nho_dem.KhoDem("tai_lieu", ".json")
    bo_nho_dem.kho_ocr = bo_nho_dem.KhoDem("ocr", ".txt")
    bo_nho_dem.kho_vision = bo_nho_dem.KhoDem("vision", ".txt")
    return goc
