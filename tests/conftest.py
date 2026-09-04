"""Cấu hình dùng chung cho toàn bộ test.

MỤC ĐÍCH DUY NHẤT: không một test nào được ghi vào thư mục dữ liệu THẬT của dự án.

Vì sao cần từ khi có bộ nhớ đệm Ingestion: `doc_thu_muc()` nay tự ghi kết quả đọc vào
`data/cache/`, nên một test dựng file DOCX tạm rồi gọi nó sẽ để lại rác trong repo - và trên
một dự án chưa có thói quen .gitignore chặt chẽ thì đám rác đó rất dễ bị commit theo. Ảnh
trích ra (`data/images/`) cũng vậy.

Đây là loại lỗi không làm test đỏ nên sẽ không ai phát hiện qua CI; cách duy nhất là chặn từ
gốc, một lần, cho mọi test hiện có lẫn mọi test viết sau này.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from rag import bo_nho_dem


@pytest.fixture(autouse=True, scope="session")
def khong_ghi_vao_thu_muc_du_an(tmp_path_factory):
    """Trỏ cache và thư mục ảnh sang thư mục tạm của phiên test.

    Gán thẳng thay vì dùng `monkeypatch`: fixture này phải ở phạm vi SESSION (chỉ dựng một
    lần cho cả lần chạy) mà `monkeypatch` thì chỉ có ở phạm vi function. Tiến trình pytest
    kết thúc là mọi thay đổi biến mất, nên không cần khôi phục.

    Các kho trong `bo_nho_dem` phải được TẠO LẠI chứ không chỉ đổi `config.CACHE_DIR`: chúng
    chốt đường dẫn ngay lúc khởi tạo ở mức module, tức từ trước khi fixture này chạy.
    """
    goc = tmp_path_factory.mktemp("du_lieu_test")
    config.CACHE_DIR = goc / "cache"
    config.IMAGES_DIR = goc / "images"
    for thu_muc in (config.CACHE_DIR, config.IMAGES_DIR):
        thu_muc.mkdir(parents=True, exist_ok=True)

    bo_nho_dem.kho_tai_lieu = bo_nho_dem.KhoDem("tai_lieu", ".json")
    bo_nho_dem.kho_ocr = bo_nho_dem.KhoDem("ocr", ".txt")
    bo_nho_dem.kho_vision = bo_nho_dem.KhoDem("vision", ".txt")
    return goc
