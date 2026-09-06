"""Test cho khung đo (evaluation/run_evaluation.py) - logic thuần, không cần model/Ollama."""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_evaluation import (
    _in_bang_theo_nhom,
    _trung_binh,
    doc_csv_ket_qua,
    so_sanh_voi_ban_truoc,
)

CAC_COT = [
    "cau_hoi", "cau_tra_loi", "precision_at_k", "recall_at_k",
    "faithfulness", "answer_relevance", "do_tre_giay", "loai_tai_lieu", "loai_cau_hoi",
]


def _viet_csv(duong_dan: Path, cac_dong: list) -> None:
    with open(duong_dan, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CAC_COT)
        w.writeheader()
        w.writerows(cac_dong)


def _dong(p=1.0, r=1.0, f=1.0, a=1.0, tre=1.0, loai_tl="dai", loai_ch="truy_xuat") -> dict:
    return {
        "cau_hoi": "câu hỏi mẫu", "cau_tra_loi": "trả lời mẫu",
        "precision_at_k": p, "recall_at_k": r, "faithfulness": f,
        "answer_relevance": a, "do_tre_giay": tre,
        "loai_tai_lieu": loai_tl, "loai_cau_hoi": loai_ch,
    }


def test_trung_binh_bo_qua_o_thieu_du_lieu():
    """CSV của lần chạy CŨ có thể thiếu cột metric mới thêm sau này - nếu _trung_binh nổ
    hoặc coi ô trống là 0 thì phần so sánh trước/sau sẽ ra kết luận sai."""
    cac_muc = [{"diem": 1.0}, {"diem": ""}, {}, {"diem": 0.5}]
    assert _trung_binh(cac_muc, "diem") == 0.75
    assert _trung_binh([], "diem") == 0.0
    assert _trung_binh([{}], "diem") == 0.0


def test_so_sanh_khong_no_khi_chua_co_lan_chay_truoc(capsys):
    with tempfile.TemporaryDirectory() as thu_muc:
        moi = Path(thu_muc) / "moi.csv"
        _viet_csv(moi, [_dong()])
        so_sanh_voi_ban_truoc(moi, Path(thu_muc) / "khong_ton_tai.csv")
    assert "lần chạy đầu tiên" in capsys.readouterr().out


def test_so_sanh_in_dung_chenh_lech(capsys):
    with tempfile.TemporaryDirectory() as thu_muc:
        cu = Path(thu_muc) / "cu.csv"
        moi = Path(thu_muc) / "moi.csv"
        _viet_csv(cu, [_dong(p=0.4, tre=2.0)])
        _viet_csv(moi, [_dong(p=0.8, tre=5.0)])

        so_sanh_voi_ban_truoc(moi, cu)
        ra = capsys.readouterr().out

    assert "+0.40" in ra
    assert "+3.00" in ra
    dong_precision = next(d for d in ra.splitlines() if d.startswith("precision_at_k"))
    dong_do_tre = next(d for d in ra.splitlines() if d.startswith("do_tre_giay"))
    assert "✔" in dong_precision
    assert "✘" in dong_do_tre


def test_doc_csv_ket_qua_giu_nguyen_tieng_viet():
    with tempfile.TemporaryDirectory() as thu_muc:
        duong_dan = Path(thu_muc) / "kq.csv"
        dong = _dong()
        dong["cau_hoi"] = "Phí phạt mỗi ngày là bao nhiêu?"
        _viet_csv(duong_dan, [dong])
        assert doc_csv_ket_qua(duong_dan)[0]["cau_hoi"] == "Phí phạt mỗi ngày là bao nhiêu?"


def test_bang_theo_nhom_tach_dung_nhom(capsys):
    ket_qua = [
        _dong(p=1.0, loai_tl="dai"),
        _dong(p=0.0, loai_tl="dai"),
        _dong(p=0.5, loai_tl="ngan"),
    ]
    _in_bang_theo_nhom(ket_qua, "loai_tai_lieu", "Loại tài liệu")
    ra = capsys.readouterr().out

    assert "dai" in ra and "ngan" in ra
    dong_dai = next(d for d in ra.splitlines() if d.startswith("dai"))
    assert "0.50" in dong_dai
    assert "2" in dong_dai.split()[1]


def test_bang_theo_nhom_bo_qua_khi_bo_cau_hoi_chua_khai_bao_nhom(capsys):
    """Bộ câu hỏi cũ không có trường loai_tai_lieu - phải bỏ qua im lặng, không được nổ."""
    _in_bang_theo_nhom([_dong(loai_tl="", loai_ch="")], "loai_tai_lieu", "Loại tài liệu")
    assert capsys.readouterr().out == ""


def test_bo_qua_so_sanh_khi_khac_bo_cau_hoi(tmp_path, capsys):
    """Bỏ qua bảng so sánh khi file kết quả cũ thuộc bộ câu hỏi khác, vì các con số chênh lệch
    khi đó vô nghĩa."""
    cu, moi = tmp_path / "cu.csv", tmp_path / "moi.csv"
    _viet_csv(cu, [{**_dong(p=0.9, r=0.9), "cau_hoi": "Thư viện số ra đời năm nào?"}])
    _viet_csv(moi, [{**_dong(p=0.4, r=0.4), "cau_hoi": "What is the bias-variance decomposition?"}])

    so_sanh_voi_ban_truoc(moi, cu)
    ket_qua = capsys.readouterr().out
    assert "BỎ QUA so sánh" in ket_qua
    assert "Chênh lệch" not in ket_qua, "không được in bảng chênh lệch vô nghĩa"


def test_van_so_sanh_khi_cung_bo_cau_hoi(tmp_path, capsys):
    cu, moi = tmp_path / "cu.csv", tmp_path / "moi.csv"
    _viet_csv(cu, [{**_dong(r=0.5), "cau_hoi": "Nhà nước là gì?"}])
    _viet_csv(moi, [{**_dong(r=0.9), "cau_hoi": "Nhà nước là gì?"}])

    so_sanh_voi_ban_truoc(moi, cu)
    ket_qua = capsys.readouterr().out
    assert "Chênh lệch" in ket_qua
    assert "BỎ QUA" not in ket_qua


import json

import config


def _nap(duong_dan):
    return json.loads(Path(duong_dan).read_text(encoding="utf-8"))


def test_bo_held_out_khong_dung_chung_tai_lieu_nao_voi_bo_in_sample():
    trong_mau = _nap(config.TEST_QUESTIONS_FILE)
    ngoai_mau = _nap(config.TEST_QUESTIONS_HELD_OUT_FILE)

    def cac_nguon(bo):
        return {p["nguon"] for m in bo for p in m["cac_trang_dung"]}

    chung = cac_nguon(trong_mau) & cac_nguon(ngoai_mau)
    assert not chung, (
        f"Tài liệu {sorted(chung)} xuất hiện ở CẢ HAI bộ - bộ held-out không còn held-out, "
        "và khoảng cách đo được sẽ nhỏ đi vì một lý do giả."
    )


def test_bo_held_out_khong_dung_chung_cau_hoi_nao():
    trong_mau = {m["cau_hoi"] for m in _nap(config.TEST_QUESTIONS_FILE)}
    ngoai_mau = {m["cau_hoi"] for m in _nap(config.TEST_QUESTIONS_HELD_OUT_FILE)}
    assert not (trong_mau & ngoai_mau)


def test_bo_held_out_du_lon_va_du_da_dang_de_so_sanh_duoc():
    """Bộ held-out phải cùng cấu trúc trường và phủ đủ các nhóm câu hỏi của bộ in-sample thì hai
    bảng mới đặt cạnh nhau so sánh được."""
    ngoai_mau = _nap(config.TEST_QUESTIONS_HELD_OUT_FILE)

    assert len(ngoai_mau) >= 15, "dưới 15 câu thì chênh lệch đo được lẫn vào nhiễu"
    for muc in ngoai_mau:
        assert {"cau_hoi", "cac_trang_dung", "dap_an_mau"} <= set(muc)
        for trang in muc["cac_trang_dung"]:
            assert {"nguon", "trang"} == set(trang)

    cac_nhom = {m["loai_cau_hoi"] for m in ngoai_mau}
    assert {"truy_xuat", "cheo_ngon_ngu", "tu_choi"} <= cac_nhom
