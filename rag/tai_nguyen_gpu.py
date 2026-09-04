"""Nhận biết phần cứng và quản lý GPU theo GIAI ĐOẠN - tự thích ứng với máy đang chạy.

HAI VIỆC, VÀ CHÚNG KHÁC NHAU:

1. **DÒ PHẦN CỨNG.** Hệ thống không được giả định máy nào cả. Có GPU NVIDIA dùng được thì
   embedding và reranker chạy trên đó; không có thì tự lùi về CPU và mọi thứ vẫn chạy đúng,
   chỉ chậm hơn. Batch size, số worker và ngưỡng VRAM đều SUY RA từ thứ máy tự báo cáo, chứ
   không phải hằng số hiệu chỉnh cho một card cụ thể - một con số hợp với card 8 GB sẽ vừa
   phí trên card 24 GB vừa gây tràn trên card 4 GB.

2. **CHIA GIAI ĐOẠN.** Trên GPU có VRAM hạn chế, các model của hệ thống cộng lại có thể
   không vừa. Đo bằng `nvidia-smi` và `/api/ps` của Ollama trên máy làm đồ án (RTX 5060,
   7,96 GB) - con số cụ thể tuỳ máy nhưng TỈ LỆ giữa các model thì không đổi:

       | Model                     | Vai trò              | VRAM     |
       |---------------------------|----------------------|----------|
       | qwen3:4b (num_ctx 16384)  | sinh câu trả lời     | 4,75 GB  |
       | bge-reranker-v2-m3        | xếp hạng lại         | 2,20 GB  |
       | multilingual-e5-base      | embedding            | 1,12 GB  |
       | **Tổng ở giai đoạn QUERY**|                      | **8,07 GB** |

   8,07 GB không vừa 7,96 GB của card - và model vision (dùng ở giai đoạn ingestion) còn
   chưa tính vào. Tràn VRAM **không báo lỗi**: driver âm thầm đẩy phần thừa sang RAM hệ
   thống, hoặc Ollama liên tục nạp/nhả model giữa các lượt gọi. Quan sát được khi để cả ba
   trên GPU: VRAM còn trống tụt xuống 288 MB, reranker mất ~58 giây mới nạp xong, lượt hỏi
   đầu tiên báo 50,8 giây cho bước truy xuất. Không một dòng lỗi nào - đúng loại hỏng im
   lặng mà cả project này được viết để tránh.

   Cách chia dựa trên một sự thật về LUỒNG SỬ DỤNG, không phải về phần cứng: người dùng bấm
   "Đọc tài liệu" rồi mới hỏi, nên hai giai đoạn không bao giờ chạy đồng thời.

       INGESTION  ->  Vision/OCR + embedding(GPU)
       QUERY      ->  reranker(GPU) + LLM(GPU), embedding lùi về CPU trên card chật

   Thứ bị đẩy xuống CPU là embedding, vì lúc truy vấn nó chỉ mã hoá 1-3 chuỗi ngắn (GPU chỉ
   nhanh hơn CPU 14 ms), còn reranker chấm vài chục cặp mỗi câu hỏi.

SỐ ĐO LÀM CƠ SỞ (RTX 5060 7,96 GB, xem KET_QUA_DO_DAC.md §8):
  - Embedding 512 chunk:      CPU 19,90 s -> GPU 1,55 s (**12,8×**).
  - Rerank 12 cặp:            CPU  1,68 s -> GPU 0,15 s (**11,2×**).
  - Truy xuất đầu-cuối 6 câu: CPU 15,21 s -> GPU 1,69 s (**9,0×**), và **6/6 câu cho kết
    quả GIỐNG HỆT** (lệch điểm tối đa 2,4e-07, tức sai số làm tròn float32).
  - Batch size 16→256 cho ra CÙNG throughput (320-326 chunk/s) trong khi VRAM đỉnh tăng
    1,21 → 2,80 GB. Vì vậy batch size ở đây được chọn theo VRAM CÒN TRỐNG chứ không theo
    tốc độ - tăng nó lên không mua được gì.

NGUYÊN TẮC KHI SỬA MODULE NÀY: không có GPU thì mọi hàm phải thành không-làm-gì chứ không
được ném lỗi. Đồ án phải chạy được trên máy chỉ có CPU - đó là môi trường phần lớn người
chấm sẽ dùng.
"""

import logging
import os
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)

# Nhớ kết quả kiểm tra CUDA để không phải import torch lại ở mỗi lần gọi (torch nặng, và
# thiet_bi() nằm trên đường khởi tạo của nhiều module).
_co_cuda: Optional[bool] = None


# ============================================================
# DÒ PHẦN CỨNG
# ============================================================
def co_cuda() -> bool:
    """Máy này có GPU dùng được cho PyTorch không.

    Bắt MỌI lỗi: `torch.cuda.is_available()` có thể ném khi driver không khớp phiên bản CUDA
    mà torch được biên dịch, khi thiếu DLL, hoặc khi chạy trong container không gắn GPU. Ở
    mọi tình huống đó câu trả lời đúng vẫn là "không có GPU" chứ không phải làm sập app.
    """
    global _co_cuda
    if _co_cuda is None:
        try:
            import torch

            _co_cuda = bool(torch.cuda.is_available())
        except Exception as loi:  # noqa: BLE001
            logger.info("Không dùng được CUDA (%s) - chạy trên CPU.", type(loi).__name__)
            _co_cuda = False
    return _co_cuda


def thiet_bi(vai_tro: str) -> str:
    """Thiết bị nên dùng cho một vai trò: "cuda" hoặc "cpu".

    vai_tro: "embedding" hoặc "rerank". Cấu hình cho phép ép RIÊNG từng vai trò thay vì một
    công tắc chung, vì hai model này có hồ sơ tài nguyên khác hẳn nhau: embedding chạy theo
    lô lớn lúc build index (rất hợp GPU), còn reranker chạy vài chục cặp mỗi câu hỏi và phải
    chia VRAM với LLM. Trên một card nhỏ, cấu hình hợp lý có thể là embedding trên GPU còn
    reranker trên CPU - và điều đó chỉ nói được nếu hai vai trò tách riêng.
    """
    cau_hinh = {
        "embedding": config.THIET_BI_EMBEDDING,
        "rerank": config.THIET_BI_RERANK,
    }[vai_tro]
    if cau_hinh != "auto":
        return cau_hinh
    if not co_cuda():
        return "cpu"
    # EMBEDDING MẶC ĐỊNH THEO GIAI ĐOẠN QUERY, không phải theo "có GPU thì dùng GPU".
    #
    # Lý do là một lỗ hổng đã quan sát được: bước chuyển giai đoạn (ket_thuc_ingestion) chỉ
    # chạy sau khi người dùng bấm "Đọc tài liệu". Nhưng phần lớn phiên làm việc lại KHÔNG bắt
    # đầu như vậy - người dùng mở app lên và hỏi ngay trên index đã có. Nếu mặc định là "cuda"
    # thì trong đúng những phiên đó, embedding nằm trên GPU tranh VRAM với LLM suốt cả phiên,
    # và đo được lượt hỏi mất 7,4 giây cho bước truy xuất lẽ ra chỉ 0,45 giây.
    #
    # Vì vậy trạng thái MẶC ĐỊNH phải là trạng thái của giai đoạn hay gặp nhất (query), còn
    # ingestion - vốn luôn đi qua bat_dau_ingestion() - thì tự nâng nó lên GPU đúng lúc cần.
    # Card đủ rộng thì không có đánh đổi nào, dùng GPU cho cả hai giai đoạn.
    if vai_tro == "embedding" and not du_cho_giu_embedding_tren_gpu():
        return "cpu"
    return "cuda"


def vram() -> Optional[Tuple[float, float, float]]:
    """(PyTorch đang giữ, tổng VRAM, VRAM còn trống) tính bằng GB. None nếu không có GPU.

    Ba con số chứ không phải một, vì chúng trả lời ba câu hỏi khác nhau: PyTorch chiếm bao
    nhiêu (phần ta điều khiển được), máy có bao nhiêu (trần cứng), và còn trống bao nhiêu -
    con số thứ ba mới là con số phải dựa vào để quyết định, vì nó tính cả phần Ollama và
    Windows đang dùng, những thứ hoàn toàn vô hình với torch.
    """
    if not co_cuda():
        return None
    import torch

    trong, tong = torch.cuda.mem_get_info(0)
    return (
        torch.cuda.memory_reserved(0) / (1 << 30),
        tong / (1 << 30),
        trong / (1 << 30),
    )


def tong_vram_gb() -> float:
    """Tổng VRAM của GPU (GB), 0.0 nếu không có GPU."""
    so_lieu = vram()
    return so_lieu[1] if so_lieu else 0.0


def vram_con_trong_gb() -> float:
    """VRAM còn trống ngay lúc này (GB), 0.0 nếu không có GPU."""
    so_lieu = vram()
    return so_lieu[2] if so_lieu else 0.0


def so_nhan_cpu() -> int:
    return os.cpu_count() or 2


def mo_ta_phan_cung() -> str:
    """Một dòng mô tả phần cứng đang được dùng, cho log và giao diện.

    Vì sao đáng có: "torch bản CPU-only trên một máy CÓ GPU" là cấu hình sai mà KHÔNG gây
    lỗi - hệ thống vẫn chạy đúng, chỉ chậm hơn cả chục lần. Không nói ra thì không ai biết,
    và đó chính là loại lỗi khiến người ta đi tối ưu nhầm chỗ suốt nhiều ngày.
    """
    if not co_cuda():
        return (
            f"Phần cứng: CPU {so_nhan_cpu()} nhân · KHÔNG dùng GPU "
            "(PyTorch bản CPU-only, hoặc máy không có GPU NVIDIA dùng được)"
        )
    import torch

    return (
        f"Phần cứng: {torch.cuda.get_device_name(0)} ({tong_vram_gb():.1f} GB VRAM) · "
        f"CPU {so_nhan_cpu()} nhân · embedding={thiet_bi('embedding')} "
        f"· rerank={thiet_bi('rerank')} · batch embedding={kich_thuoc_lo_embedding()}"
    )


# ============================================================
# THAM SỐ SUY TỪ PHẦN CỨNG
# ============================================================
def kich_thuoc_lo_embedding() -> int:
    """Batch size encode, suy từ VRAM CÒN TRỐNG. Trên CPU thì giữ nguyên giá trị cấu hình.

    Đây là chỗ một phép đo đã lật ngược trực giác thông thường ("GPU thì cứ tăng batch lên
    cho nhanh"). Đo trên RTX 5060 với 768 chunk:

        batch |  chunk/s | VRAM đỉnh
           16 |      320 |   1,21 GB
           32 |      326 |   1,31 GB
           64 |      325 |   1,52 GB
          128 |      325 |   1,92 GB
          256 |      324 |   2,80 GB

    Throughput ĐỨNG YÊN trên toàn dải (320-326 chunk/s, chênh lệch nằm trong nhiễu) trong
    khi VRAM đỉnh tăng hơn gấp đôi. Nghĩa là nút thắt không nằm ở độ song song của lô mà ở
    chỗ khác (băng thông bộ nhớ / phần tiền xử lý trên CPU), nên batch lớn chỉ mua thêm rủi
    ro tràn VRAM mà không mua được tốc độ.

    Vì vậy hàm này KHÔNG cố tăng batch khi máy khoẻ - nó chỉ HẠ batch khi máy chật. VRAM còn
    trống là con số đúng để dựa vào (không phải tổng VRAM): trên cùng một card, còn trống bao
    nhiêu phụ thuộc việc Ollama đang giữ model nào, và điều đó đổi theo từng giai đoạn.

    Giá trị cấu hình EMBEDDING_BATCH_SIZE đóng vai TRẦN chứ không phải giá trị đích - người
    dùng hạ nó xuống thì tôn trọng, nâng nó lên thì vẫn bị VRAM chặn.
    """
    tran = config.EMBEDDING_BATCH_SIZE
    # Chỉ hỏi co_cuda(), KHÔNG hỏi thiet_bi("embedding"): model được CHUYỂN qua lại giữa GPU
    # và CPU theo giai đoạn, nên thiết bị mặc định trong cấu hình không nói lên model đang
    # nằm ở đâu tại thời điểm encode. Nhìn nhầm chỗ đó thì đúng lúc build index - lúc model
    # vừa được đưa lên GPU - hàm này lại tưởng nó ở CPU và bỏ qua chốt VRAM.
    #
    # Chặt hơn cần thiết ở chiều ngược lại (model đang ở CPU mà vẫn hạ batch) là vô hại: đo
    # được batch 16 và 64 chênh nhau 3% throughput, tức không đáng để đổi lấy rủi ro tràn.
    if not co_cuda():
        return tran
    trong = vram_con_trong_gb()
    if trong >= config.VRAM_DU_CHO_LO_LON_GB:
        return tran
    if trong >= config.VRAM_DU_CHO_LO_VUA_GB:
        return min(tran, 32)
    logger.info(
        "VRAM còn trống chỉ %.1f GB - hạ batch embedding xuống 16 để tránh tràn.", trong
    )
    return min(tran, 16)


def so_worker_vision() -> int:
    """Số luồng gọi model vision/OCR song song, suy từ phần cứng đang có.

    Ba ràng buộc, lấy giá trị NHỎ NHẤT:

      1. **Trần cấu hình** (`SO_WORKER_VISION`) - người dùng luôn có tiếng nói cuối cùng.
      2. **Số nhân CPU** - phần render trang PDF và mã hoá ảnh sang base64 chạy trên CPU;
         mở nhiều luồng hơn số nhân chỉ tạo tranh chấp.
      3. **VRAM còn trống** - đây là ràng buộc mà một máy khác sẽ cho câu trả lời khác. Mỗi
         yêu cầu song song tới Ollama cần thêm một ngữ cảnh riêng trong VRAM; card càng nhỏ
         thì càng ít chỗ. Không có GPU thì ràng buộc này biến mất (Ollama chạy CPU, lúc đó
         chính số nhân CPU mới là trần thật).

    Đo trên RTX 5060 (OCR 6 trang Bishop, cache rỗng ở mỗi mức):

        worker | tổng    | mỗi trang | GPU util TB
             1 | 49,4 s  |   8,2 s   |     84%
             2 | 31,9 s  |   5,3 s   |     93%
             4 | 32,0 s  |   5,3 s   |     92%

    Tức **1→2 lợi 1,55×, 2→4 không lợi gì** (chênh 0,1 s, nằm trong nhiễu). Cột GPU
    utilization nói rõ vì sao: GPU đã bão hoà 84% ngay từ MỘT worker nên gần như không còn
    thời gian rảnh để lấp.

    Con số tối ưu khác nhau trên mỗi máy - chạy `evaluation/do_worker_gpu.py` để đo trên máy
    của bạn, script tự đề xuất giá trị.
    """
    tran = max(1, config.SO_WORKER_VISION)
    theo_cpu = max(1, so_nhan_cpu() // 2)
    if not co_cuda():
        return min(tran, theo_cpu)
    theo_vram = max(1, int(vram_con_trong_gb() // config.VRAM_MOI_WORKER_VISION_GB))
    ket_qua = min(tran, theo_cpu, theo_vram)
    if ket_qua < tran:
        logger.info(
            "Số worker vision: %d (trần cấu hình %d, theo CPU %d, theo VRAM còn trống %d).",
            ket_qua, tran, theo_cpu, theo_vram,
        )
    return ket_qua


# ============================================================
# QUẢN LÝ THEO GIAI ĐOẠN
# ============================================================
def ghi_log_vram(nhan: str) -> None:
    """Ghi tình trạng VRAM kèm nhãn giai đoạn. Không có GPU thì im lặng bỏ qua."""
    so_lieu = vram()
    if so_lieu is None:
        return
    torch_giu, tong, trong = so_lieu
    logger.info(
        "VRAM [%s]: PyTorch giữ %.2f GB · còn trống %.2f/%.2f GB.",
        nhan, torch_giu, trong, tong,
    )


def don_bo_nho_cuda() -> None:
    """Trả lại cho driver phần VRAM PyTorch đã cấp phát nhưng không còn dùng.

    PyTorch giữ lại bộ nhớ đã cấp phát trong một bộ đệm riêng để lần sau cấp phát nhanh hơn.
    Với một tiến trình chỉ dùng PyTorch thì đó là tối ưu đúng; ở đây thì không, vì phần VRAM
    bị giữ lại đó chính là phần Ollama cần để nạp LLM. Đây là lý do việc dọn phải làm TƯỜNG
    MINH ở ranh giới giai đoạn chứ không phó mặc cho bộ thu gom rác của Python.
    """
    if not co_cuda():
        return
    import torch

    torch.cuda.empty_cache()


def nha_model_ollama(ten_model: str, client=None) -> bool:
    """Bảo Ollama nhả một model khỏi VRAM NGAY, thay vì đợi hết thời gian giữ mặc định.

    Ollama giữ model trong VRAM 5 phút sau lượt gọi cuối. Với một máy chủ chỉ phục vụ một
    loại tác vụ thì đó là tối ưu đúng - lượt sau khỏi nạp lại. Nhưng ở đây, 5 phút đó rơi
    đúng vào lúc người dùng vừa build index xong và bắt đầu đặt câu hỏi: model vision vẫn
    nằm nguyên trong VRAM trong khi thứ đang cần là LLM và reranker.

    `keep_alive=0` là cách Ollama cung cấp để nhả ngay. Gửi kèm prompt rỗng vì ta không cần
    kết quả sinh nào - chỉ cần chạm vào model để đính tham số keep_alive vào.

    Trả về True nếu đã gửi được yêu cầu. Nuốt mọi lỗi: Ollama chưa chạy, model chưa từng
    được nạp, hay client phiên bản khác không nhận tham số này đều KHÔNG phải lý do làm hỏng
    một lần build vừa chạy xong.
    """
    if not config.NHA_MODEL_SAU_INGESTION:
        return False
    try:
        import ollama

        client = client or ollama.Client(host=config.OLLAMA_HOST)
        client.generate(model=ten_model, prompt="", keep_alive=0)
    except Exception as loi:  # noqa: BLE001
        logger.debug("Không nhả được model '%s' khỏi VRAM (%s).", ten_model, type(loi).__name__)
        return False
    logger.info("Đã yêu cầu Ollama nhả model '%s' khỏi VRAM.", ten_model)
    return True


def du_cho_giu_embedding_tren_gpu() -> bool:
    """Card này có đủ chỗ để GIỮ embedding trên GPU trong lúc truy vấn không?

    Ở giai đoạn query, GPU phải chứa cùng lúc ba thứ. Đo trên máy làm đồ án:

        LLM qwen3:4b (num_ctx 16384) 4,75 GB
        reranker bge-v2-m3           2,20 GB
        embedding e5-base            1,12 GB
        ------------------------------------
        tổng                         8,07 GB   > 7,96 GB của card

    Và hậu quả đã QUAN SÁT ĐƯỢC chứ không phải phòng xa: khi cả ba cùng nằm trên GPU, VRAM
    còn trống tụt xuống **288 MB**, model reranker mất gần một phút mới nạp xong, và lượt hỏi
    đầu tiên báo "truy xuất 50,8 giây" - trong khi phép đo cùng bước đó trên máy thoáng chỉ
    mất 2,6 giây.

    Thứ hy sinh ĐÚNG là embedding, và lý do nằm ở quy mô công việc chứ không ở kích thước
    model: lúc truy vấn nó chỉ mã hoá 1-3 chuỗi ngắn (đo được 20,3 ms trên CPU so với 6,1 ms
    trên GPU - chênh 14 mili giây). Reranker thì ngược lại: nó chấm vài chục cặp mỗi câu hỏi,
    và ở đó GPU nhanh hơn 11,2× (1,68 s xuống 0,15 s cho 12 cặp).

    Ngưỡng tính theo TỔNG VRAM (không phải phần còn trống): quyết định này phải ổn định cho
    cả phiên, mà phần còn trống thì dao động theo việc Ollama vừa nạp hay vừa nhả model. Card
    lớn hơn ngưỡng thì giữ tất cả trên GPU, không cần đánh đổi gì.
    """
    if not co_cuda():
        return False
    return tong_vram_gb() >= config.VRAM_DU_GIU_EMBEDDING_TREN_GPU_GB


def bat_dau_ingestion(embedding_service=None) -> None:
    """Vào giai đoạn INGESTION: đưa embedding trở lại GPU nếu máy có.

    Cần vì ket_thuc_ingestion() có thể đã đẩy nó xuống CPU ở lần build trước; không đưa lại
    thì mọi lần build sau đều chạy embedding trên CPU và mất khoản 13,3× đúng ở chỗ nó có giá
    trị nhất (mã hoá hàng nghìn chunk theo lô).
    """
    if not config.BAT_QUAN_LY_VRAM or embedding_service is None or not co_cuda():
        return
    if config.THIET_BI_EMBEDDING == "auto":
        embedding_service.chuyen_thiet_bi("cuda")


def ket_thuc_ingestion(client=None, embedding_service=None) -> None:
    """Chuyển từ giai đoạn INGESTION sang QUERY: nhả những gì query không cần.

    Gọi ở cuối luồng build index. Ba việc, theo đúng thứ tự:
      1. Nhả model vision - nó chiếm nhiều VRAM nhất và tuyệt đối không dùng ở luồng query.
      2. Trên card CHẬT, đẩy embedding xuống CPU để nhường chỗ cho reranker và LLM - xem
         du_cho_giu_embedding_tren_gpu() để biết vì sao hy sinh đúng model này.
      3. Dọn bộ đệm CUDA của PyTorch, trả phần vừa giải phóng lại cho Ollama.

    Không làm gì khi máy không có GPU: lúc đó không có VRAM nào để tranh, và việc nhả model
    khỏi RAM hệ thống chỉ khiến lượt hỏi đầu tiên phải nạp lại.
    """
    if not config.BAT_QUAN_LY_VRAM or not co_cuda():
        return
    ghi_log_vram("kết thúc ingestion")
    nha_model_ollama(config.VISION_MODEL_NAME, client)
    if embedding_service is not None and config.THIET_BI_EMBEDDING == "auto":
        if not du_cho_giu_embedding_tren_gpu():
            logger.info(
                "VRAM %.1f GB không đủ cho cả LLM + reranker + embedding cùng lúc - chuyển "
                "embedding sang CPU cho giai đoạn truy vấn (mất ~15ms mỗi câu, trả lại ~1,1 GB).",
                tong_vram_gb(),
            )
            embedding_service.chuyen_thiet_bi("cpu")
    don_bo_nho_cuda()
    ghi_log_vram("sau khi nhả vision")
