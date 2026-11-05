# Cài môi trường và chạy lại notebook

Tổng quan dự án và tiến độ nằm ở `README.md` tại thư mục gốc. File này chỉ hướng dẫn cài đặt và chạy lại.

## Đọc trước khi chạy

Sáu notebook (EDA, QA, protocol, quyết định nhóm, preprocessing, đánh giá B0) đã lưu sẵn output thật. Chỉ cần mở ra là đọc được; **không cần chạy lại** nếu chỉ để xem kết quả. Split/protocol vẫn là bản dự thảo, chưa khóa chính thức.

Muốn chạy lại phải có ảnh gốc (`data/raw/`, `data/benchmark_views/`, khoảng 6 GB), vì ZIP không chứa ảnh.

## Cài môi trường (Windows, GPU NVIDIA)

Đứng tại thư mục gốc của dự án (thư mục chứa `ban_giao_nhom/` và `data/`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r ban_giao_nhom/requirements-eda-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements-eda-lock.txt` ghi đúng phiên bản thư viện của máy đã chạy (Windows, Python 3.11.9, CUDA 12.1, RTX 4060 Laptop). Máy chỉ có CPU hoặc dùng hệ điều hành khác thì thay hai gói torch/torchvision cho phù hợp; chạy detector trên CPU sẽ chậm hơn.

## Chạy lại

1. Mở notebook trong VSCode/Jupyter, chọn kernel Python của `.venv` vừa tạo.
2. **Restart Kernel → Run All.** Chạy theo thứ tự EDA → QA → protocol → quyết định nhóm → preprocessing → đánh giá B0. Mỗi notebook đọc bảng của notebook trước; protocol sẽ dừng nếu input quyết định/review đã thay đổi mà chưa chạy lại QA. Preprocessing chỉ đọc ảnh DigiFace validation theo `protocol/notebook_outputs/tables/05_image_roles_draft.csv`.
3. Notebook tự tìm thư mục gốc của dự án, không phụ thuộc ổ đĩa.

Các tùy chọn đầu notebook:

- `RUN_DOWNLOAD=False` (EDA): không tải lại dữ liệu. Code tải vẫn có trong notebook; chỉ bật khi cần tải từ đầu (nhiều GB, một số link ROF có thể đã chết).
- `PIXEL_CACHE_POLICY` / `DETECTOR_CACHE_POLICY` (EDA), `REUSE_SCAN_CACHE` (QA): dùng lại kết quả quét khi dữ liệu không đổi. Máy mới không có cache sẽ tự tính lại; lần đầu có thể mất nhiều phút.

Notebook tự cập nhật `eda/notebook_outputs/execution_summary.json`, `qa/notebook_outputs/qa_execution_summary.json` và bảng provenance/cache tương ứng. Các tệp `eda/notebook_execution_validation.json` và `qa/qa_notebook_execution_validation.json` ghi bằng chứng từ runner của một lượt chạy cụ thể; chạy bằng giao diện VSCode/Jupyter không tự cập nhật chúng.

**Kernel mới và quét mới là hai việc khác nhau.** Restart Kernel → Run All chạy lại logic theo thứ tự, nhưng có thể dùng cache khớp. EDA ghi `cached_before_run`/`newly_computed` trong cache metadata; QA ghi `matching_cache`/`computed_now` trong `qa/notebook_outputs/tables/19_scan_provenance.csv`. Nếu cần quét lại, đặt cả hai chính sách EDA thành `recompute` và `REUSE_SCAN_CACHE=False` ở QA. Giữ provenance để biết lần chạy nào thực sự tính mới.

## Ghi quyết định QA của nhóm

Dùng `qa/group_decisions.csv` làm input quyết định theo ảnh; đây là tệp do nhóm quản lý, notebook chỉ đọc và không ghi đè. Schema và quy tắc điền nằm ở `qa/README_QA.md`. Không điền quyết định vào manifest trong `notebook_outputs/`, vì đó là output được dựng lại. Header-only không xác nhận ảnh nào; quyết định ghi nhận không tự sửa raw/nhãn/split.

## Thư mục đầu ra

Notebook protocol tự ghi `protocol/notebook_outputs/execution_summary.json`; `protocol_draft.json` lưu cấu hình và hash với `frozen=false`. Các bảng vai trò ảnh/kiểm rò rỉ và hình nằm trong `protocol/notebook_outputs/tables/` và `figures/`.

Notebook preprocessing ghi `preprocessing/notebook_outputs/execution_summary.json`, cấu hình ứng viên `preprocessing_candidate_config.json` và pixel pilot `pilot_prepared_pixels.npz` (RGB uint8, chuẩn hóa khi nạp). Cache detector nằm trong `preprocessing/notebook_outputs/cache/` và chỉ được dùng khi dấu vân tay khớp; `cache_use_summary.json` ghi lượt chạy dùng cache hay tính mới. Notebook dừng nếu có lỗi phần mềm ở bước đọc ảnh/detector, để lỗi code không bị ghi nhầm thành ảnh không dùng được.

Notebook đánh giá B0 (`danh_gia_b0/`) đọc 8.000 ảnh DigiFace validation, tính embedding cho hai cách chuẩn bị ảnh (lần đầu khoảng 3–4 phút trên RTX 4060) và lưu vào `danh_gia_b0/notebook_outputs/cache/`; Run All sau chỉ dùng cache khi dấu vân tay khớp. Trọng số VGGFace2 (~107 MB) được tải tự động ở lần chạy đầu trên máy mới. Kết quả chính ở `danh_gia_b0/notebook_outputs/tables/04_operating_point_fpir5.csv` và `b0_validation_thresholds.json`.

- `eda/notebook_outputs/tables/`, `qa/notebook_outputs/tables/`: bảng CSV do notebook sinh ra (manifest, thống kê, hàng đợi QA, bảng quyết định).
- `eda/notebook_outputs/figures/`, `qa/notebook_outputs/figures/`: hình PNG/SVG dùng cho slide.
- `qa/notebook_outputs/contact_sheets/`: các trang ảnh để duyệt bằng mắt.
- `*/notebook_outputs/cache/`: kết quả quét để chạy lại nhanh; không có trong ZIP.

## Script baseline B0

`code/face_occlusion_baseline_compare.py` so một ảnh tham chiếu với nhiều ảnh query bằng B0 và xuất cosine. Cách cài và chạy xem `code/README_BASELINE.md`. Code B0 tương tự cũng có trong notebook EDA (phần 19).

## Dựng lại PDF hướng dẫn

Sửa `huong_dan_trien_khai_hoan_chinh.md` rồi chạy `python ban_giao_nhom/build_guide_pdf.py` (cần `reportlab`, font Arial của Windows).
