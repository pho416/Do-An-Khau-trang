# Chia tập và protocol đánh giá

Mở `Chia_tap_va_protocol_danh_gia.ipynb`. Notebook có code tạo split/role thật từ QA, Markdown, biểu đồ, ảnh minh họa và output đã chạy; không cần chạy lại để đọc.

Trạng thái **draft_unconfirmed**, `frozen=false`. Đã có danh sách cụ thể để kiểm tra và phát triển loader, chưa thay xác nhận của nhóm về nhãn, gallery hoặc preprocessing. Không có recognition/training hay raw mutation ở bước này.

## Thiết kế

- DigiFace: 1.600 / 200 / 200 identity train/validation/test. Mỗi identity train có 36 ảnh train và 4 ảnh monitor trong dữ liệu hiện tại.
- Mỗi validation/test: 100 known, 100 unknown; known có 3 ảnh gallery nền và 37 query/người, unknown có 40 query/người.
- Gallery DigiFace chưa được gọi là clean; nhãn điều kiện vẫn là `synthetic_unannotated`.
- Cách ly hold/secondary theo QA trong draft, giữ cờ review chất lượng/detector. Không xóa/chuyển ảnh.
- ROF/MFR2 dành ngoài tuning; chỉ tính inventory, gallery feasibility, alias guard và kiểm pairs ở bước này.

## Chạy lại

Chọn kernel môi trường CV Python 3.11, mở trong cấu trúc dự án rồi Restart Kernel → Run All. Có thể đặt biến môi trường `CV_PROJECT_ROOT` nếu mở từ nơi khác. Notebook không truy cập mạng hoặc chạy model. Cần dữ liệu raw cho phần minh họa và file pairs nguồn.

Trước khi chạy lại sau thay đổi review: cập nhật input `../qa/group_decisions.csv` hoặc các log review nguồn, chạy QA, rồi chạy notebook protocol. Không sửa tay CSV trong notebook_outputs. Tăng protocol version khi đổi chính sách; không giữ ngưỡng cũ mặc định khi đổi gallery/preprocessing.

## Đầu ra

`notebook_outputs/protocol_draft.json` chứa cấu hình, seed, hash nguồn và hợp đồng evaluator. `tables/05_image_roles_draft.csv` chứa vai trò từng ảnh; `06_leakage_assertions.csv` và `06_negative_tests.csv` lưu kiểm chứng; các bảng `07_*` dành cho dữ liệu thật. `execution_summary.json` ghi phạm vi tính mới và những bước chưa chạy.

Bước kế tiếp là preprocessing thí điểm trên validation dự thảo, đối chiếu direct-resize/redetect-crop, tensor và failure. Chốt gallery và bằng chứng semantic trước kết quả đánh giá xác nhận; giữ nguyên external test ngoài tuning.
