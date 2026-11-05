# Preprocessing thí điểm trên validation (bước 4)

Mở `Preprocessing_thi_diem_validation.ipynb` để xem đầy đủ code, hình và diễn giải. Notebook đã chạy từ kernel mới, 13/13 cell code, không lỗi.

## Câu hỏi

Hai cách chuẩn bị ảnh có tạo đúng đầu vào cho encoder (tensor RGB `float32[3,160,160]`, chuẩn hóa `(pixel − 127,5)/128` đúng một lần) và giữ đủ mẫu số không?

- **A – resize thẳng:** ảnh DigiFace đã crop sẵn, resize về 160 × 160.
- **B – chạy lại MTCNN:** tìm mặt, chỉ chấp nhận khi có đúng một mặt, crop theo bbox rồi resize về 160 × 160.

Chỉ dùng ảnh DigiFace **validation**; không dùng train/test, ROF hay MFR2; chưa chạy nhận diện.

## Kết quả

| | Mẫu chính (300 ảnh) | Mẫu stress (8 ảnh từng lỗi detector ở EDA) |
|---|---|---|
| A chuẩn bị được | 300/300 | 8/8 |
| B có đúng một mặt | 296/300 (3 không thấy mặt, 1 nhiều mặt) | 0/8 (cả 8 không thấy mặt) |
| Gallery đủ 3/3 ảnh (20 người) | A: 20, B: 20 | — |

- 604 tensor (A + B) đều đúng quy ước; 9 phép thử lỗi cố ý (chia 255, chuẩn hóa hai lần, đảo BGR, cache sai, hash sai…) đều bị bắt.
- 272/296 bbox của B vượt khung ảnh, đúng như EDA (ảnh DigiFace đã crop sát).
- A và B cho ảnh khác nhau khá nhiều: trung vị sai khác 32,7/255 mỗi pixel.
- Ca B thất bại là mặt tối, nghiêng hoặc bị vành mũ che, đúng loại ảnh khó của đề tài.

**Đề xuất:** dùng A cho DigiFace; dùng B (có detector) cho webcam và ảnh toàn cảnh. Quyết định cuối phải dựa trên B0 + evaluator ở bước 5, so trên cùng mẫu và cùng mẫu số.

## Tệp đầu ra

- `notebook_outputs/tables/04_per_image_preprocessing.csv`: mọi ảnh, trạng thái A/B, bbox, lỗi, thời gian.
- `notebook_outputs/tables/06_*`, `07_*`, `08_*`: tỷ lệ theo vai trò, đăng ký gallery 3/3, so sánh A/B.
- `notebook_outputs/figures/09_raw_direct_redetect_examples.png`: ảnh gốc / A / B cho các ca lỗi và ca khác biệt nhiều.
- `notebook_outputs/preprocessing_candidate_config.json`: cấu hình ứng viên có version (chưa khóa).
- `notebook_outputs/pilot_prepared_pixels.npz`: pixel RGB uint8 của pilot, nạp bằng `allow_pickle=False` rồi chuẩn hóa.

## Ghi chú về lần sửa

Bản script của Codex (`_cong_cu/build_preprocessing_notebook.py`) có lỗi cú pháp nên chưa từng chạy. Sau khi sửa cú pháp, lần chạy đầu cho thấy B "thất bại" 297/308 ảnh. Nguyên nhân là facenet-pytorch 2.6 trả bbox dạng `dtype=object` khiến `np.isfinite` lỗi; đây là bug code, không phải đặc điểm dữ liệu. Đã sửa, và thêm kiểm tra bắt buộc "0 lỗi phần mềm" để lỗi tương tự không bị che thành ảnh lỗi nữa.
