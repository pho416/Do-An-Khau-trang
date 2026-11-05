# B1/B2 FaceNet fine-tuning (DigiFace protocol, 28/10/2026)

Đặt file `train.py` vào **`training/train.py`** tại project root `DoAn_CV_NangCao/` (tạo thư mục `training/` nếu chưa có). Không thay thế hoặc chạy đè `B0_evaluator_validation.ipynb`. Script này dùng riêng **pipeline A** (resize 160×160) đã chọn tại P5.

## Chuẩn bị

Repo cần có:

- `ban_giao_nhom/protocol/notebook_outputs/tables/05_image_roles_draft.csv`
- `ban_giao_nhom/protocol/notebook_outputs/protocol_draft.json`
- `ban_giao_nhom/quyet_dinh_nhom/input/project_decisions.csv`
- `ban_giao_nhom/code/requirements-baseline.txt`
- Ảnh DigiFace tại đường dẫn tương đối từ root, ví dụ `data/raw/digiface/images_eda/1652/8.png`. `data/` **không đưa lên GitHub**; đồng bộ riêng lên server. Có thể để ảnh ở mount khác và dùng `--data-root` trỏ tới một thư mục *chứa `data/raw/...`*, hoặc symlink `data` vào repo.

Khuyến nghị Python 3.10/3.11. PyTorch/CUDA phải phù hợp với GPU server; môi trường B0 trước đây dùng `torch==2.2.2`, `torchvision==0.17.2`, `facenet-pytorch==2.6.0`, `numpy==1.26.4`, `Pillow==10.2.0`. Nếu dùng CUDA, cài đúng wheel CUDA của PyTorch 2.2.2 trước, rồi cài `ban_giao_nhom/code/requirements-baseline.txt`, sau đó `pip check`.

## Chạy trên Linux server

Chạy trong **project root**:

```bash
# Bước 1: preflight (kiểm file và SHA-256 của 57.600 ảnh train + 8.000 ảnh val)
python training/train.py --mode B1 --preflight-only

# Bước 2: kiểm B0 với cùng pipeline A (không train, không dùng test)
python training/train.py --mode B0 --device cuda --batch-size 32 --eval-batch-size 128

# Bước 3: chạy 2 optimizer steps để kiểm forward/backward/checkpoint (không chọn model)
python training/train.py --mode B1 --device cuda --smoke-steps 2 --run-name smoke_B1
python training/train.py --mode B2 --device cuda --smoke-steps 2 --run-name smoke_B2

# Bước 4: B1 và B2 chính thức, cùng số epoch và batch/step budget
python training/train.py --mode B1 --device cuda --epochs 5 --batch-size 32 \
  --eval-batch-size 128 --num-workers 4 --seed 20261028
python training/train.py --mode B2 --device cuda --epochs 5 --batch-size 32 \
  --eval-batch-size 128 --num-workers 4 --seed 20261028 --occlusion-prob 0.5
```

Hai lệnh B1/B2 khởi tạo cùng trọng số VGGFace2 + cùng head initialization (đã hash trong `config.json`) và cùng thứ tự ảnh/budget optimizer. Khi không có CUDA, dùng `--device cpu` để kiểm luồng; không khuyến khích full train bằng CPU.

Tự chọn `--batch-size` mà GPU đủ bộ nhớ, **nhưng phải dùng cùng giá trị cho B1 và B2**. Nếu thay đổi batch sau khi một model đã train, hãy train lại cả cặp để so sánh công bằng.

## Đầu ra

Mặc định mọi output nằm trong `runs/` (**đã gitignore**):

- `runs/B1_seed20261028/config.json`, `manifest_summary.json`, `environment.json`, `label_map.json`.
- `runs/B1_seed20261028/epoch_NNN_validation.json`, `history.csv`.
- `runs/B1_seed20261028/best.pt` và `last.pt` (encoder, head, optimizer và trạng thái resume); B2 tương tự.
- `runs/B1_seed20261028/best_validation.json`: DIR/FPIR/threshold của epoch tốt nhất.
- `runs/B2_seed20261028/masks/epoch_NNN.csv`: cho **mọi ảnh train**, seed phát sinh, kiểu mask và tọa độ rectangle từng epoch. Ảnh đã che được sinh online, không tạo dataset mới trên ổ đĩa.
- `smoke_B1` / `smoke_B2` chỉ tạo `smoke.pt`, không tính validation và **không phải kết quả đồ án**.

Để tiếp tục một lần train **sau khi chạy xong một epoch**:

```bash
python training/train.py --mode B1 --device cuda --epochs 5 --batch-size 32 \
  --eval-batch-size 128 --num-workers 4 --seed 20261028 \
  --resume runs/B1_seed20261028/last.pt
```

Lưu ý: `best.pt` là **checkpoint training**, không thể truyền trực tiếp cho notebook B0 đang hardcode pretrained. Khi tích hợp evaluator riêng cho B1/B2, khởi tạo encoder `InceptionResnetV1(pretrained=None, classify=False)` rồi nạp `checkpoint['encoder_state_dict']` với `load_state_dict(..., strict=True)`. Script bỏ lớp `logits` 8.631 danh tính VGGFace2 không được dùng trong nhánh embedding nên hai chế độ có cùng keys `state_dict`. **Không chạy lại notebook B0 vào chung output directory để tránh ghi đè B0 gốc.**

## Hợp đồng thí nghiệm

- `role=train` có 57.600 ảnh từ 1.600 người dùng cho gradient training. `train_monitor` (6.400 ảnh) được tách riêng, không được đưa vào training.
- DigiFace validation: 300 gallery + 3.700 known-query + 4.000 unknown-query. `test`, ROF và MFR2 **không được mở** bởi script trong training.
- B1: không thêm ảnh che; B2: che ngẫu nhiên 50% mẫu theo 4 mask hình chữ nhật đen (`eyes`, `lower`, `left`, `right`) và seed xác định theo `(script_version, seed, epoch, image_id)`.
- Loss: cross-entropy với linear head 1.600 lớp. Optimizer: AdamW, LR backbone `1e-5`, LR head `1e-3`, weight decay `1e-4`. Đây là **cấu hình thử nghiệm đề xuất**, chưa phải hyperparameters được nhóm tối ưu.
- Mỗi epoch chọn ngưỡng trên val theo quy tắc gốc: tối đa DIR với empirical FPIR ≤ 5%, khi hòa chọn threshold cao hơn. Chọn checkpoint tốt nhất theo DIR, hòa theo FPIR thấp hơn, rồi epoch sớm hơn.
- Không dựa trên validation để điều chỉnh augmentation/epoch/lr quá nhiều lần; nếu điều chỉnh, ghi lại các thử nghiệm và có nguy cơ overfit validation. Khóa cấu hình trước khi chạy test.
- Script kiểm SHA của các ảnh **train + val** ở preflight mặc định. `--skip-hash-check` bỏ kiểm hash để tăng tốc, chỉ dành cho debug. Trường `protocol_status=draft_unconfirmed` của manifest được giữ nguyên; script **không** thay thế khâu QA xác minh nhãn.

## Giới hạn kiểm chứng

Đã kiểm cú pháp, parsing manifest thật (80.000 ảnh), split 57.600 train/8.000 val, hàm augmentation, tensor hóa RGB/alpha, và chọn threshold bằng dữ liệu tổng hợp. **Chưa chạy fine-tune hoặc tái hiện DIR 57,2% thực tế** vì không có các ảnh DigiFace trên máy tạo code và không có môi trường `facenet-pytorch` cùng GPU để chạy inference.
