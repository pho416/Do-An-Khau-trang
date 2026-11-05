# Nguồn dữ liệu fine-tune: kiểm tra và thu thập ngày 21/10/2026

## Quyết định nguồn

1. Ưu tiên CASIA-WebFace từ trang dataset chính thức của InsightFace:
   https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md
   (link Drive ID `1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l`). HEAD trả tên
   `faces_webface_112x112.zip`, kích thước 2,787,480,599 bytes; **GET thực tế trả
   trang HTML “Quota exceeded”**. Không có ảnh CASIA nào được tải thành công.
   Bằng chứng nằm trong `train_source_casia_quota_response.html`.
   README InsightFace cho biết các training datasets được đóng gói MXNet binary;
   đây không phải nguồn ảnh đầy đủ độ phân giải để chạy EDA detector gốc.
   Không đổi đuôi HTML thành ZIP, không báo CASIA đã thu thập và không dùng mirror
   không rõ nguồn. Chính sách InsightFace: training data và pretrained models dùng
   cho nghiên cứu phi thương mại; không suy diễn MIT của code sang dữ liệu.

2. Nguồn dự phòng đã chọn: **Microsoft DigiFace-1M**, part
   `subjects_0-1999_72_imgs.zip`, liên kết trực tiếp từ:
   https://github.com/microsoft/DigiFace1M
   https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip
   Server xác nhận kích thước 2,938,553,163 bytes, hỗ trợ byte ranges.
   Nguồn công bố là 2,000 synthetic identities × 72 ảnh/identity cho part này.
   Số lượng thực tế sau kiểm tra nằm trong `train_source_status.json`.

**DigiFace là ảnh khuôn mặt render tổng hợp từ tác giả; không phải ảnh người thật
thu thập trên web, và không phải ảnh do nhóm/AI tự sinh thay thế dữ liệu.** Không
gộp nó với ROF/MFR2 dưới một loại “ảnh thực”. Mục đích là có một tập train có danh
tính nhất quán, khả năng tải và điều kiện dùng rõ; kiểm chứng khả năng chuyển miền
sang che khuất thật phải dựa trên tập real riêng. Cần nêu domain gap synthetic → real
trong báo cáo. Nếu nhóm yêu cầu fine-tune chỉ trên ảnh người thật, cần lấy CASIA từ
nguồn được cấp quyền sau khi hết quota trước khi chốt thí nghiệm B1/B2.

## Điều kiện sử dụng và chia sẻ

- DigiFace áp dụng Research Use of Data Agreement v1.0 (R-UDA), computational use
  cho nghiên cứu phi thương mại; không bán dữ liệu/quyền dùng. Khi phân phối lại phải
  giữ attribution và ràng buộc người nhận theo R-UDA. Xem toàn văn đã lưu tại
  `train_source_digiface_LICENSE.txt`, nguồn:
  https://github.com/microsoft/DigiFace1M/blob/main/LICENSE
- Gửi tài liệu/script/hướng dẫn tải cho thành viên; khi chia sẻ dữ liệu nội bộ,
  gửi kèm LICENSE và attribution. Không đưa toàn bộ dữ liệu vào ZIP tài liệu/code
  hoặc repository Git công khai theo mặc định.
- Cite: Gwangbin Bae et al., *DigiFace-1M: 1 Million Digital Face Images for Face
  Recognition*, WACV 2023. Bản gốc:
  https://microsoft.github.io/DigiFace1M/

## Nội dung thu thập và ý nghĩa integrity

- Archive gốc được giữ tại `data/raw/digiface/subjects_0-1999_72_imgs.zip`.
- Thư mục `data/raw/digiface/images_eda/` chứa tối đa 40 ảnh gốc/identity, chọn bằng
  seed `20261021` và SHA-256 của `seed/identity`; không đổi pixel, không resize,
  không crop, không augmentation, không chia train/val/test. Các byte ảnh giữ
  nguyên từ archive. Đây là **lấy mẫu cho EDA**, không phải bước tiền xử lý.
- Archive inventory liệt kê toàn bộ member với kích thước, CRC và loại image;
  identity counts tính trên toàn part, không suy rộng thành toàn DigiFace-1M.
- Selected manifest có đường dẫn nguồn, identity, member trong ZIP và SHA-256
  cho từng ảnh lấy mẫu. Kiểm tra CRC của các member được đọc; chưa tự khẳng định
  CRC cho các ảnh không lấy mẫu.
- SHA-256 archive là fingerprint tính tại máy sau tải; tác giả không cung cấp
  checksum đã xác thực trong nguồn kiểm tra. Không gọi SHA local là publisher hash.
- Dữ liệu thuộc part cố định IDs 0–1999 nên lấy mẫu 2,000 identities theo shard,
  **không phải mẫu ngẫu nhiên toàn bộ 110,000 identities DigiFace-1M**. Chọn 40/72
  ảnh/ID có thể không giữ đủ mọi accessories condition; ảnh không có nhãn occlusion
  tự động đáng tin cậy. Không dùng chỉ số ảnh để suy ra nhãn mask/glasses khi chưa
  xác minh bằng metadata/quan sát.

## Tái lập

Python >=3.10, chỉ dùng thư viện chuẩn. Tại thư mục `Đồ Án`:

```powershell
python data_acquisition/train_source_acquire.py --workers 8 --per-identity 40 --seed 20261021
```

Tải chia các range 64 MiB, tiếp tục từ phần đã có khi gián đoạn. Tệp `.part`
không được coi là download hoàn chỉnh. Sau ghép, kiểm tra số bytes và chữ ký ZIP,
rồi sinh inventory, trích ảnh, SHA-256 và báo cáo trạng thái. Có thể giảm
`--workers 2` khi đường truyền chung hạn chế.

Chỉ tạo lại inventory/extraction từ archive hiện có:

```powershell
python data_acquisition/train_source_acquire.py --skip-download --per-identity 40 --seed 20261021
```

Không thay seed hoặc per-identity trong thư mục đã có subset nếu chưa dọn/đổi output:
script không xóa ảnh cũ, và thư mục có thể chứa ảnh từ lần lấy mẫu trước. Manifest
mới là danh sách chính xác cho lần chạy; nên dùng cấu hình cố định trên để tái lập.

Sau EDA mới quyết định: nguồn train cuối; cách xử lý crop synthetic có sẵn; việc
resize tới input 160×160 của InceptionResnetV1; alignment trên dữ liệu thực; loại
ảnh không dùng; cách giữ danh tính tách biệt; augmentation và tỷ lệ che. Dữ liệu
đã crop nhỏ không cung cấp bằng chứng về detection trên ảnh bối cảnh rộng.

## Các snapshot nguồn

README/License tải từ repository tác giả được lưu với URL/bytes/SHA-256 trong
`train_source_source_snapshots.json`. Đây là snapshot ngày thu thập, không phải
commit-pinned release. Nội dung nguồn có thể thay đổi; báo cáo dùng snapshot đã lưu.
