# Hướng dẫn thu ảnh webcam (chuẩn bị song song, dùng ở bước 8)

Ảnh webcam **không dùng để train**. Nó dùng cho: (1) chọn lại ngưỡng trên ảnh người thật, (2) test ứng dụng thực tế, (3) demo. Vì cần nhiều người và nhiều buổi, nhóm nên bắt đầu thu sớm.

## Ai và bao nhiêu

- **10–20 người đăng ký** (người quen) + **ít nhất 5–10 người lạ** riêng, không bao giờ đưa vào gallery.
- Người lạ dùng để chọn ngưỡng và người lạ dùng để test phải là hai nhóm khác nhau.
- Xin đồng ý trước: mục đích (đồ án môn học), ai được xem ảnh, khi nào xóa. Không đăng ảnh lên mạng.

## Ba phiên, khác ngày

| Phiên | Dùng cho | Chụp gì |
|---|---|---|
| 1. Đăng ký | Gallery | 5–10 ảnh rõ mặt, nhìn thẳng và hơi nghiêng, ánh sáng bình thường |
| 2. Hiệu chỉnh | Chọn ngưỡng webcam | Ảnh rõ + đeo khẩu trang + đeo kính, vài ánh sáng khác nhau |
| 3. Test | Báo kết quả cuối, demo | Như phiên 2, thêm tay che một phần mặt, quay đầu nhẹ, ánh sáng yếu |

Không dùng ảnh của phiên đăng ký để test. Người lạ chỉ xuất hiện ở phiên 2 hoặc 3.

## Cách lưu

```text
data/raw/webcam/<phien>/<ma_nguoi>/<dieu_kien>_<so_thu_tu>.jpg
  phien:      s1_dang_ky | s2_hieu_chinh | s3_test
  ma_nguoi:   W01, W02, ... (không ghi tên thật trong tên file)
  dieu_kien:  ro | khau_trang | kinh | tay_che | nghieng | toi
```

Kèm một file `data/raw/webcam/nguoi_tham_gia.csv` (chỉ nhóm giữ): `ma_nguoi, vai_tro (known/unknown), phien_tham_gia, dong_y (yes/no), ngay_xoa_du_kien`.

## Lưu ý

- Mỗi khung hình chỉ có **một người** (phạm vi bắt buộc của đề tài).
- Nếu quay video, chỉ trích vài khung mỗi điều kiện; các khung liền nhau gần như giống hệt, không được tính là nhiều mẫu độc lập.
- Ghi lại camera, độ phân giải, ngày giờ từng phiên.
