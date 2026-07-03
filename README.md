# Dự án phân loại bệnh da liễu từ ảnh da liễu (HAM10000)
## Tài liệu Giai đoạn 1: EDA & Tiền xử lý dữ liệu

### 1. Giới thiệu dự án
Mục tiêu: xây dựng mô hình phân loại 7 nhóm bệnh da liễu từ ảnh da liễu soi
(dermoscopic images), sử dụng bộ dữ liệu HAM10000 (Human Against Machine
with 10000 training images).

Nguồn dữ liệu: 
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data

### 2. Giới thiệu dataset
- Gồm 10.015 ảnh dermoscopy + file metadata HAM10000_metadata.csv
- Các cột chính:
  - lesion_id: mã tổn thương (1 tổn thương có thể có nhiều ảnh)
  - image_id: mã ảnh, khớp tên file .jpg
  - dx: nhãn chẩn đoán (biến mục tiêu)
  - dx_type: phương pháp xác nhận chẩn đoán (histo, follow_up, consensus, confocal)
  - age, sex, localization: thông tin nhân khẩu học và vị trí tổn thương
- 7 lớp bệnh (dx):
  - akiec: Actinic keratoses / Bowen's disease (tiền ung thư)
  - bcc:   Basal cell carcinoma (ung thư biểu mô tế bào đáy)
  - bkl:   Benign keratosis-like lesions (tổn thương sừng lành tính)
  - df:    Dermatofibroma (u xơ da lành tính)
  - mel:   Melanoma (u hắc tố ác tính)
  - nv:    Melanocytic nevi (nốt ruồi lành tính) — lớp chiếm đa số
  - vasc:  Vascular lesions (tổn thương mạch máu)
- Cấu trúc thư mục sau khi tải:
  data/
  ├── HAM10000_metadata.csv
  ├── HAM10000_images_part_1/
  └── HAM10000_images_part_2/

### 3. Mục tiêu giai đoạn EDA
- Hiểu phân bố nhãn, mức độ mất cân bằng dữ liệu
- Hiểu phân bố nhân khẩu học (tuổi, giới tính, vị trí tổn thương)
- Phát hiện rủi ro rò rỉ dữ liệu (data leakage) do 1 lesion có nhiều ảnh
- Kiểm tra chất lượng ảnh: kích thước, giá trị thiếu, dữ liệu bất thường
- Từ insight thu được, đề xuất chiến lược tiền xử lý và augmentation phù hợp

### 4. Kế hoạch EDA chi tiết
1. Thống kê tổng quan: shape, dtype, missing values
2. Phân bố lớp dx (bar chart + tỉ lệ %)
3. Phân bố age (histogram), sex (count plot), localization (bar chart)
4. Boxplot age theo từng loại bệnh dx
5. Kiểm tra số ảnh / lesion_id (phát hiện lesion có nhiều ảnh)
6. Kiểm tra kích thước ảnh gốc (thường đồng nhất 600x450)
7. Trực quan hóa mẫu ảnh đại diện cho từng lớp bệnh

### 5. Insight kỳ vọng
- Mất cân bằng lớp nghiêm trọng: nv chiếm khoảng 65-67% tổng số ảnh, 
  trong khi df và vasc chỉ chiếm dưới 2% → tỉ lệ lớp nhiều nhất/ít nhất 
  có thể lên tới ~50-60 lần
- Một phần lesion_id có nhiều hơn 1 ảnh → nếu chia train/test theo ảnh
  (thay vì theo lesion_id) sẽ gây rò rỉ dữ liệu, đánh giá mô hình sai lệch
- Cột age có một số giá trị thiếu, cần xử lý (impute hoặc loại bỏ tùy use-case)
- Kích thước ảnh gốc đồng nhất → không cần xử lý tỉ lệ khung hình phức tạp,
  chỉ cần resize đồng loạt
- Phân bố tuổi và vị trí tổn thương khác nhau rõ giữa các loại bệnh →
  tiềm năng làm feature phụ nếu về sau dùng mô hình đa modal (ảnh + metadata)

### 6. Kế hoạch tiền xử lý dữ liệu
a) Resize
   - Đưa toàn bộ ảnh về kích thước cố định phù hợp backbone CNN dự kiến
     (ví dụ 224x224 cho ResNet/EfficientNet, 299x299 cho Inception)
   - Chuẩn hóa (normalize) pixel về [0,1] hoặc theo mean/std ImageNet nếu 
     dùng transfer learning

b) Data Augmentation
   - Dùng thư viện albumentations (hoặc tf.keras ImageDataGenerator)
   - Phép biến đổi: Horizontal/Vertical Flip, Rotate, Brightness/Contrast,
     Shift-Scale-Rotate, Gaussian Blur nhẹ
   - Chiến lược theo lớp:
     - Lớp đa số (nv): augmentation nhẹ, không cần oversample
     - Lớp thiểu số (df, vasc, akiec, mel): augmentation mạnh hơn + tạo
       thêm bản sao augmented để giảm mất cân bằng
   - Cân nhắc thêm: class weighting / focal loss ở giai đoạn train thay vì
     chỉ dựa vào oversample thủ công

c) Chia tập dữ liệu (Train/Val/Test)
   - Bắt buộc chia theo lesion_id (không chia theo ảnh) để tránh rò rỉ
     dữ liệu, vì cùng 1 lesion có thể có nhiều ảnh
   - Dùng stratified split theo dx để giữ tỉ lệ lớp tương đối đồng đều
     giữa các tập
   - Tỉ lệ đề xuất: 70% train / 15% validation / 15% test

### 7. Công cụ sử dụng
- Xử lý dữ liệu: pandas, numpy
- Trực quan hóa: matplotlib, seaborn
- Xử lý ảnh: Pillow (PIL), opencv-python
- Augmentation: albumentations
- Chia tập dữ liệu: scikit-learn (train_test_split, stratify)

### 8. Bước tiếp theo (Giai đoạn 2 — Modeling)
- Xây dựng data pipeline (Dataset/DataLoader hoặc tf.data) áp dụng resize
  và augmentation đã thử nghiệm ở giai đoạn EDA
- Lựa chọn kiến trúc CNN (transfer learning: EfficientNet/ResNet/DenseNet)
- Áp dụng class weighting hoặc focal loss để xử lý mất cân bằng
- Đánh giá bằng metric phù hợp cho bài toán mất cân bằng: F1-score theo
  lớp, confusion matrix, ROC-AUC macro/micro — không chỉ dựa vào accuracy
