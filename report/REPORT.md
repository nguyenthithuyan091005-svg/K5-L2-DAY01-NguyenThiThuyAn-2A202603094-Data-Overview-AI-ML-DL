# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026**

**Runtime Colab:** GPU      T4

**Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

class_id = 468

class_name = cab

rank = 1

score = 0.510915

taxonomy_name = ImageNet-1K


2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
  
Record này nói về cả bức ảnh, không phải một vật thể riêng.

Vì sao: Task trong JSON là image_classification, nên model chỉ gán lớp và score cho toàn bộ ảnh. Record chỉ có class_id, class_name, rank, score và taxonomy_name, không có bounding box, mask hay ID của từng đối tượng nên không thể biết cab ứng với vật nào ở đâu. Ảnh traffic có rất nhiều xe buýt (xanh lá, cam, đỏ) cùng vài xe con, nhưng model vẫn xếp cab hạng 1 với score 0.510915. Vì vậy cab đứng đầu không có nghĩa mọi xe trong ảnh đều là taxi và nhãn này chỉ là dự đoán của model chứ chưa phải ground truth.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
  
Danh sách tên lớp thuộc taxonomy ImageNet-1K, đi kèm checkpoint phân loại đã huấn luyện, mô hình không tự nghĩ ra một nhãn mới ngoài bộ lớp được định nghĩa. class_id là mã ánh xạ tới tên lớp trong taxonomy tương ứng. 

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
  
class_id hỗ trợ máy xử lý và đối chiếu nhất quán; class_name giúp con người hiểu lớp; taxonomy_name xác định bộ nhãn và ngữ cảnh giải nghĩa. Một tên lớp hoặc mã số có thể được sử dụng khác nhau giữa những taxonomy nên chỉ ghi tên là chưa đủ truy xuất và kiểm tra.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
  
Phải quy định rõ mục tiêu nhãn ở cấp toàn ảnh (chủ thể chính, nội dung nổi bật hay loại cảnh), tiêu chí xác định chủ thể chính (diện tích, mức độ nổi bật và mục đích tác vụ), cách xử lý khi có nhiều đối tượng cùng nổi bật và khi nào gắn cờ không rõ/đưa reviewer quyết định. Một ảnh chỉ được chọn một lớp khi tác vụ thật sự quy định single-label classification.

6. Vì sao không được lấy `score` của model làm ground truth?
  
score = 0.510915 là mức độ mô hình ưu tiên lớp cab trong dự đoán; đây không phải xác suất chắc chắn nhãn đúng, cũng không phải điểm đánh giá chất lượng của người gán nhãn. Ground truth phải được xác lập bằng quan sát, taxonomy và guideline, sau đó kiểm tra độc lập nếu cần.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

class_id = 0; class_name = person; score = 0.912625 (hiển thị làm tròn 0.91 trên PNG).
bbox_xyxy = [385.33, 69.24, 498.92, 348.92] (pixel).
bbox_width = 113.58 px; bbox_height = 279.68 px.
taxonomy_name = COCO-80; image_width = 640; image_height = 427

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Gốc tọa độ (0, 0) nằm tại góc trên-trái của ảnh, trục x tăng sang phải và trục y tăng xuống dưới. Box có góc trên-trái tại (385.33, 69.24) và góc dưới-phải tại (498.92, 348.92), bao quanh người mặc áo trắng đứng gần thiết bị bên phải. Chiều rộng được JSON ghi là 113.58 px, chiều cao 279.68 px (giá trị chiều rộng tính từ tọa độ đã làm tròn có thể chênh 0.01 px).

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']

threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi hạ ngưỡng, reviewer cần kiểm tra thêm các hộp score thấp: loại false positive, hộp trùng, hộp sai lớp và đánh giá các vật thể dễ bị bỏ sót. Số prediction tăng không có nghĩa số vật thể thật tăng. Một hộp person với score 0.610931 chỉ khoanh phần người bị cắt ở mép trái; trường hợp này cần phân biệt người xuất hiện một phần với dự đoán sai.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Gán một hộp chữ nhật sát nhất có thể quanh phần nhìn thấy của từng vật thể thuộc taxonomy, hạn chế đưa nền vào hộp. Với đối tượng bị cắt mép, hộp dừng tại ranh giới ảnh; với đối tượng bị che, chỉ dùng phạm vi có thể quan sát theo guideline, không tự ước đoán phần khuất. Mỗi đối tượng có nhãn lớp riêng, tọa độ nằm trong giới hạn ảnh; các trường hợp không xác định được lớp hoặc hộp cần gắn cờ để hỏi Lab Coach.


6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Ở góc dưới-trái của detection_predictions.png, chỉ thấy một phần bàn tay/cánh tay của người bị cắt bởi mép ảnh, trong khi phần lớn cơ thể nằm ngoài khung hình và các vật trên bàn che thêm. Model vẫn dự đoán person với score 0.610931, bbox [0.12, 263.18, 61.55, 310.78]. Đây là trường hợp vừa bị cắt mép (truncation) vừa bị che (occlusion).

Quyết định: đưa lên Lab Coach (escalation), chưa tự viết thành quy tắc. Vì đây là ca người chỉ lộ một phần rất nhỏ mà guideline hiện tại chưa nêu mức nhìn thấy tối thiểu để được gán nhãn person. Nếu mỗi annotator tự quyết thì có người gán, có người bỏ, làm nhãn thiếu nhất quán. Sau khi Lab Coach trả lời, ngưỡng đó mới được ghi vào guideline để áp dụng cho mọi ảnh.

Ngoài ra, các hộp bowl nhỏ (score 0.70 và 0.46) nằm ở vùng giá treo nồi/chảo phía trên, reviewer phải đối chiếu từng hộp với vật thật để phát hiện khả năng nhầm lớp, không kết luận đúng chỉ dựa vào score.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

instance_id = kitchen-001; class_id = 0; class_name = person.
score = 0.899318; polygon_point_count = 348.
Năm điểm đầu trong polygon_xy: [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]].
bbox_xyxy = [385.45, 66.44, 498.02, 348.58] (pixel); taxonomy_name = COCO-80.

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Bounding box chỉ cung cấp hình chữ nhật bao quanh, bao gồm cả các pixel nền nằm bên trong hộp. Polygon gồm nhiều điểm nối thành đường biên gần với hình dạng đối tượng, nhờ đó thể hiện vùng chiếm chỗ của đối tượng chính xác hơn, đặc biệt khi vật có dạng không đều. Tuy vậy, polygon do model dự đoán vẫn có thể lệch biên và cần kiểm tra.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

Đây là mã định danh một đối tượng cụ thể trong tập prediction đã xuất. Hai vật cùng lớp (ví dụ hai bowl là kitchen-002 và kitchen-003) vẫn phải có instance_id và mask riêng. instance_id khác class_id (mã loại vật) và không phải tracking_id (mã theo dõi cùng một vật qua nhiều khung hình/video).

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Quy tắc vẽ mask đề xuất: Bám sát đường biên phần thực sự nhìn thấy của từng vật thể, loại bỏ vùng nền và không lấy pixel thuộc vật khác. Mỗi instance có mask riêng dù cùng class; không kéo mask xuyên qua vùng che khuất chỉ bằng suy đoán. Với mép mờ, phần trong suốt, lỗ rỗng hay hai vật chạm nhau, ưu tiên quy tắc nhất quán của guideline; những trường hợp vượt quá quy tắc thì gắn cờ để reviewer/Lab Coach quyết định.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Trên segmentation_prediction.png, vùng góc trái dưới có nhiều mask chồng lấn: mask oven 0.49 (kitchen-007, bbox [0.25, 188.95, 194.35, 294.62]) trùm lên mask person 0.43 (kitchen-009, bbox [0.3, 263.27, 59.5, 309.92]) và mask dining table 0.58 (kitchen-005, bbox [0.97, 234.74, 345.86, 423.87]). Vùng này còn có nhiều bowl sát nhau (kitchen-003, kitchen-006…) và bàn tay của người bị cắt mép. Các biên ở đây chồng và không tách được.

Quyết định: escalation. Biên giữa lò, mặt bàn và người ở đây không thể xác định rõ bằng mắt nên một quy tắc vẽ mask chung không đủ để chọn đúng. Annotator cần Lab Coach quyết định một cách xử lý thống nhất (ví dụ ưu tiên vật nằm phía trước hay để vùng chồng không thuộc mask nào) rồi mới ghi vào guideline.

Một ca khác cần reviewer chú ý: mask potted plant 0.63 (kitchen-004) phủ lên bó rau thơm treo ở góc trên-trái, nhiều khả năng là nhầm lớp chứ không phải lỗi biên. Reviewer sẽ kiểm tra đúng lớp trước rồi mới kiểm tra biên.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn lớp ở cấp toàn ảnh nếu guideline quy định single-label; có mã/tên lớp và taxonomy.|traffic chứa nhiều xe, nhưng model xếp cab hạng 1 với score 0.510915; nhãn toàn ảnh dễ mơ hồ khi có nhiều chủ thể.|Quan sát toàn ảnh, đối chiếu mục tiêu và tiêu chí chủ thể chính, gán nhãn theo taxonomy; gắn cờ trường hợp khó.|Kiểm tra nhãn có mô tả đúng toàn ảnh không, có nhất quán guideline và taxonomy không; không dựa độc nhất vào score.|
| Phát hiện vật thể |Mỗi đối tượng một class và bounding box xyxy theo pixel.|Trên PNG kitchen, một người chỉ lộ một phần ở mép trái (person, score 0.610931; bbox [0.12, 263.18, 61.55, 310.78]); nhiều hộp bowl nhỏ gần khu vực treo dụng cụ cần kiểm tra nhầm lớp.|Vẽ hộp chặt, gán lớp, tìm vật bị bỏ sót, ghi chú trường hợp bị che/cắt mép theo guideline.|Kiểm tra đúng lớp, đủ vật, hộp không quá rộng/hẹp, không trùng và các ca che khuất được xử lý thống nhất.|
| Instance segmentation |Mỗi đối tượng có class, instance_id và polygon/mask riêng.|PNG cho thấy dining table (kitchen-005, 598 điểm polygon) có mask trải rộng cả mặt bàn và phần cấu trúc bên dưới, cần rà phần giáp đồ vật khác. Mask oven 0.49 chồng lên person 0.43 ở góc trái dưới, và mask potted plant 0.63 phủ lên bó rau thơm treo (nhiều khả năng nhầm lớp), đều là ca cần QC.|Vẽ biên theo phần nhìn thấy, tách từng instance, không tô vùng của vật khác; chuyển ca khó khi cần.|Kiểm tra mask sát biên, không nhập hai vật làm một, không bỏ vùng đối tượng; kiểm tra tính nhất quán giữa các instance.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Quy tắc bảo vệ dữ liệu: Chỉ sử dụng ba ảnh COCO công khai được chỉ định, giữ thông tin nguồn/giấy phép và không đưa ảnh cá nhân, dữ liệu nội bộ, thông tin định danh hoặc dữ liệu nhạy cảm vào notebook, báo cáo và repository.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Nếu gặp dữ liệu ngoài phạm vi: Dừng xử lý, không tải lên hoặc chia sẻ thêm, và báo Lab Coach/giảng viên để nhận hướng dẫn trước khi tiếp tục.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
