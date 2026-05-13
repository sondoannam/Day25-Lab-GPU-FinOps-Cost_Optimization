# Báo Cáo Phân Tích: GPU FinOps & Cost Optimization Lab

- Họ tên: Đoàn Nam Sơn
- MSV: 2A202600045

## 1. Giới thiệu

### Mục tiêu của bài lab
Bài lab nhằm mục đích trang bị các kiến thức và kỹ năng thực tế về quản lý chi phí (FinOps) đối với hệ thống GPU cluster. Thông qua các kịch bản mô phỏng và thực tế, báo cáo này đánh giá khả năng theo dõi, quản lý vòng đời tài nguyên, cũng như ứng dụng các chiến lược tối ưu hóa để giảm thiểu lãng phí và tối đa hóa hiệu suất sử dụng GPU.

### Tổng quan về GPU FinOps
GPU FinOps là sự kết hợp giữa tài chính (Finance) và vận hành (Operations) chuyên biệt cho hệ thống GPU. Khác với CPU, GPU là một tài nguyên cực kỳ đắt đỏ. Việc thiếu kiểm soát trong quy trình cấp phát, theo dõi mức độ sử dụng (utilization), hoặc không khai thác các chiến lược như Spot Instances, Auto-scaling và Mixed Precision Training (AMP) có thể dẫn đến việc lãng phí ngân sách khổng lồ. 

---

## 2. Phân tích từng phần

### Part 1-7: Phân tích kết quả từ mock cluster

**1. Cluster monitoring insights**
Quá trình giám sát cluster cung cấp cái nhìn tổng quan về trạng thái của các GPU nodes. Thông qua các báo cáo metrics, chúng ta có thể dễ dàng xác định được node nào đang chạy, số lượng GPU khả dụng, và tỷ lệ phần trăm tài nguyên đang được tiêu thụ.
![Cluster Metrics](./screenshots/part1_cluster_metrics.png)
![Cluster Monitoring](./screenshots/part1_cluster_monitoring.png)

**2. Cost tracking observations**
Việc theo dõi chi phí (Cost Tracking) ở mức workload cho phép gán chi phí chính xác cho từng tác vụ hoặc team. Kết quả từ việc submit các workload cho thấy hệ thống có khả năng tự động cập nhật chi phí billing dựa trên thời gian thực hiện và loại instance được cấp phát.
![Workload Submission](./screenshots/part2_workload_submission.png)
![Billing Summary](./screenshots/part2_billing_summary.png)

**3. Spot instance savings analysis**
Spot Instances cung cấp mức giá chiết khấu cực kỳ lớn (thường lên đến 70-90% so với On-Demand). Khi request Spot instances, chi phí vận hành giảm đi đáng kể. Tuy nhiên, phân tích từ việc mô phỏng Spot Preemption (thu hồi tài nguyên) cho thấy các workload cần được thiết kế theo hướng Fault-Tolerant (ví dụ: liên tục checkpointing) để đảm bảo không mất mát tiến độ khi bị gián đoạn.
![Spot Pricing](./screenshots/part3_spot_pricing.png)
![Spot Request](./screenshots/part3_spot_request.png)
![Spot Preemption](./screenshots/part3_spot_preemption.png)

**4. Autoscaling behavior**
Autoscaling đóng vai trò thiết yếu trong việc xử lý khối lượng công việc linh hoạt. Thông qua việc phân tích policy và 5 chu kỳ đánh giá (evaluation cycles), hệ thống autoscaler đã tự động tăng/giảm số lượng node để đáp ứng số lượng job đang chờ, giúp tránh tình trạng idle nodes gây lãng phí chi phí.
![Autoscaler Policy](./screenshots/part4_autoscaler_policy.png)
![Autoscaler Evaluation](./screenshots/part4_autoscaler_evaluation.png)

**5. Waste analysis và recommendations**
Waste analysis report chỉ ra rõ các vùng lãng phí ngân sách lớn nhất, thường là từ các instance có tỷ lệ utilization thấp (<30%) hoặc bị quên không tắt. Hệ thống Recommendations đã gợi ý các phương án hữu ích như: Right-sizing (thu nhỏ cấu hình), chuyển sang Spot, hoặc Stop các instances không hoạt động.
![Cost Snapshots](./screenshots/part5_cost_snapshots.png)
![Waste Report](./screenshots/part5_waste_report.png)
![Recommendations](./screenshots/part5_recommendations.png)
![Dashboard](./screenshots/part5_dashboard.png)
![Cost Breakdown](./screenshots/part6_cost_breakdown_viz.png)
![Time-series Viz](./screenshots/part6_timeseries_viz.png)
![Full Workflow](./screenshots/part7_full_workflow.png)

---

### Part 8: Phân tích real GPU training

**1. FP32 vs Mixed Precision (AMP) comparison**
Khi huấn luyện mô hình thực tế, việc sử dụng Automatic Mixed Precision (AMP) cho thấy sự ưu việt vượt trội so với độ phân giải tiêu chuẩn FP32. Bằng cách sử dụng FP16 cho các phép toán tensor core, thời gian huấn luyện giảm đáng kể trong khi bộ nhớ VRAM được giải phóng, cho phép tăng batch size.
![GPU Detection](./screenshots/part8_gpu_detection.png)
![Metrics Diagnostic](./screenshots/part8_metrics_diagnostic.png)
![FP32 Summary](./screenshots/part8_fp32_summary.png)
![AMP Summary](./screenshots/part8_amp_summary.png)

**2. Cost savings achieved**
Báo cáo so sánh chi phí chỉ ra rằng: việc rút ngắn thời gian huấn luyện (bằng AMP) tỷ lệ thuận với việc giảm trực tiếp Cloud Cost cho mỗi epoch. Nhờ AMP, lượng tiền tiết kiệm được cho các dự án dài hạn là cực kỳ lớn.
![FP32 vs AMP Comparison](./screenshots/part8_fp32_vs_amp_comparison.png)
![Real GPU Cost Report](./screenshots/part8_real_gpu_cost_report.png)

**3. GPU utilization patterns**
Sự khác biệt trong Utilization cho thấy FP32 có thể không vắt kiệt sức mạnh của các Tensor Cores đời mới, trong khi AMP tối ưu phần cứng hơn, dẫn đến thông lượng dữ liệu (Throughput) cao hơn hẳn.

---

### Part 8.5: Advanced analysis

**1. Multi-GPU scaling efficiency**
Khi phân tích cấu hình từ 1, 2, 4 đến 8 GPUs, biểu đồ cho thấy "diminishing returns" (quy luật hiệu suất giảm dần). Dù tốc độ train nhanh hơn, nhưng chi phí tổng lại tăng do overhead của communication giữa các GPU (gradient synchronization). Hiệu suất scaling giảm dần và đòi hỏi phải đánh đổi (trade-off) giữa "thời gian" và "tổng ngân sách".
![Multi-GPU Scaling](./screenshots/part85_multi_gpu_analysis.png)

**2. Project cost forecasting**
Bằng việc phân tích trend lịch sử tiêu dùng chi phí (Burn rate), biểu đồ Forecast cho chúng ta một góc nhìn rõ ràng trong 30 ngày tiếp theo. Khoảng tin cậy (Confidence intervals) rất quan trọng để phòng ngừa các rủi ro phát sinh hoặc spike về giá.
![Project Forecast](./screenshots/part85_project_forecast.png)

**3. Optimization strategy prioritization**
Hệ thống phân tích đã xây dựng thành công một quy trình thuật toán dựa trên rules để đánh giá mức độ ưu tiên: High (Cần downsize/terminate do utilization < 30%), Medium (Cân nhắc Spot/Scheduling) và Low (Monitor thêm).
Chiến lược này cũng được hiện thực hóa qua bản kế hoạch roadmap rõ ràng, từ việc cắm cờ (tagging) cho đến điều chỉnh kiến trúc hạ tầng (AMP, Auto-scaling).
![Optimization Analysis](./screenshots/part85_optimization_analysis.png)
![Challenge Strategy Roadmap](./screenshots/part85_challenge_strategy.png)
![Integrated Dashboard](./screenshots/part85_integrated_dashboard.png)

---

## 3. Kết luận và học hỏi

**Những kỹ năng FinOps đã học**
- Cách thức truy xuất và phân tích metrics từ tài nguyên GPU.
- Khả năng tính toán chi phí (Cost projection) trên thời gian thực và quản lý ngân sách.
- Hiểu được sự khác biệt sâu sắc giữa tối ưu ở mức System (Hardware/Spot/Autoscaling) và mức Application/Code (Mixed Precision).

**Các chiến lược cost optimization hiệu quả**
- Sử dụng **Spot Instances** cho các job có khả năng chịu lỗi và gián đoạn.
- Triển khai **Auto-scaling** bám sát Queue-length thay vì chạy Idle cố định.
- Vận dụng **Automatic Mixed Precision (AMP)** để đẩy nhanh thời gian training và hạ VRAM footprint, giảm thẳng thời gian thuê máy.
- Chủ động theo dõi mức độ Utilization để đưa ra quyết định **Right-sizing** hoặc **Termination** kip thời.

**Ứng dụng thực tế trong projects**
Trong các dự án huấn luyện LLM hay Computer Vision ở môi trường Production (AWS/GCP), chi phí có thể lên tới hàng ngàn USD mỗi tháng. Từ kiến thức lab này, tôi có thể ngay lập tức thiết lập dashboard cảnh báo phí (Billing Alarms), thiết kế policy tận dụng các spot nodes, và tham mưu cho đội Data Science luôn enable cờ AMP khi train model. Những hành động này sẽ tạo ra giá trị thiết thực và trực tiếp cho bài toán kinh tế của mọi dự án AI.
