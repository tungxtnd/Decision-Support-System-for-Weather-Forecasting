
# 🌤️ Smart Weather AI - Ứng dụng Dự báo Thời tiết Thông minh
Chào mừng đến với **Smart Weather AI**! Đây là dự án ứng dụng Trí tuệ nhân tạo (AI) để dự báo thời tiết và đưa ra lời khuyên sinh hoạt cá nhân hóa. Hệ thống kết hợp giữa các mô hình Deep Learning (BiLSTM) để xử lý chuỗi thời gian và Generative AI (Google Gemini) để tư vấn ngữ cảnh.
## 🚀 Tính năng nổi bật
* **Dự báo Đa chiều:**
    * 🌧️ **Dự đoán Mưa:** Tính toán xác suất trời mưa vào ngày mai (Có mưa/Không mưa).
    * 🌡️ **Dự đoán Chỉ số:** Dự báo cụ thể Nhiệt độ (Min/Max), Độ ẩm, Tốc độ gió, Áp suất khí quyển.
* **Smart Advisor (Trợ lý ảo):** Sử dụng **Google Gemini** để phân tích các chỉ số thời tiết và đưa ra lời khuyên cụ thể về:
    * Trang phục nên mặc.
    * Hoạt động ngoài trời/trong nhà.
    * Lưu ý sức khỏe đặc biệt.
* **Giao diện trực quan:** Xây dựng trên nền tảng **Streamlit**, dễ dàng tương tác và theo dõi biểu đồ.
## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.8+
* **Frontend:** Streamlit
* **AI/Deep Learning Core:**
    * TensorFlow / Keras (LSTM & BiLSTM models)
    * Scikit-learn (Preprocessing)
* **GenAI:** Google GenAI SDK (Gemini 1.5 Flash/Pro)
* **Xử lý dữ liệu:** Pandas, NumPy, Matplotlib
## 📂 Cấu trúc dự án
```text
├── app.py                      # File chạy chính của ứng dụng Streamlit
├── 01_Rainfall_Prediction.ipynb # Notebook huấn luyện model phân loại Mưa (BiLSTM)
├── Group61.ipynb               # Notebook huấn luyện model hồi quy (Nhiệt độ, độ ẩm...)
├── weatherAUS.csv              # Dữ liệu huấn luyện (Nguồn: Kaggle)
├── requirements.txt            # Danh sách thư viện cần cài đặt
├── models/                     # Thư mục chứa các file Model & Scaler đã train
│   ├── lstm_weather_model.h5       # Model dự báo mưa
│   ├── weather_regression_model.keras # Model dự báo chỉ số
│   ├── scaler.pkl                  # Scaler cho model mưa
│   ├── scaler_reg_input.pkl        # Scaler input cho model hồi quy
│   └── scaler_reg_target.pkl       # Scaler target cho model hồi quy
└── README.md                   # Tài liệu hướng dẫn
```
## ⚙️ Hướng dẫn cài đặt & Chạy ứng dụng
### 1. Chuẩn bị môi trường
Clone dự án về máy và cài đặt các thư viện cần thiết:
```bash
git clone [https://github.com/your-username/smart-weather-ai.git](https://github.com/your-username/smart-weather-ai.git)
cd smart-weather-ai
pip install -r requirements.txt
*(Nội dung file `requirements.txt` gợi ý: `streamlit`, `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `google-genai`)*
### 2. Cấu hình API Key
Mở file `app.py`, tìm dòng chứa biến `MY_API_KEY` và dán Google AI Studio API Key của bạn vào:
```python
MY_API_KEY = "AIzaSyAf-xxxxxxxxxxxxxxxxxxxxxxxx"
### 3. Khởi chạy ứng dụng
Chạy lệnh sau trong terminal:
```bash
streamlit run app.py
```
Sau đó truy cập vào đường dẫn hiển thị (thường là `http://localhost:8501`).
## 🧠 Chi tiết Mô hình (Model Architecture)
Hệ thống sử dụng cơ chế **Ensemble Models** với kỹ thuật xử lý dữ liệu linh hoạt:
1. **Model Phân loại (Rainfall Classifier):**
* **Kiến trúc:** Bi-Directional LSTM (BiLSTM).
* **Timestep:** 14 ngày.
* **Nhiệm vụ:** Học các mẫu hình thời tiết ngắn hạn để quyết định khả năng mưa.
2. **Model Hồi quy (Weather Regression):**
* **Kiến trúc:** LSTM Stacked.
* **Timestep:** 30 ngày (hoặc 14 ngày tùy phiên bản).
* **Nhiệm vụ:** Dự báo các giá trị liên tục (Nhiệt độ, Độ ẩm...).
*Lưu ý kỹ thuật: Ứng dụng tự động xử lý Data Slicing để đảm bảo dữ liệu đầu vào khớp với Timestep của từng model (Window Sliding).*
4.  Điền tên thành viên nhóm vào phần cuối cùng.

```
