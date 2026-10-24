import os
import shutil

from huggingface_hub import hf_hub_download

# 手動下載寶寶臉部 YOLO 模型的小工具。
# 執行後會把 Hugging Face 的 model.pt 複製到專案 models 資料夾。

# 模型輸出資料夾。
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# 從 Hugging Face 下載模型。
downloaded_path = hf_hub_download(
    repo_id="AdamCodd/YOLOv11n-face-detection",
    filename="model.pt"
)

# 複製到自己的專案 models 資料夾，並改成比較好辨識的名字。
target_path = os.path.join(output_dir, "yolov11n_face_detection.pt")
shutil.copyfile(downloaded_path, target_path)

print("模型已下載完成：")
print(os.path.abspath(target_path))
