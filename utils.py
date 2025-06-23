import os
import time

def saveImage(image, folder = "", prefix = ""):
    # Tên thư mục bạn muốn lưu ảnh vào
    if folder == "":
        output_dir = "generated_images"
    else: 
        output_dir = f"generated_images/{folder}"

    # Tên file ảnh sẽ được lưu
    now = int(round(time.time_ns() / 1_000_000))
    if prefix == "":
        file_name = f"{now}.jpg"
    else: 
        file_name = f"{prefix}_{now}.jpg"
    
    # --- Kết thúc cấu hình ---

    # 1. Tạo thư mục nếu nó chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    print(f"Thư mục '{output_dir}' đã sẵn sàng.")

    # 2. Tạo một đối tượng ảnh mới (ví dụ: ảnh màu đỏ kích thước 400x300)
    # Hoặc bạn có thể mở một ảnh có sẵn: my_image = Image.open("path/to/your/image.jpg")
    try:
        # 3. Tạo đường dẫn đầy đủ đến file sẽ lưu
        full_path = os.path.join(output_dir, file_name)

        # 4. Dùng phương thức .save() để lưu ảnh
        # Pillow sẽ tự động nhận diện định dạng (PNG, JPEG,...) dựa vào đuôi file.
        image.save(full_path)
        print(f"Saved at: '{full_path}'")
        
    except Exception as e:
        print(f"Exception: {e}")