import json
import requests
import os

output_dir = r"C:\Users\utkar\Downloads\Utkarsh Gupta\NUIG\Semester 2\DA Project\Extracted Dataset updated"

def save_image(image_url, image_path):
    try:
        img_response = requests.get(image_url, stream=True)
        if img_response.status_code == 200:
            with open(image_path, 'wb') as img_file:
                for chunk in img_response.iter_content(1024):
                    img_file.write(chunk)
        else:
            print(f"Sorry, failed to download {image_url}")
    except Exception as err:
        print(f"There is an error while downloading {image_url}: {err}")

os.makedirs(output_dir, exist_ok=True)

with open(r"C:\Users\utkar\Downloads\Utkarsh Gupta\NUIG\Semester 2\DA Project\Json file of Bates\template_info.json", 'r') as json_file:
    for json_line in json_file:
        try:
            image_data = json.loads(json_line)
            for main_key, main_value in image_data.items():
                if "original_info" in main_value:
                    for info in main_value["original_info"]:
                        img_url = info["url"]
                        img_title = info["title"]

                        file_path = os.path.join(output_dir, os.path.basename(img_title) + '.jpg')

                        save_image(img_url, file_path)
        except json.JSONDecodeError as json_err:
            print(f"There is an error in processing JSON file- {json_err}")
        except Exception as general_err:
            print(f"There is an error in processing line- {general_err}")

print("Images downloaded")