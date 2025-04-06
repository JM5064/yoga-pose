import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def download_images(input_dir: str, output_dir: str) -> None:
    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
        # if file == "atest.txt":
            print("Downloading images from", file)
            print("=====================================================")

            textfile = open(f'{input_dir}/{file}')
            images = []

            for line in textfile:
                line = line.strip()

                name, url = line.split("\t")
                images.append((name, url))

            failed_downloads = download_batch(images, output_dir)

            print(f"Finished downloading {file}")
            print(f'{len(images) - len(failed_downloads)} / {len(images)} downloaded')
            print("Failed downloads:", failed_downloads)
            print()
        
    print("Finished downloading all images")


def download_image(name, url, output_dir, failed_downloads, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            pose_dir = os.path.dirname(name)
            os.makedirs(f'{output_dir}/{pose_dir}', exist_ok=True)
            
            with open(f'{output_dir}/{name}', "wb") as image:
                image.write(response.content)
            
        else:
            failed_downloads.append((url, response.status_code, response.headers.get("Content-Type", "")))

    except Exception as e:
        failed_downloads.append((url, e))


def download_batch(images, output_dir, max_workers=8):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    }

    failed_downloads = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_image, name, url, output_dir, failed_downloads, headers)
            for name, url in images
        ]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


    return failed_downloads


input_dir = "Yoga-82/filtered_yoga_dataset_links"
output_dir = "dataset"

download_images(input_dir, output_dir)
