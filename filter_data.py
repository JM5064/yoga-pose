import os
import requests
from urllib.parse import urlparse


def is_valid_url(url: str, unreliable_domains: dict) -> bool:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    if domain in unreliable_domains and unreliable_domains[domain] >= 10:
        return False

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        req = requests.get(url, headers=headers, timeout=5, stream=True)
        
        if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
            if domain in unreliable_domains:
                del unreliable_domains[domain]
            
            return True
        else:
            if domain not in unreliable_domains:
                unreliable_domains[domain] = 1
            else:
                unreliable_domains[domain] += 1

            return False

    except:
        if domain not in unreliable_domains:
            unreliable_domains[domain] = 1
        else:
            unreliable_domains[domain] += 1
            
        return False


def create_filtered_dataset(original_dir: str, output_dir: str, finished_dir: str) -> None:
    unreliable_domains = {'www.yogatrail.com': 99}

    for file in os.listdir(original_dir):
        if file.endswith(".txt"):
        # if file == "atest.txt":
            print("Filtering file", file)
            print("=====================================================")

            new_file = open(f'{output_dir}/{file}', 'a')
            old_file = open(f'{original_dir}/{file}', "r")
            
            for line in old_file:
                line = line.strip()
                name, url = line.split("\t")

                is_valid = is_valid_url(url, unreliable_domains)
                if is_valid:
                    new_file.write(f'{name}\t{url}\n')

                print(is_valid, url)

            new_file.close()
            old_file.close()

            os.rename(f"{original_dir}/{file}", f"{finished_dir}/{file}")
            print("Finished filtering", file)
            print(unreliable_domains)
            print()

    print("DONE")

original_dir = "Yoga-82/yoga_dataset_links"
output_dir = "Yoga-82/filtered_yoga_dataset_links"
finished_dir = "Yoga-82/finished_dataset_links"

create_filtered_dataset(original_dir, output_dir, finished_dir)

