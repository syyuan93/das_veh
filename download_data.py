import requests
import zipfile
from tqdm import tqdm
import os

from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """
    Download a file from a given URL to the specified output path.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(file.write(data))

def extract_zip(zip_path, output_dir):
    """
    Extract a zip file to a specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def load_data_from_zenodo():
    date = '20221223'
    file_url = f"https://zenodo.org/record/8404396/files/{date}.zip"
    temp_zip_path = f"{date}.zip"
    output_dir = "subsurface_imaging_dataset/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    download_file(file_url, temp_zip_path)
    print(f'extracting {temp_zip_path}...')
    extract_zip(temp_zip_path, output_dir)

    # Delete the temporary zip file after extraction
    os.remove(temp_zip_path)
    print(f'Data loaded to {os.path.join(output_dir, date)}')
