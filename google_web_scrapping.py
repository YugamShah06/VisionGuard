import os
import requests
from PIL import Image
from io import BytesIO

def persist_image(folder_path: str, url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            filename = os.path.join(folder_path, url.split('/')[-1])
            image.save(filename, "JPEG", quality=85)
            print(f"SUCCESS - saved {url} as {filename}")
        else:
            print(f"Failed to download image: {url}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download_images(search_term: str, api_key: str, target_folder: str, num_images: int):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # SerpAPI URL for image search
    url = f"https://serpapi.com/search.json?q={search_term}&tbm=isch&num={num_images}&api_key={api_key}"

    # Perform the search
    response = requests.get(url)
    
    # Check for 401 error
    if response.status_code == 401:
        print("ERROR - Invalid API Key. Please check your API key.")
        return

    if response.status_code == 200:
        results = response.json()
        if 'images_results' in results:
            for image in results['images_results']:
                img_url = image.get('original')
                if img_url:
                    persist_image(target_folder, img_url)
        else:
            print("No images found in the results.")
    else:
        print(f"ERROR - Could not retrieve results: {response.status_code}")

# Example usage
API_KEY = "2106b8cd3c1af95b9feef445c3cad92100d4d3e16bd524865bc91c257cc95413"  # Replace with your SerpAPI key
TARGET_FOLDER = "./scraping/infection"    # Specify the target folder for saving images
SEARCH_TERM = "infection only eye images"             # Specify your search term
NUM_IMAGES = 10                  # Specify the number of images to download

search_and_download_images(SEARCH_TERM, API_KEY, TARGET_FOLDER, NUM_IMAGES)
