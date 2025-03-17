import os
import time
import threading
import queue
import requests
import base64
import streamlit as st
import torch
import torch.nn.functional as F
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import random

# Device configuration
device = "cpu"

# Constants for Scraper
MAX_IMAGES = 100
SCROLL_PAUSE_TIME = 2
SCROLL_ATTEMPTS = 5

# RunPod Configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/igmwmcy233clp8/run"
RUNPOD_API_KEY = "rpa_ISZ14WVLKJOZBF0GQFJDLSKCMY67BRF6975QSNACjqn662"

# Queue and global variables
image_queue = queue.Queue()
collected_urls = set()
lock = threading.Lock()

# Selenium Driver Setup
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36"
    ]
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win64",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    
    driver.execute_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.navigator.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
    """)
    return driver

# Navigate to Yandex Similar Images
def navigate_to_similar_images(driver, image_path):
    print("Navigating to Yandex Images...")
    driver.get("https://yandex.com/images")
    wait = WebDriverWait(driver, 60)
    
    try:
        print("Locating 'Image search' button...")
        upload_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Image search']")))
        upload_button.click()
        print("Clicked 'Image search' button.")
        
        print("Locating file input...")
        file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        absolute_path = os.path.abspath(image_path)
        print(f"Uploading image: {absolute_path}")
        file_input.send_keys(absolute_path)
        
        print("Waiting for results to load...")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "CbirNavigation-TabsItem_active")))
        print("Locating 'Similar' tab...")
        similar_images_tab = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-cbir-page-type='similar']")))
        similar_images_tab.click()
        
        print("Waiting for similar images to load...")
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "serp-item")))
        print("Similar images loaded.")
        
    except Exception as e:
        print(f"Error in navigate_to_similar_images: {e}")
        driver.quit()
        raise

# Scroll until sufficient images are found
def scroll_until_no_new_images(driver):
    prev_count = 0
    attempts = 0
    print("Starting scroll...")
    while len(collected_urls) < MAX_IMAGES:
        images = driver.find_elements(By.CSS_SELECTOR, "img.serp-item__thumb")
        curr_count = len(images)
        print(f"Found {curr_count} images (previous: {prev_count})")
        if curr_count == prev_count and curr_count > 0:
            attempts += 1
            if attempts >= SCROLL_ATTEMPTS:
                break
        else:
            attempts = 0
        prev_count = curr_count
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)

# Scraper function to collect URLs
def scraper_function(driver, image_path):
    navigate_to_similar_images(driver, image_path)
    urls = []
    while len(urls) < MAX_IMAGES:
        scroll_until_no_new_images(driver)
        images = driver.find_elements(By.CSS_SELECTOR, "img.serp-item__thumb")
        for img in images:
            src = img.get_attribute("src")
            if src and src.startswith("http") and src not in urls:
                urls.append(src)
                if len(urls) >= MAX_IMAGES:
                    break
        if len(urls) >= MAX_IMAGES:
            break
    driver.quit()
    image_queue.put(urls)
    print(f"Scraping complete. Total URLs collected: {len(urls)}")

# Fetch embeddings from RunPod
def get_embeddings(image_data_list, endpoint_url=RUNPOD_ENDPOINT, api_key=RUNPOD_API_KEY, timeout=300, poll_interval=1):
    """Fetch embeddings from RunPod serverless endpoint."""
    payload = {"input": {"images": image_data_list}}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    # Submit job
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error submitting job: {e}")
        return None
    
    result = response.json()
    job_id = result.get("id")
    if not job_id:
        print("Job ID not found in response:", result)
        return None
    
    status_url = endpoint_url.replace("/run", f"/status/{job_id}")
    
    # Poll for status
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status_response = requests.get(status_url, headers=headers, timeout=30)
            status_response.raise_for_status()
            status_result = status_response.json()
            
            if status_result.get("status") == "COMPLETED":
                output = status_result.get("output", {})
                if output.get("statusCode") == 200:
                    embeddings_dict = output["body"]["embeddings"]
                    return [(name, emb) for name, emb in embeddings_dict.items()]
                else:
                    print(f"Job failed with statusCode {output.get('statusCode')}: {output.get('body')}")
                    return None
            elif status_result.get("status") in ["FAILED", "CANCELLED"]:
                print(f"Job {status_result.get('status').lower()}: {status_result}")
                return None
            time.sleep(poll_interval)
        except requests.RequestException as e:
            print(f"Error checking status: {e}")
            return None
    
    print(f"Job timed out after {timeout} seconds")
    return None

# Similarity calculation functions
def manhattan_to_similarity(distance):
    return torch.exp(-distance)

def similarity_to_label(similarity):
    similarity = similarity.item() if isinstance(similarity, torch.Tensor) else similarity
    if similarity >= 0.95:
        return 1
    elif similarity >= 0.90:
        return 2
    elif similarity >= 0.80:
        return 3
    elif similarity >= 0.70:
        return 4
    else:
        return 5

# Main Streamlit application
def main():
    st.title("Image Similarity Checker with Yandex Scraping (RunPod)")

    st.sidebar.header("Input Options")
    ref_image_file = st.sidebar.file_uploader("Upload Reference Image", type=['jpg', 'jpeg', 'png'], key="ref_image")

    if not ref_image_file:
        return

    st.write(f"Uploaded file: {ref_image_file.name}")
    st.write("### Processing Reference Image")
    ref_image = Image.open(ref_image_file).convert("RGB")
    st.image(ref_image, caption="Reference Image", width=150)

    # Encode reference image
    ref_image_data = base64.b64encode(ref_image_file.getvalue()).decode('utf-8')
    ref_dict = {"image_data": ref_image_data, "image_name": "reference"}

    # Scrape images
    temp_image_path = "temp_reference_image.png"
    ref_image.save(temp_image_path)
    with st.spinner(f"Scraping {MAX_IMAGES} similar images from Yandex..."):
        driver = setup_driver()
        scraper_thread = threading.Thread(target=scraper_function, args=(driver, temp_image_path))
        scraper_thread.start()
        urls = image_queue.get()
        scraper_thread.join()

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    if not urls:
        st.warning("No images were scraped.")
        return

    # Process images
    with st.spinner("Processing images and fetching embeddings..."):
        # Download and encode scraped images
        scraped_images = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.content
                    img_data = base64.b64encode(content).decode('utf-8')
                    img_dict = {"image_data": img_data, "image_name": url}
                    scraped_images.append((img_dict, content))
            except Exception as e:
                print(f"Error downloading {url}: {e}")

        # Fetch embeddings for all images
        all_images = [ref_dict] + [img_dict for img_dict, _ in scraped_images]
        embeddings_list = get_embeddings(all_images)
        
        if not embeddings_list:
            st.error("Failed to retrieve embeddings from RunPod.")
            return

        # Convert embeddings to tensors
        embeddings_dict = {name: torch.tensor(emb, dtype=torch.float32).to(device) 
                          for name, emb in embeddings_list}
        ref_embedding = embeddings_dict["reference"]

        # Calculate similarities
        results = []
        for img_dict, content in scraped_images:
            url = img_dict["image_name"]
            if url in embeddings_dict:
                embedding = embeddings_dict[url]
                embedding1 = F.normalize(ref_embedding, p=2, dim=0)
                embedding2 = F.normalize(embedding, p=2, dim=0)
                distance = torch.sum(torch.abs(embedding1 - embedding2))
                similarity = manhattan_to_similarity(distance)
                label = similarity_to_label(similarity)
                img = Image.open(io.BytesIO(content)).convert("RGB")
                results.append((img, url, similarity, label))

    # Display results
    st.write("### Similarity Results")
    if results:
        for idx, (img, url, similarity, label) in enumerate(results):
            col1, col2, col3, col4 = st.columns(4)
            col1.image(ref_image, caption="Reference Image", width=150)
            col2.image(img, caption=f"Image {idx + 1}", width=150)
            col3.write(f"Similarity: {round(similarity.item() * 100, 2)}%")
            col4.write(f"Label: {label}")
        st.success(f"Processed {len(results)} images!")
    else:
        st.warning("No images were processed.")

if __name__ == "__main__":
    main()