import requests
import time
import sys
import io
from PIL import Image

def wait_for_service(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Service is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print("Waiting for service...")
    return False

def test_prediction(url):
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("Prediction successful!")
            print("Response:", response.json())
            return True
        else:
            print(f"Prediction failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Error during prediction test: {e}")
        return False

if __name__ == "__main__":
    base_url = "http://localhost:8000"
    
    print("Starting smoke tests...")
    
    if not wait_for_service(f"{base_url}/health"):
        print("Service failed to start.")
        sys.exit(1)
        
    if not test_prediction(f"{base_url}/predict"):
        print("Prediction test failed.")
        sys.exit(1)
        
    print("All smoke tests passed!")
    sys.exit(0)
