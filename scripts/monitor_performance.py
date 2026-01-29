import requests
import os
import time
import json
import random
from pathlib import Path

def monitor_performance(api_url="http://localhost:8000/predict", test_dir="data/processed/test", num_samples=20):
    print(f"Starting performance monitoring on {num_samples} samples...")
    print(f"Source: {test_dir}")
    
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: Test directory {test_dir} not found.")
        return

    # Get all images
    all_images = [f for f in test_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    
    if not all_images:
        print("No images found for testing.")
        return

    # Select random samples
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    results = []
    correct_predictions = 0
    total_latency = 0
    
    for img_path in samples:
        # Extract true label from filename (assuming format contains 'cat' or 'dog')
        true_label = "dog" if "dog" in img_path.name.lower() else "cat"
        
        start_time = time.time()
        try:
            with open(img_path, "rb") as f:
                # Guess content type based on extension
                mime_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                files = {"file": (img_path.name, f, mime_type)}
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                data = response.json()
                
                prediction = data["prediction"]
                confidence = float(data["confidence"])
                
                is_correct = (prediction == true_label)
                if is_correct:
                    correct_predictions += 1
                
                latency = (time.time() - start_time) * 1000 # ms
                total_latency += latency
                
                results.append({
                    "filename": img_path.name,
                    "true_label": true_label,
                    "prediction": prediction,
                    "confidence": confidence,
                    "latency_ms": round(latency, 2),
                    "correct": is_correct
                })
                print(f"Processed {img_path.name}: Pred={prediction}, True={true_label}, Time={latency:.2f}ms")
                
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

    # Calculate metrics
    accuracy = correct_predictions / len(samples) if samples else 0
    avg_latency = total_latency / len(samples) if samples else 0
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(samples),
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "details": results
    }
    
    # Save report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nPerformance Monitoring Complete")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg Latency: {avg_latency:.2f}ms")
    print("Report saved to performance_report.json")

if __name__ == "__main__":
    monitor_performance()
