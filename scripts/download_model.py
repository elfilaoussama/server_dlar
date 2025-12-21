from huggingface_hub import hf_hub_download
import shutil
import os

def download_model():
    repo_id = "hantian/yolo-doclaynet"
    filename = "best.pt" # Assuming this is the filename, usually it is.
    # If not, we might need to list repo files or try 'yolov8n-doclaynet.pt'
    # Let's try 'best.pt' first as it's standard for YOLO exports.
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded to {model_path}")
        
        # Copy to local dir
        target_path = "yolo_doclaynet.pt"
        shutil.copy(model_path, target_path)
        print(f"Copied to {target_path}")
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Trying alternative filename 'weights/best.pt'...")
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename="weights/best.pt")
            shutil.copy(model_path, "yolo_doclaynet.pt")
            print("Success!")
        except Exception as e2:
            print(f"Failed again: {e2}")

if __name__ == "__main__":
    download_model()
