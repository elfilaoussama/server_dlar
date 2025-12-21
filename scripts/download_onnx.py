from huggingface_hub import hf_hub_download
import shutil
import os

def download_model():
    repo_id = "Oblix/yolov10m-doclaynet_ONNX_document-layout-analysis"
    filename = "onnx/model.onnx"
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded to {model_path}")
        
        # Copy to local dir
        target_path = "yolov10m_doclaynet.onnx"
        shutil.copy(model_path, target_path)
        print(f"Copied to {target_path}")
        
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    download_model()
