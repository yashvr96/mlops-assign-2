import zipfile
import os
from pathlib import Path

def package_submission(output_filename="mlops_submission.zip"):
    # Define what to include and exclude
    root_dir = Path(".")
    
    # Exclude patterns (using simple substring matching or glob)
    excludes = [
        "venv", ".venv", "env", "__pycache__", ".git", ".github",
        ".dvc", "mlruns", "data", "mlops_submission.zip",
        ".pytest_cache", ".vscode", ".idea", "*.pyc"
    ]
    
    # Explicitly include important files even if in root
    includes = [
        "src", "texts", "scripts", "models",
        "dvc.yaml", "dvc.lock", "dvc.yaml",
        "requirements.txt", "Dockerfile", "docker-compose.yml",
        "README.md", "task.md", "implementation_plan.md"
    ]
    
    print(f"Packaging submission to {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directories
        for root, dirs, files in os.walk(root_dir):
            # Resolve path
            current_path = Path(root)
            rel_path = current_path.relative_to(root_dir)
            
            # Skip excluded top-level directories
            if any(part in excludes for part in rel_path.parts):
                continue
                
            # Filter subdirectories in-place to avoid walking them
            dirs[:] = [d for d in dirs if d not in excludes]
            
            for file in files:
                file_path = current_path / file
                file_rel_path = file_path.relative_to(root_dir)
                
                # Check exclusion for files
                if any(ex in str(file_rel_path) for ex in excludes):
                    continue
                
                # Special handling: Skip large model files IF they are somehow not in models directory, 
                # but we want to KEEP models/model.pth usually.
                # Here we generally include everything not excluded.
                
                if file_path.stat().st_size > 100 * 1024 * 1024:
                     print(f"Skipping large file: {file_rel_path}")
                     continue

                print(f"Adding: {file_rel_path}")
                zipf.write(file_path, file_rel_path)
                
    print(f"Submission packaged successfully: {output_filename}")

if __name__ == "__main__":
    package_submission()
