import os

# Root project name
root = "ecg-ai"

# Folder structure
structure = {
    "config": ["settings.py"],
    "data": ["mitbih_loader.py", "patient_split.py", "heartbeat_segment.py"],
    "models": ["cnn_model.py", "train_global.py", "personalize.py"],
    "evaluation": ["metrics.py", "calibration.py", "noise_test.py", "early_detection.py"],
    "explainability": ["gradcam.py"],
    "app": ["dashboard.py"]
}

# Root files
root_files = ["main.py", "requirements.txt"]

def create_project():
    os.makedirs(root, exist_ok=True)

    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w") as f:
                f.write(f"# {file}\n")

    for file in root_files:
        file_path = os.path.join(root, file)
        with open(file_path, "w") as f:
            f.write(f"# {file}\n")

    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project()
