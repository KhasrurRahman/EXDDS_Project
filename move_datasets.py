import os
import shutil
from pathlib import Path

def move_datasets_to_project():
    old_location = Path.home() / ".ir_datasets"
    new_location = Path("data") / ".ir_datasets"
    
    print("Checking for existing datasets...")
    print("=" * 80)
    
    if not old_location.exists():
        print(f"Old location ({old_location}) doesn't exist.")
        print("Datasets will be downloaded to project folder on first run.")
        return False
    
    old_size = sum(f.stat().st_size for f in old_location.rglob('*') if f.is_file())
    print(f"Found datasets in: {old_location}")
    print(f"Total size: {old_size / (1024**3):.2f} GB")
    
    print(f"\nMoving to: {new_location}")
    print("This may take a few minutes...")
    
    new_location.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if new_location.exists():
            print(f"Target location already exists. Removing old data...")
            shutil.rmtree(new_location)
        
        shutil.move(str(old_location), str(new_location))
        print("\n" + "=" * 80)
        print("Successfully moved datasets to project folder!")
        print(f"Location: {new_location}")
        print("=" * 80)
        return True
    except Exception as e:
        print(f"\nError moving datasets: {e}")
        print("\nYou can manually copy datasets:")
        print(f"  From: {old_location}")
        print(f"  To:   {new_location}")
        return False

if __name__ == "__main__":
    move_datasets_to_project()

