import shutil
from pathlib import Path

def cleanup_unused_datasets():
    data_dir = Path("data/.ir_datasets")
    
    if not data_dir.exists():
        print("Dataset folder not found!")
        return
    
    keep_folders = {"msmarco-passage"}
    all_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print("Cleaning up unused datasets...")
    print("=" * 80)
    print(f"Keeping: {keep_folders}")
    print(f"\nFound {len(all_folders)} dataset folders")
    
    removed = []
    kept = []
    
    for folder in all_folders:
        if folder.name in keep_folders:
            kept.append(folder.name)
            print(f"  ✓ Keeping: {folder.name}")
        else:
            try:
                shutil.rmtree(folder)
                removed.append(folder.name)
                print(f"  ✗ Removed: {folder.name}")
            except Exception as e:
                print(f"  ✗ Error removing {folder.name}: {e}")
    
    print("\n" + "=" * 80)
    print(f"Cleanup complete!")
    print(f"  Kept: {len(kept)} folders")
    print(f"  Removed: {len(removed)} folders")
    print("=" * 80)
    
    if removed:
        print("\nFreed up space by removing unused datasets.")
        print("Only msmarco-passage remains (required for experiments).")

if __name__ == "__main__":
    cleanup_unused_datasets()

