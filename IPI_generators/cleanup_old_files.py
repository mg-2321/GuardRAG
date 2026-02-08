#!/usr/bin/env python3
"""
Cleanup script to remove old/unnecessary files before regeneration
"""

import os
from pathlib import Path

def cleanup():
    """Remove old/unnecessary files"""
    base_dir = Path(__file__).parent
    
    # Files to remove (old generators that are not used)
    old_files = [
        'dense_aligned_ipi_generator.py',
        'query_aligned_ipi_generator.py',
    ]
    
    # Check if these files are actually unused
    removed = []
    kept = []
    
    for old_file in old_files:
        file_path = base_dir / old_file
        if file_path.exists():
            # Check if it's imported anywhere
            import_check = os.popen(f"grep -r '{old_file.replace('.py', '')}' {base_dir.parent} --include='*.py' 2>/dev/null | grep -v '{old_file}' | head -1").read().strip()
            if not import_check:
                print(f"Removing unused file: {old_file}")
                file_path.unlink()
                removed.append(old_file)
            else:
                print(f"Keeping {old_file} (still referenced)")
                kept.append(old_file)
        else:
            print(f"File not found: {old_file}")
    
    print(f"\n✓ Cleanup complete: Removed {len(removed)}, Kept {len(kept)}")
    return removed, kept

if __name__ == "__main__":
    cleanup()

