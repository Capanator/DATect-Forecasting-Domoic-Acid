#!/usr/bin/env python3
"""Debug script to check if file modification is working."""

import shutil
import re
import os

def debug_file_modification():
    original_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory.py"
    backup_file = "/Users/ansonchen/Documents/GitHub/DATect-Forecasting-Domoic-Acid/forecasting/model_factory_debug.py"
    
    # Create backup
    shutil.copy2(original_file, backup_file)
    
    # Read the original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    print("=== ORIGINAL CONTENT (max_depth lines) ===")
    for i, line in enumerate(content.split('\n'), 1):
        if 'max_depth=' in line:
            print(f"Line {i}: {line}")
    
    # Test regex replacement
    test_params = {'max_depth': 99}
    modified_content = content
    
    for param_name, value in test_params.items():
        pattern = f"{param_name}=[\\d.]+,?"
        replacement = f"{param_name}={value},"
        
        print(f"\n=== REGEX TEST ===")
        print(f"Pattern: {pattern}")
        print(f"Replacement: {replacement}")
        
        # Test if pattern matches anything
        matches = re.findall(pattern, content)
        print(f"Found matches: {matches}")
        
        if matches:
            modified_content = re.sub(pattern, replacement, modified_content)
            print("Replacement applied")
        else:
            print("No matches found - pattern not working!")
    
    print("\n=== MODIFIED CONTENT (max_depth lines) ===")
    for i, line in enumerate(modified_content.split('\n'), 1):
        if 'max_depth=' in line:
            print(f"Line {i}: {line}")
    
    # Write and check
    with open(original_file, 'w') as f:
        f.write(modified_content)
    
    # Verify the file was changed
    with open(original_file, 'r') as f:
        final_content = f.read()
    
    print("\n=== FINAL FILE CONTENT (max_depth lines) ===")
    for i, line in enumerate(final_content.split('\n'), 1):
        if 'max_depth=' in line:
            print(f"Line {i}: {line}")
    
    # Restore original
    shutil.copy2(backup_file, original_file)
    os.remove(backup_file)
    print("\nRestored original file")

if __name__ == "__main__":
    debug_file_modification()