#!/usr/bin/env python3
"""
Script to add Apache 2.0 License headers to Python files in the repository.
"""

import os
import re
import sys

# Define the license header template
LICENSE_HEADER = """
# Copyright 2025 coScene. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# Root directory of the repository
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Pattern to match existing license headers (to avoid duplicates)
LICENSE_PATTERN = re.compile(r'# Copyright.*?Apache License.*?\n', re.DOTALL)

# Pattern to match docstrings at the beginning of files
DOCSTRING_PATTERN = re.compile(r'^(""".+?""")\s*', re.DOTALL)

def find_python_files(directory):
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def add_license_header(file_path):
    """Add license header to a Python file if it doesn't already have one."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file already has a license header
    if LICENSE_PATTERN.search(content):
        print(f"Skipping {file_path} - already has a license header")
        return False
    
    # Check for docstring at the beginning of the file
    docstring_match = DOCSTRING_PATTERN.match(content)
    if docstring_match:
        # If there's a docstring, insert the license header after it
        docstring = docstring_match.group(1)
        rest_of_file = content[len(docstring):].lstrip()
        new_content = f"{docstring}\n{LICENSE_HEADER.strip()}\n\n{rest_of_file}"
    else:
        # If there's no docstring, add the license header at the beginning
        new_content = f"{LICENSE_HEADER.strip()}\n\n{content}"
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Added license header to {file_path}")
    return True

def main():
    """Main function to add license headers to all Python files."""
    python_files = find_python_files(ROOT_DIR)
    print(f"Found {len(python_files)} Python files")
    
    modified_count = 0
    for file_path in python_files:
        if add_license_header(file_path):
            modified_count += 1
    
    print(f"\nAdded license headers to {modified_count} files")

if __name__ == "__main__":
    main()