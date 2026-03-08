"""Unpack a .docx (ZIP) file into a directory."""
import sys
import zipfile
import os

def unpack(docx_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(docx_path, 'r') as z:
        z.extractall(output_dir)

if __name__ == '__main__':
    unpack(sys.argv[1], sys.argv[2])
