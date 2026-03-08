"""Pack a directory back into a .docx (ZIP) file."""
import sys
import zipfile
import os

def pack(input_dir, docx_path):
    with zipfile.ZipFile(docx_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, input_dir)
                z.write(file_path, arcname)

if __name__ == '__main__':
    pack(sys.argv[1], sys.argv[2])
