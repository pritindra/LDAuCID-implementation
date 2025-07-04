import os
import shutil
import zipfile
import urllib.request

DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=19DPObbiUbzGFEbCoPAyixrv_JT5QCQXE&export=download&authuser=0&"
    "confirm=t&uuid=7869aa1b-8a2e-4169-a9ee-f1f2d7311078&"
    "at=AENtkXYJgijttsPeTTrrX2CrUGaz%3A1730284122447"
)
ZIP_FILENAME = "dataset.zip"

print("Clearing current dataset folder...")
for item in os.listdir("."):
    if item == os.path.basename(__file__):
        continue 
    if os.path.isfile(item) or os.path.islink(item):
        os.unlink(item)
    elif os.path.isdir(item):
        shutil.rmtree(item)

print(f"Downloading dataset zip...")
urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILENAME)

print("Unzipping dataset...")
with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
    zip_ref.extractall("temp_extracted")

print("Moving extracted content into current dataset folder...")
for root, dirs, files in os.walk("temp_extracted"):
    dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]
    
    if files or dirs:
        for item in os.listdir(root):
            src = os.path.join(root, item)
            dst = os.path.join(os.getcwd(), item)
            if os.path.exists(dst):
                print(f"Skipping existing: {dst}")
            else:
                shutil.move(src, dst)
        break  # Only process the first valid folder

print("Cleaning up temporary files...")
os.remove(ZIP_FILENAME)
shutil.rmtree("temp_extracted")
print("Dataset setup complete.")

