from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
archive = DATA_DIR / "ml-latest-small.zip"

print("Downloading MovieLens dataset...")
urlretrieve(url, archive)

print("Extracting dataset...")
with ZipFile(archive, "r") as z:
    for name in ["ml-latest-small/movies.csv", "ml-latest-small/ratings.csv", "ml-latest-small/tags.csv"]:
        z.extract(name, DATA_DIR)

source = DATA_DIR / "ml-latest-small"
for name in ["movies.csv", "ratings.csv", "tags.csv"]:
    (source / name).replace(DATA_DIR / name)
source.rmdir()
archive.unlink()
print("Done! Dataset files are in the data folder.")
