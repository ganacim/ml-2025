#!/bin/bash
mkdir data
cd data
if [ -d cats_and_dogs/ ] ; then
    echo done
    exit 0
fi

wget "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
unzip kagglecatsanddogs_5340.zip

rm kagglecatsanddogs_5340.zip

mv PetImages cats_and_dogs
cd cats_and_dogs

# remove corrupted
rm Cat/666.jpg
rm Dog/10733.jpg
rm Dog/11702.jpg
rm Dog/4203.jpg

# remove thumbnails
rm Dog/Thumbs.db Cat/Thumbs.db