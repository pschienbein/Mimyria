#/bin/bash


set -eE -o pipefail
trap 'echo "Error on line \"${BASH_COMMAND}\""; exit 1' ERR

ZIP="mimyria-examples-data.zip"
URL="https://schienbein.eu/data/${ZIP}"

if [ -d data ]; then
    echo "data directory already existing, please remove it before downloading again"
    exit 1
fi

echo "Downloading ${URL} ..."
curl -L "${URL}" -o "${ZIP}"

echo "Unzipping archive..."
unzip ${ZIP}

rm ${ZIP}
