
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
  echo "Source this script in our venv!! script ${BASH_SOURCE[0]} is not being sourced ..."
  exit 1
fi

sudo apt install liblz4-dev libflann-dev libeigen3-dev

git clone https://github.com/uhlmanngroup/pyshot.git
cd pyshot
pip install .