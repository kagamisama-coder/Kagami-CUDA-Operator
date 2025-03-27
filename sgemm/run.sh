set -e

cd build
make
cd ..

if [ $# -ne 1 ]; then
    echo "Usage: $0 <kernel_num>"
    exit 1
fi

./sgemm "$1"