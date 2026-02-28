#!/bin/bash
#
# install.sh - Install XFOIL 6.97 to /usr/local
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/usr/local}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check that binaries exist
for bin in xfoil pplot pxplot; do
    if [ ! -x "$SCRIPT_DIR/bin/$bin" ]; then
        error "'bin/$bin' not found. Run ./build.sh first."
        exit 1
    fi
done

# Check for root/sudo if installing to system dirs
if [ ! -w "$PREFIX/bin" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        error "Install to $PREFIX requires root. Run: sudo ./install.sh"
        exit 1
    fi
fi

info "Installing XFOIL to $PREFIX..."

# Install binaries
install -d "$PREFIX/bin"
install -m 755 "$SCRIPT_DIR/bin/xfoil"  "$PREFIX/bin/xfoil"
install -m 755 "$SCRIPT_DIR/bin/pplot"  "$PREFIX/bin/pplot"
install -m 755 "$SCRIPT_DIR/bin/pxplot" "$PREFIX/bin/pxplot"

# Install Orr-Sommerfeld data file
install -d "$PREFIX/share/xfoil"
if [ -f "$SCRIPT_DIR/orrs/osmap.dat" ]; then
    install -m 644 "$SCRIPT_DIR/orrs/osmap.dat" "$PREFIX/share/xfoil/osmap.dat"
    info "Installed osmap.dat to $PREFIX/share/xfoil/"
fi

info "Installed xfoil, pplot, pxplot to $PREFIX/bin/"
echo ""
echo "To use Orr-Sommerfeld transition data, set:"
echo "  export OSMAP=$PREFIX/share/xfoil/osmap.dat"
echo ""
echo "You can add this to your ~/.bashrc for persistence."
info "Installation complete."
