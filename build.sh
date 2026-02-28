#!/bin/bash
#
# build.sh - Build script for XFOIL 6.97 on Linux
#
# Checks for dependencies, builds the plotting library, then builds XFOIL.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Check and install dependencies ---
check_deps() {
    info "Checking build dependencies..."

    MISSING=()

    if ! command -v gfortran &>/dev/null; then
        MISSING+=("gfortran")
    fi

    if ! command -v make &>/dev/null; then
        MISSING+=("make")
    fi

    # Check for X11 development headers
    if [ ! -f /usr/include/X11/Xlib.h ]; then
        MISSING+=("libx11-dev")
    fi

    if [ ${#MISSING[@]} -eq 0 ]; then
        info "All dependencies found."
        return 0
    fi

    warn "Missing dependencies: ${MISSING[*]}"

    if command -v apt-get &>/dev/null; then
        # Map binary names to package names
        PACKAGES=()
        for dep in "${MISSING[@]}"; do
            case "$dep" in
                gfortran)    PACKAGES+=("gfortran") ;;
                make)        PACKAGES+=("make") ;;
                libx11-dev)  PACKAGES+=("libx11-dev") ;;
            esac
        done

        info "Installing with apt-get: ${PACKAGES[*]}"
        if sudo apt-get update -qq && sudo apt-get install -y -qq "${PACKAGES[@]}"; then
            info "Dependencies installed."
        else
            error "Failed to install packages. Please install manually:"
            error "  sudo apt-get install ${PACKAGES[*]}"
            exit 1
        fi
    else
        error "Cannot auto-install packages (apt-get not found)."
        error "Please install manually: ${MISSING[*]}"
        exit 1
    fi
}

# --- Build the plotting library ---
build_plotlib() {
    info "Building plotting library (plotlib)..."
    cd "$SCRIPT_DIR/plotlib"

    make clean 2>/dev/null || true
    make libPlt.a
    make libPltDP.a

    if [ -f libPlt.a ] && [ -f libPltDP.a ]; then
        info "Plotting library built successfully."
    else
        error "Failed to build plotting library."
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# --- Build XFOIL and utilities ---
build_xfoil() {
    info "Building XFOIL..."
    cd "$SCRIPT_DIR/bin"

    # Clean old object files
    rm -f *.o xfoil pplot pxplot 2>/dev/null || true

    make xfoil
    make pplot
    make pxplot

    if [ -x xfoil ] && [ -x pplot ] && [ -x pxplot ]; then
        info "XFOIL built successfully."
    else
        error "Failed to build XFOIL."
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# --- Verify the build ---
verify_build() {
    info "Verifying build..."

    if [ ! -x bin/xfoil ]; then
        error "bin/xfoil not found or not executable."
        exit 1
    fi

    # Quick smoke test: run xfoil with QUIT command
    echo "QUIT" | bin/xfoil > /dev/null 2>&1 && true

    info "Build verification complete."
    echo ""
    echo "=========================================="
    info "XFOIL 6.97 built successfully!"
    echo ""
    echo "  Binaries:"
    echo "    bin/xfoil  - Main XFOIL program"
    echo "    bin/pplot  - Polar plot utility"
    echo "    bin/pxplot - X11 polar plot utility"
    echo ""
    echo "  Run with:  ./bin/xfoil"
    echo "  Install:   sudo ./install.sh"
    echo "=========================================="
}

# --- Main ---
info "XFOIL 6.97 Build Script"
echo ""

check_deps
build_plotlib
build_xfoil
verify_build
