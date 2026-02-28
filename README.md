# XFOIL 6.97 for Linux

A ready-to-build Linux port of **XFOIL**, the subsonic airfoil analysis and design tool originally developed by **Mark Drela** and **Harold Youngren** at MIT.

XFOIL is an interactive program for the design and analysis of subsonic isolated airfoils. It provides:

- Viscous (or inviscid) analysis of existing airfoils
- Airfoil design and redesign by interactive modification of surface speed distributions
- Drag polar calculation and plotting
- Blowing/suction surface definition for boundary-layer control
- Lift and drag predictions with transition prediction via the e^N method
- NACA 4-digit and 5-digit airfoil generation
- Geometry manipulation (flap deflection, camber/thickness changes)

## Prerequisites

- **OS**: Ubuntu/Debian Linux (tested on Ubuntu 20.04+)
- **Compiler**: `gfortran` (GNU Fortran)
- **Build tool**: `make`
- **Graphics**: `libx11-dev` (X11 development headers)

On Ubuntu/Debian, install everything with:

```bash
sudo apt-get install gfortran make libx11-dev
```

## Quick Start

```bash
git clone <this-repo-url> xfoil-linux
cd xfoil-linux
./build.sh
./bin/xfoil
```

The `build.sh` script will:
1. Check for required dependencies (and install them via `apt-get` if missing)
2. Build the X11 plotting library (`plotlib/`)
3. Build XFOIL and companion utilities (`bin/`)
4. Verify the build

## Manual Build

If you prefer to build manually:

```bash
# 1. Build the plotting library
cd plotlib
make libPlt.a
make libPltDP.a
cd ..

# 2. Build XFOIL
cd bin
make xfoil
make pplot
make pxplot
cd ..
```

## Installation

To install XFOIL system-wide to `/usr/local/bin`:

```bash
sudo ./install.sh
```

To install to a custom prefix:

```bash
sudo PREFIX=/opt/xfoil ./install.sh
```

After installation, set the Orr-Sommerfeld data path for transition prediction:

```bash
export OSMAP=/usr/local/share/xfoil/osmap.dat
```

Add the line above to your `~/.bashrc` to make it persistent.

## Usage Examples

### Analyze a NACA 2412 airfoil

```
$ xfoil
 XFOIL   c> NACA 2412
 XFOIL   c> OPER
 .OPERi  c> VISC 1e6
 .OPERv  c> ALFA 5.0
```

### Load an airfoil from file and generate a drag polar

```
$ xfoil
 XFOIL   c> LOAD airfoil.dat
 XFOIL   c> OPER
 .OPERi  c> VISC 200000
 .OPERv  c> PACC
 .OPERv  c> <return>
 .OPERv  c> <return>
 .OPERv  c> ASEQ -2 10 0.5
 .OPERv  c> PPLO
```

### Design modifications

```
$ xfoil
 XFOIL   c> NACA 0012
 XFOIL   c> GDES
 .GDES   c> FLAP 0.75 0.0 -10.0
 .GDES   c> EXEC
 XFOIL   c> OPER
 .OPERi  c> VISC 500000
 .OPERv  c> ALFA 4.0
```

### Tips

- Type `?` at any command prompt to see available commands
- The `runs/` directory contains sample airfoil data files (e.g., `e387.dat`)
- See `sessions.txt` for full example interactive sessions
- See `xfoil_doc.txt` for the complete user guide

## Repository Structure

```
xfoil-linux/
├── src/          Fortran source code for XFOIL
├── osrc/         Orr-Sommerfeld routines (C + Fortran)
├── orrs/         Orr-Sommerfeld stability database
├── plotlib/      X11 plotting library (Xplot11)
├── bin/          Build directory (Makefile, compiled binaries)
├── runs/         Example airfoil data files
├── build.sh      Automated build script
├── install.sh    System installation script
├── xfoil_doc.txt Complete XFOIL user guide
└── sessions.txt  Example interactive sessions
```

## Credits

XFOIL was written by **Mark Drela** (MIT Department of Aeronautics and Astronautics) with contributions from **Harold Youngren**. The Xplot11 plotting library is copyright (C) 1996 Harold Youngren and Mark Drela, released under the GNU Library General Public License v2+.

- Mark Drela's XFOIL page: https://web.mit.edu/drela/Public/web/xfoil/
- Original author contact: drela@mit.edu

## License

The plotting library (`plotlib/`) is released under the **GNU Library General Public License v2** or later. See the header in `plotlib/Makefile` for details. XFOIL itself is distributed by its authors for academic and research use.
