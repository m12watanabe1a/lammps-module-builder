#%Module1.0

proc ModulesHelp { } {
    puts stderr "LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator"
}

set prefix "$PREFIX"
set pythonpath "$PYTHONPATH"
set version "$VERSION"

module-whatis "Name: LAMMPS"
module-whatis "Version: $version"
module-whatis "Description: LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator"
module-whatis "URL: https://www.lammps.org"

$MODULE_DEPENDS_ON

prepend-path PATH [file join $prefix "bin"]
prepend-path CPATH [file join $prefix "include"]
prepend-path DYLD_LIBRARY_PATH [file join $prefix "lib"]
prepend-path CMAKE_PREFIX_PATH $prefix
prepend-path PYTHONPATH $pythonpath
