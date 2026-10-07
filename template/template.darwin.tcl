#%Module1.0

proc ModulesHelp { } {
    puts stderr "LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator"
}

set prefix "$PREFIX"
set version "$VERSION"
set python_version "$PYTHON_VERSION"

module-whatis "Name: LAMMPS"
module-whatis "Version: $version"
module-whatis "Description: LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator"
module-whatis "URL: https://www.lammps.org"

family "LAMMPS"

prepend-path PATH [file join $prefix "bin"]
prepend-path CPATH [file join $prefix "include"]
prepend-path DYLD_LIBRARY_PATH [file join $prefix "lib"]
prepend-path PYTHONPATH [file join $prefix "lib" "python${python_version}" "site-packages"]
prepend-path CMAKE_PREFIX_PATH $prefix

$MODULE_DEPENDS_ON