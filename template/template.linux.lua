---
--- THIS FILE IS GENERATED AUTOMATICALLY DO NOT EDIT MANUALLY.
---

help([[
LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
]])

local prefix = "$PREFIX"
local pythonpath = "$PYTHONPATH"
local version = "$VERSION"

whatis("Name: LAMMPS")
whatis("Version: " .. version)
whatis("Description: LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator")
whatis("URL: https://www.lammps.org")

prepend_path("PATH", pathJoin(prefix, "bin"))
prepend_path("CPATH", pathJoin(prefix, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(prefix, "lib"))
prepend_path("CMAKE_PREFIX_PATH", prefix)
prepend_path("PYTHONPATH", pythonpath)
