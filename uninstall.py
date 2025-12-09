#!/usr/bin/env python3


from lammps_module_builder.build_info import BuildInfo


def main():
    if not BuildInfo.exists():
        print("No build information file found.")
        return
    build_info = BuildInfo.load()
    build_info.delete()
    print("Build information file removed.")


if __name__ == "__main__":
    main()
