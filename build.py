#!/usr/bin/env python3

import os
import sys
import yaml
from pathlib import Path
import tempfile
import subprocess
import glob
from jinja2 import Environment, FileSystemLoader


def verify_sha256(file_path: str | os.PathLike[str], expected_sha256: str) -> bool:
    """Verify the SHA256 checksum of a file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == expected_sha256


def download(url: str, dest_dir: str | os.PathLike[str], checksum: str) -> str:
    """Download file from url to dest."""
    dest = Path(dest_dir) / Path(url).name
    subprocess.run(["wget", "-O", str(dest), str(url)], check=True)

    checksum_type, checksum_value = checksum.split(":")
    if checksum_type == "sha256":
        if not verify_sha256(dest, checksum_value):
            raise ValueError("Checksum verification failed")
    else:
        raise ValueError(f"Unsupported checksum type: {checksum_type}")

    return str(dest)


def extract_archive(
    archive_path: str | os.PathLike[str], extract_to: str | os.PathLike[str]
) -> None:
    """Extract a tar.gz archive."""
    import tarfile

    original_cwd = os.getcwd()

    os.chdir(Path(archive_path).parent)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            extracted_dir = tar.getnames()[0]
            tar.extractall()
        subprocess.run(["mv", extracted_dir, extract_to], check=True)
    finally:
        os.chdir(original_cwd)

    print(f"Extracted {archive_path} -> {extract_to}")


def download_and_extract_all(
    targets: list[dict],
    url_pattern: str,
    dest_dir: str | os.PathLike[str],
    no_cache: bool = False,
) -> list[Path]:
    """Download and extract all targets."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    ret: list[Path] = []
    for target in targets:
        url: str = url_pattern.format(**target)
        extracted_name = Path(url).name.split(".")[0]
        ret.append(dest_path / extracted_name)
        if (dest_path / extracted_name).exists() and not no_cache:
            print(f"Target {url} already exists, skipping download.")
            continue
        with tempfile.TemporaryDirectory() as tempdir:
            tar_gz_file = download(url, tempdir, target["checksum"])
            extract_archive(tar_gz_file, dest_path / extracted_name)

    return ret


def build_target(folder: str | os.PathLike[str], recipe: dict, variables: dict) -> None:
    """Build the target using the provided recipe."""
    original_cwd = os.getcwd()

    def eval_variables(args: list[str]) -> list[str]:
        tmp = []
        for arg in args:
            for var in variables:
                if "{" + var + "}" not in arg:
                    continue
                arg = arg.format(**variables)
            tmp.append(arg)
        return tmp

    def eval_asktarisk(args: list[str]) -> list[str]:
        tmp = []
        for arg in args:
            if "*" in arg:
                tmp.extend(glob.glob(arg))
            else:
                tmp.append(arg)
        return tmp

    def args_filter(args: list[str]) -> list[str]:
        if not args:
            return []
        args_list = (" ".join(args)).split()
        args_list = eval_variables(args_list)
        args_list = eval_asktarisk(args_list)
        args_list = list(filter(lambda x: x is not None, args_list))
        return args_list

    try:
        for i, step in enumerate(recipe["steps"]):
            print(f"Run {i + 1}/{len(recipe['steps'])}: {step['name']}")
            os.chdir(Path(folder))
            command: str = step["command"]
            args: list[str] = args_filter(step.get("args"))
            subprocess.run(command.split() + args, check=True)
    finally:
        os.chdir(original_cwd)


def generate_modulefile(
    template: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str],
    config: dict,
) -> None:
    """Generate modulefile for the installed target."""
    print(f"Generating modulefile: {Path(template).parent} -> {output}")
    env = Environment(
        loader=FileSystemLoader(Path(template).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    print(env.list_templates())
    template_obj = env.get_template(Path(template).name)
    rendered_content = template_obj.render(config)
    with open(output, "w") as f:
        f.write(rendered_content)


def main():
    THIS_SCRIPT_DIR = Path(__file__).parent.resolve()
    OUTPUT_DIR = THIS_SCRIPT_DIR
    SRC_DIR = THIS_SCRIPT_DIR / "src"
    CONFIG_DIR = THIS_SCRIPT_DIR / "config"
    TEMPLATE_DIR = THIS_SCRIPT_DIR / "template"

    MODULES_DIR = OUTPUT_DIR / "modules"
    MODULEFILES_DIR = OUTPUT_DIR / "modulefiles"

    target_info_filename = CONFIG_DIR / "target_info.yaml"
    recipe_filename = CONFIG_DIR / "recipe.yaml"
    modulefile_template = TEMPLATE_DIR / "lammps.lua.jinja"
    no_cache: bool = False

    target_info = yaml.safe_load(target_info_filename.read_text())
    build_recipe = yaml.safe_load(recipe_filename.read_text())

    # Download & Extract
    src_folders: list[Path] = download_and_extract_all(
        target_info["targets"], target_info["url_pattern"], SRC_DIR, no_cache
    )

    for src, info in zip(src_folders, target_info["targets"], strict=True):
        install_prefix = MODULES_DIR / "{name}_{category}_{version}".format(
            **(info | target_info)
        )
        variables = {
            "install_prefix": install_prefix,
            "python_install_prefix": install_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages",
        }

        # Build & Install
        build_target(src, build_recipe, variables)

        # Generate modulefiles
        generate_modulefile(
            template=modulefile_template,
            output=MODULEFILES_DIR
            / "{name}_{category}_{version}.lua".format(**(info | target_info)),
            config=variables,
        )


if __name__ == "__main__":
    main()
