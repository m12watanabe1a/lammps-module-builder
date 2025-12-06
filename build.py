#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
from pathlib import Path
import tempfile
import subprocess
import glob
from jinja2 import Environment, FileSystemLoader
import logging


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

    logger.info(f"Extracted {archive_path} -> {extract_to}")


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
            logger.warning(f"Target {url} already exists, skipping download.")
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
            logger.info(f"Run {i + 1}/{len(recipe['steps'])}: {step['name']}")
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
    logger.info(f"Generating modulefile: {Path(template).parent} -> {output}")
    env = Environment(
        loader=FileSystemLoader(Path(template).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template_obj = env.get_template(Path(template).name)
    rendered_content = template_obj.render(config)
    if not Path(output).parent.exists():
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(rendered_content)


def main(
    *,
    prefix: str | os.PathLike[str],
    config_filename: str | os.PathLike[str],
    recipe_filename: str | os.PathLike[str],
    modulefile_template: str | os.PathLike[str],
    src: str | os.PathLike[str],
    no_cache: bool = False,
):

    MODULES_DIR = Path(prefix) / "modules"
    MODULEFILES_DIR = Path(prefix) / "modulefiles"

    target_info = yaml.safe_load(Path(config_filename).read_text())
    build_recipe = yaml.safe_load(Path(recipe_filename).read_text())

    # Download & Extract
    src_folders: list[Path] = download_and_extract_all(
        target_info["targets"], target_info["url_pattern"], src, no_cache
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
            / target_info["name"]
            / "{category}_{version}.lua".format(**info),
            config=variables,
        )


def setup_args():
    THIS_SCRIPT_DIR = Path(__file__).parent.resolve()
    default_prefix = Path.home() / ".local" / "opt"
    default_recipe = THIS_SCRIPT_DIR / "config" / "recipe.yaml"
    default_config = THIS_SCRIPT_DIR / "config" / "target_info.yaml"
    default_template = THIS_SCRIPT_DIR / "template" / "lammps.lua.jinja"
    default_src = THIS_SCRIPT_DIR / "src"
    parser = argparse.ArgumentParser(
        description="Build LAMMPS from source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=str(default_prefix),
        help="Installation prefix directory.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=str(default_recipe),
        help="Path to the build recipe configuration file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to the target info configuration file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=str(default_template),
        help="Path to the modulefile template.",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(default_src),
        help="Path to the source directory. If not exists, source files will be downloaded here.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of downloaded and extracted sources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    logger = logging.getLogger(__name__)

    try:
        assert Path(args.recipe).exists(), f"Recipe file not found: {args.recipe}"
        assert Path(args.config).exists(), f"Config file not found: {args.config}"
        assert Path(args.template).exists(), f"Template file not found: {args.template}"

        main(
            prefix=args.prefix,
            recipe_filename=args.recipe,
            config_filename=args.config,
            modulefile_template=args.template,
            src=args.src,
            no_cache=args.no_cache,
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)