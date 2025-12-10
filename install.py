#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from lammps_module_builder.build_info import BuildInfo


def verify_sha256(file_path: str | os.PathLike[str], expected_sha256: str) -> bool:
    """Verify the SHA256 checksum of a file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == expected_sha256


def download(url: str, dest_dir: str | os.PathLike[str], checksum: str | None) -> str:
    """Download file from url to dest."""
    dest = Path(dest_dir) / Path(url).name
    logger.info(f"Downloading from {url}...")
    subprocess.run(
        ["wget", "-O", str(dest), str(url)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if checksum is None:
        logger.warning("No checksum provided, skipping verification.")
        return str(dest)

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
    """Extract archive to the specified directory."""

    original_cwd = os.getcwd()
    extract_path = Path(extract_to).resolve()

    os.chdir(Path(archive_path).parent)
    try:
        suffix = "".join(Path(archive_path).suffixes)

        if suffix == ".zip":
            raise NotImplementedError("zip format is not supported yet.")
        if suffix == ".tar.gz" or suffix == ".tgz":
            import tarfile

            with tarfile.open(archive_path, "r:gz") as tar:
                extracted_dir = tar.getnames()[0]
                tar.extractall()
        else:
            raise ValueError(f"Unsupported archive format: {suffix}")
        subprocess.run(
            ["mv", str(Path(extracted_dir).resolve()), str(extract_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    finally:
        os.chdir(original_cwd)

    logger.info(f"Extracted {archive_path} -> {extract_to}")


def download_and_extract_all(
    targets: list[dict],
    sources: dict[str, str],
    dest_dir: str | os.PathLike[str],
    no_cache: bool = False,
) -> list[Path]:
    """Download and extract all targets."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    ret: list[Path] = []
    for target in targets:
        url: str = (sources[target["source"]]).format(**target)
        extracted_name = Path(url).name.split(".")[0]
        ret.append(dest_path / extracted_name)
        if (dest_path / extracted_name).exists() and not no_cache:
            logger.info(
                f"{(dest_path / extracted_name)} already exists. "
                "Skipping download and extraction."
            )
            continue

        if no_cache:
            if (dest_path / extracted_name).exists():
                logger.info(
                    f"Removing existing extracted folder at {dest_path / extracted_name}"
                )
                subprocess.run(
                    ["rm", "-rf", str(dest_path / extracted_name)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
        with tempfile.TemporaryDirectory() as tempdir:
            tar_gz_file = download(url, tempdir, target.get("checksum", None))
            extract_archive(tar_gz_file, dest_path / extracted_name)

    return ret


def build_target(
    folder: str | os.PathLike[str], recipe: dict, variables: dict, action: dict
) -> None:
    """Build the target using the provided recipe."""
    original_cwd = os.getcwd()

    log_file = Path(folder) / "log" / "build.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

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
        with open(log_file, "w") as f:
            for i, step in enumerate(recipe["steps"]):
                # If step has use: then run it from action
                if "use" in step:
                    matched_steps = [
                        s for s in action["action"] if s["name"] == step["use"]
                    ]
                    if not matched_steps:
                        raise ValueError(f"Action step not found: {step['use']}")
                    if len(matched_steps) > 1:
                        raise ValueError(f"Multiple action steps found: {step['use']}")
                    step = matched_steps[0]

                logger.info(f"Run {i + 1}/{len(recipe['steps'])}: {step['name']}")
                os.chdir(Path(folder))
                command: list[str] = step["command"].split()
                args: list[str] = args_filter(step.get("args"))
                subprocess.run(command + args, check=True, stdout=f, stderr=f)

    finally:
        os.chdir(original_cwd)


def generate_modulefile(
    template: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str],
    config: dict,
) -> None:
    """Generate modulefile for the installed target."""
    logger.info(f"Generating modulefile: {Path(output)}")
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
    action_filename: str | os.PathLike[str],
    modulefile_template: str | os.PathLike[str],
    src: str | os.PathLike[str],
    no_cache: bool = False,
):
    build_info = BuildInfo.load()

    target_info = yaml.safe_load(Path(config_filename).read_text())
    build_recipe = yaml.safe_load(Path(recipe_filename).read_text())
    action = yaml.safe_load(Path(action_filename).read_text())

    name = target_info["name"]

    MODULES_DIR = Path(prefix) / name
    MODULEFILES_DIR = Path(prefix) / "modulefiles"

    # Download & Extract
    src_folders: list[Path] = download_and_extract_all(
        target_info["targets"], target_info["sources"], src, no_cache
    )

    for src, info in zip(src_folders, target_info["targets"], strict=True):
        install_prefix = MODULES_DIR / f"{name}_{info['category']}_{info['version']}"
        python_install_prefix = (
            install_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        variables = {
            "install_prefix": install_prefix,
            "python_install_prefix": python_install_prefix,
        }

        if no_cache:
            if install_prefix.exists():
                logger.info(f"Removing existing installation at {install_prefix}")
                subprocess.run(
                    ["rm", "-rf", str(install_prefix)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

        logger.info(f"Installing: {install_prefix}")

        # Build & Install
        try:
            build_target(src, build_recipe, variables, action=action)
        except subprocess.CalledProcessError as _e:
            logger.error(
                f"Build failed for {name} {info['version']} ({info['category']})"
            )
            logger.error(f"Check log file at: {Path(src) / 'log' / 'build.log'}")
            sys.exit(1)
        build_info.add_file(install_prefix)

        # Generate modulefiles
        lua_file = MODULEFILES_DIR / name / "{category}_{version}.lua".format(**info)
        generate_modulefile(
            template=modulefile_template,
            output=lua_file,
            config=variables,
        )
        build_info.add_file(lua_file)

        logger.info(f"Installed {name} {info['version']} ({info['category']})")

    build_info.save()


def get_default(filename: str | os.PathLike[str]) -> Path:
    filepath = Path(filename)
    if filepath.exists():
        return filepath

    return filepath.with_name(f"{filepath.stem}.default{filepath.suffix}")


def setup_args():
    THIS_SCRIPT_DIR = Path(__file__).parent.resolve()
    default_prefix = Path.home() / ".local" / "opt"
    default_recipe = get_default(
        THIS_SCRIPT_DIR / "config" / f"recipe-{platform.system().lower()}.yaml"
    )
    default_action = get_default(THIS_SCRIPT_DIR / "config" / "action.yaml")
    default_config = get_default(THIS_SCRIPT_DIR / "config" / "target.yaml")
    default_template = get_default(THIS_SCRIPT_DIR / "template" / "lammps.lua.jinja")

    default_src = THIS_SCRIPT_DIR / "src"
    parser = argparse.ArgumentParser(description="Build LAMMPS from source.")
    parser.add_argument(
        "--prefix",
        type=str,
        default=str(default_prefix),
        help="Installation prefix directory. (default: ~/.local/opt)",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=str(default_recipe),
        help="Path to the build recipe configuration file. "
        f"(default: {default_recipe.relative_to(THIS_SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to the target info configuration file. "
        f"(default: {default_config.relative_to(THIS_SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--action",
        type=str,
        default=str(default_action),
        help="Path to the action configuration file. "
        f"(default: {default_action.relative_to(THIS_SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=str(default_template),
        help="Path to the modulefile template."
        f"(default: {default_template.relative_to(THIS_SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(default_src),
        help="Path to the source directory. If not exists, source files will be downloaded here. "
        f"(default: {default_src.relative_to(THIS_SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). (default: ERROR)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of downloaded and extracted sources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()

    logger = logging.getLogger("modules builder")
    fmt = "%(asctime)s[%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    try:
        import colorlog

        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                f"%(log_color)s{fmt}%(reset)s",
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        assert Path(args.recipe).exists(), f"Recipe file not found: {args.recipe}"
        assert Path(args.config).exists(), f"Config file not found: {args.config}"
        assert Path(args.template).exists(), f"Template file not found: {args.template}"
    except AssertionError as e:
        logger.error(e)
        sys.exit(1)

    main(
        prefix=args.prefix,
        recipe_filename=args.recipe,
        config_filename=args.config,
        action_filename=args.action,
        modulefile_template=args.template,
        src=args.src,
        no_cache=args.no_cache,
    )

    print(f"""
All done! Modulefiles are located at: {Path(args.prefix) / "modulefiles"}
You can load the modulefiles using Lmod. For example:
    module use {Path(args.prefix) / "modulefiles"}
    module load <module_name>
""")
