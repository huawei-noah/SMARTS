import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__file__)


def install_evaluation_deps(requirements_dir: Path, reset_wheelhouse_cache: bool):
    wheelhouse = (
        Path(tempfile.gettempdir()) / "wheelhouse" / "driving_smarts_competition"
    )
    try:
        subprocess.check_call(["git", "lfs", "--version"])
    except:
        logger.exception("`git lfs` is required to correctly install the dependencies.")
        exit(1)
    if not wheelhouse.exists() or reset_wheelhouse_cache:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                f"--wheel-dir={str(wheelhouse)}",
                "-r",
                str(requirements_dir / "requirements.txt"),
            ]
        )

    stdout, _ = subprocess.Popen(
        ["ls", str(wheelhouse)], stdout=subprocess.PIPE
    ).communicate()
    files = stdout.decode(sys.getdefaultencoding())
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-index",
            "--no-deps",
            *(
                str(wheelhouse / dep)
                for dep in files.split("\n")
                if dep.endswith(".whl")
            ),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="auto-install-deps")
    parser.add_argument(
        "--no-cache",
        help="Resets the underlying wheelhouse cache. This is useful if there is a change in the dependencies.",
        action="store_true",
    )
    args = parser.parse_args()
    requirements_dir = (Path(__file__).parent).absolute()

    install_evaluation_deps(requirements_dir, args.no_cache)
