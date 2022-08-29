import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__file__)


def resolve_codalab_dirs(
    root_path: str,
    input_dir: str,
    output_dir: str,
    local: bool = False,
) -> Tuple[str, str, str]:
    """Returns appropriate directories for Codalab remote and local evaluation.

    Args:
        root_path (str): The path to the calling file.
        input_dir (str): The path containing the "res" and "ref" directories
            provided by CodaLab.
        output_dir (str): The path to output the scores.txt file.
        local (bool): If local directories should be used.

    Returns:
        Tuple[str, str, str]: The submission, evaluation-scenarios, and the
            scores directory, respectively. The submission directory contains
            the user submitted files, the evaluation-scenarios directory
            contains the contents of the unzipped evaluation scenarios, and the
            scores directory is where the scores.txt file is written.
    """
    logger.info(f"root_path={root_path}")
    logger.info(f"input_dir={input_dir}")
    logger.info(f"output_dir={output_dir}")

    if not local:
        submission_dir = os.path.join(input_dir, "res")
        evaluation_dir = os.path.join(input_dir, "ref")
    else:
        submission_dir = input_dir
        evaluation_dir = root_path
    scores_dir = output_dir

    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={submission_dir}")
    logger.info(f"evaluation_dir={evaluation_dir}")
    logger.info(f"scores_dir={scores_dir}")

    if not os.path.isdir(submission_dir):
        logger.error(f"submission_dir={submission_dir} does not exist.")

    return submission_dir, evaluation_dir, scores_dir


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    import yaml

    config = None
    if path.exists():
        with open(path, "r") as file:
            config = yaml.safe_load(file)
    return config


def merge_config(self: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    # If other is None or empty, return self.
    if not other:
        return self
    # Else, merge the two, with the other winning any tiebreakers.
    return {**self, **other}


def validate_config(config: Dict[str, Any], keys: Set[str]) -> bool:
    unaccepted_keys = {*config.keys()} - keys
    assert len(unaccepted_keys) == 0, f"Unaccepted config keys: {unaccepted_keys}"


def write_output(text, output_dir):
    if output_dir:
        with open(output_dir, "w") as file:
            file.write(text)
