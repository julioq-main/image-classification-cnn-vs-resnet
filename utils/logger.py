"""

"""

import logging
import sys
from pathlib import Path

def set_logger(log_level: str, log_file: str = None):
    
    level = getattr(logging, log_level.upper(), "INFO")
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)s | %(name)s : %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=handlers
                        )