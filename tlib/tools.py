import argparse
import logging


# A self-modified logger containing logger, parser, wandb, tensorboard
class TTools:
    def __init__(self):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d:%H:%M:%S",
            level=logging.INFO,
        )
        self._logger = logging.getLogger(__name__)
        self._parser = argparse.ArgumentParser()

    def add_arg(self, *args, **kwargs):
        kwargs["help"] = "(default: %(default)s)"
        if not kwargs.get("type", bool) == bool:
            kwargs["metavar"] = ""
        self._parser.add_argument(*args, **kwargs)

    def get_args(self):
        return self._parser.parse_args()

    @property
    def logger(self):
        return self._logger
