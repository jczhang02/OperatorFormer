import hydra
from rich import print

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


@hydra.main(version_base=None, config_path="../configs/model", config_name="operatorformer")
def my_app(cfg):
    print(cfg)
    model = hydra.utils.instantiate(cfg)
    print(model)


if __name__ == "__main__":
    my_app()
