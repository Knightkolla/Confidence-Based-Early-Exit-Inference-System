from dataclasses import asdict

import yaml

from src.config.types import ExperimentConfig


class PrettyPrinter:
    def format(self, config: ExperimentConfig) -> str:
        """Serialize an ExperimentConfig to a valid YAML string."""
        data = asdict(config)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
