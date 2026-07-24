"""Shared configuration schemas and command-line parsing."""

from collections.abc import Sequence
from typing import TypeVar

from jsonargparse import ActionConfigFile, ArgumentParser
from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Base for immutable configuration with documented, validated fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_attribute_docstrings=True,
        validate_default=True,
    )


ConfigT = TypeVar("ConfigT", bound=ConfigModel)


def parse_and_validate(config_type: type[ConfigT], args: Sequence[str] | None = None) -> ConfigT:
    """Load YAML and command-line overrides into a configuration schema.

    Parameters
    ----------
    config_type
        Pydantic schema defining the accepted configuration.
    args
        Command-line arguments to parse, or the process arguments when omitted.

    Returns
    -------
    ConfigT
        Parsed and validated immutable configuration.
    """
    parser = ArgumentParser(description=config_type.__doc__)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(config_type)
    parsed_config = parser.parse_args(args)
    values = parser.instantiate(parsed_config).as_dict()
    values.pop("config", None)
    return config_type.model_validate(values)
