"""Parsing utilities for ParaDime.

The :mod:`paradime.utils.parsing` module implements functions to validate
specifications, parse them, and prepare them for later use.
"""

from typing import Any, Union

from cerberus import Validator
from ruamel.yaml import YAML

from paradime.exceptions import SpecificationError

schema = {
    "derived data": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": True},
                "data func": {
                    "type": "string",
                    "required": True,
                    "allowed": ["pca", "spectral"],
                },
                "keys": {
                    "type": "list",
                    "required": True,
                    "schema": {
                        "type": "list",
                        "items": [
                            {"type": "string", "allowed": ["data", "rels"]},
                            {"type": "string"},
                        ],
                    },
                },
                "options": {"type": "dict"},
            },
        },
    },
    "relations": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": True},
                "level": {
                    "type": "string",
                    "allowed": ["global", "batch"],
                    "required": True,
                },
                "type": {
                    "type": "string",
                    "required": True,
                    "allowed": [
                        "precomp",
                        "pdist",
                        "neighbor",
                        "pdistdiff",
                        "fromto",
                    ],
                },
                "field": {
                    "type": "string",
                },
                "options": {"type": "dict"},
                "transforms": {
                    "type": "list",
                    "required": False,
                    "schema": {
                        "type": "dict",
                        "schema": {
                            "type": {
                                "type": "string",
                                "required": True,
                                "allowed": [
                                    "symmetrize",
                                    "normalize",
                                    "normalize rows",
                                    "perplexity",
                                    "t-dist",
                                    "connect",
                                ],
                            },
                            "options": {"type": "dict"},
                        },
                    },
                },
            },
        },
    },
    "losses": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": True},
                "type": {
                    "type": "string",
                    "required": True,
                    "allowed": [
                        "relation",
                        "classification",
                        "reconstruction",
                        "position",
                    ],
                },
                "func": {
                    "type": "string",
                    "allowed": [
                        "mse",
                        "kl div",
                        "cross entropy",
                        "umap cross entropy",
                    ],
                },
                "keys": {
                    "type": "dict",
                    "schema": {
                        "data": {"type": "list", "schema": {"type": "string"}},
                        "rels": {"type": "list", "schema": {"type": "string"}},
                        "methods": {
                            "type": "list",
                            "schema": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
    "training phases": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "epochs": {"type": "integer"},
                "sampling": {
                    "type": "dict",
                    "schema": {
                        "type": {
                            "type": "string",
                            "allowed": ["item", "edge"],
                        },
                        "options": {"type": "dict"},
                    },
                },
                "optimizer": {
                    "type": "dict",
                    "schema": {
                        "type": {
                            "type": "string",
                            "required": True,
                            "allowed": ["adam", "sgd"],
                        },
                        "options": {
                            "type": "dict",
                        },
                    },
                },
                "loss": {
                    "type": "dict",
                    "required": True,
                    "schema": {
                        "components": {
                            "type": "list",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        "weights": {
                            "type": "list",
                            "schema": {"type": "float"},
                        },
                    },
                },
            },
        },
    },
}

spec_validator = Validator(schema=schema)


def validate_spec(file_or_spec: Union[str, dict]) -> dict[str, Any]:
    """Validates a ParaDime specification.

    Args:
        file_or_spec: The specification, either as a dictionary or as a path
            to a YAML/JSON file.

    Returns:
        The validated specification as a dictionary.

    Raises:
        :class:`paradime.exceptions.SpecificationError`: If the validation of
            the specification failed.
    """

    yaml = YAML(typ="safe")

    if isinstance(file_or_spec, dict):
        spec = file_or_spec
    elif isinstance(file_or_spec, str):
        with open(file_or_spec) as f:
            spec = yaml.load(f)
    else:
        raise TypeError(
            "Expected specification dict or path to YAML/JSON file."
        )

    if spec_validator.validate(spec):
        return spec
    else:
        raise SpecificationError(f"{spec_validator.errors}")
