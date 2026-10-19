"""Read and write observation settings alongside trained model archives."""

from __future__ import annotations

import json
from pathlib import Path

from config import ObservationConfig


def observation_metadata_path(model_path: Path) -> Path:
    """Return the metadata filename associated with a model archive."""
    return model_path.with_name(f"{model_path.stem}.observation.json")


def save_observation_config(
    model_path: Path,
    observation_config: ObservationConfig,
) -> Path:
    """Persist observation options next to a newly saved model."""
    metadata_path = observation_metadata_path(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "include_ghost_direction": observation_config.include_ghost_direction,
                "include_previous_pacman_action": observation_config.include_previous_pacman_action,
                "include_nearest_pellet": observation_config.include_nearest_pellet,
                "include_previous_action": observation_config.include_previous_action,
                "include_nearest_ghost": observation_config.include_nearest_ghost,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def load_observation_config(model_path: Path) -> ObservationConfig:
    """Load model observation settings; missing metadata means baseline."""
    metadata_path = observation_metadata_path(model_path)
    if not metadata_path.exists():
        return ObservationConfig()
    try:
        values = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid observation metadata in {metadata_path}: {error}"
        ) from error
    if not isinstance(values, dict):
        raise ValueError(f"Observation metadata must be a JSON object: {metadata_path}")
    legacy_keys = {
        "include_ghost_direction",
        "include_previous_pacman_action",
    }
    current_keys = legacy_keys | {
        "include_nearest_pellet",
        "include_previous_action",
    }
    extended_keys = current_keys | {"include_nearest_ghost"}
    expected = (
        extended_keys
        if set(values) == extended_keys
        else current_keys
        if set(values) == current_keys
        else legacy_keys
    )
    if set(values) != expected or any(
        not isinstance(values[key], bool) for key in expected
    ):
        raise ValueError(
            f"Observation metadata in {metadata_path} must contain boolean "
            f"fields: {sorted(expected)}."
        )
    return ObservationConfig(**values)
