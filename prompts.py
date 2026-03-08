"""Persona and prompt management for the OCF bot."""

import os
import re
import json
from typing import Optional

from config import PERSONA_DIR, SETTINGS_DIR


def is_valid_persona_name(name: str) -> bool:
    """Ensures persona names are strictly lowercase alphanumeric.
    
    Args:
        name: The persona name to validate.
        
    Returns:
        True if the name is valid, False otherwise.
    """
    return bool(re.match(r"^[a-z0-9]+$", name))


def format_persona_prompt(base_prompt: str) -> str:
    """Ensures a persona prompt has the required template variables.
    
    Args:
        base_prompt: The base prompt text.
        
    Returns:
        The formatted prompt with context_str and query_str placeholders.
    """
    if "{context_str}" not in base_prompt and "{query_str}" not in base_prompt:
        return (
            f"{base_prompt}\n\n"
            "Context:\n---------\n{context_str}\n---------\n"
            "Query: {query_str}\nAnswer: "
        )
    return base_prompt


def get_user_default_persona(user_id: int) -> str:
    """Fetches the user's default persona name.
    
    Args:
        user_id: The Discord user ID.
        
    Returns:
        The persona name, defaults to 'default'.
    """
    settings_file = os.path.join(SETTINGS_DIR, f"{user_id}.json")
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("default_persona", "default")
        except Exception:
            pass
    return "default"


def set_user_default_persona(user_id: int, persona_name: str) -> None:
    """Sets the user's default persona.
    
    Args:
        user_id: The Discord user ID.
        persona_name: The persona name to set as default.
    """
    settings_file = os.path.join(SETTINGS_DIR, f"{user_id}.json")
    data = {}
    
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    
    data["default_persona"] = persona_name
    
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_persona_prompt(persona_name: str) -> str:
    """Loads the prompt for a given persona directly from its JSON file.
    
    Args:
        persona_name: The name of the persona.
        
    Returns:
        The formatted prompt template string.
    """
    file_path = os.path.join(PERSONA_DIR, f"{persona_name}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return format_persona_prompt(data["prompt"])
        except Exception:
            pass
    
    # Bare minimum fallback just in case the default.json is deleted or corrupted
    return "Context:\n---------\n{context_str}\n---------\nQuery: {query_str}\nAnswer: "


def persona_exists(name: str) -> bool:
    """Check if a persona exists.
    
    Args:
        name: The persona name to check.
        
    Returns:
        True if the persona exists, False otherwise.
    """
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")
    return os.path.exists(file_path)


def get_persona_data(name: str) -> Optional[dict]:
    """Get the full data for a persona.
    
    Args:
        name: The persona name.
        
    Returns:
        The persona data dict or None if not found.
    """
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_persona(name: str, creator_id: int, prompt: str) -> None:
    """Save a persona to disk.
    
    Args:
        name: The persona name.
        creator_id: The Discord user ID of the creator.
        prompt: The persona prompt text.
    """
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")
    data = {"creator_id": creator_id, "prompt": prompt}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def delete_persona(name: str) -> bool:
    """Delete a persona.
    
    Args:
        name: The persona name.
        
    Returns:
        True if deleted, False if not found.
    """
    file_path = os.path.join(PERSONA_DIR, f"{name}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


def list_personas() -> list[str]:
    """List all available persona names.
    
    Returns:
        List of persona names.
    """
    return [f[:-5] for f in os.listdir(PERSONA_DIR) if f.endswith(".json")]
