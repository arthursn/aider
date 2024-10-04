import importlib
import os
import warnings
from typing import List, Dict

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

AIDER_SITE_URL = "https://aider.chat"
AIDER_APP_NAME = "Aider"

os.environ["OR_SITE_URL"] = AIDER_SITE_URL
os.environ["OR_APP_NAME"] = AIDER_APP_NAME
os.environ["LITELLM_MODE"] = "PRODUCTION"

# `import litellm` takes 1.5 seconds, defer it!


class LazyLiteLLM:
    _lazy_module = None

    @staticmethod
    def ensure_alternating_roles(
        messages: List[Dict[str, str]],
        filler_messages: Dict[str, str] = {
            "user": "",
            "assistant": "",
        },
    ) -> List[Dict[str, str]]:
        """
        Ensures that the sequence of messages alternates between 'user' and
        'assistant' roles, inserting filler messages if necessary.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with
            'role' and 'content' keys.
            filler_messages (Dict[str, str], optional): Filler messages for
            'user' and 'assistant' roles. Defaults to {"user": "", "assistant": ""}.

        Returns:
            List[Dict[str, str]]: List of message dictionaries with alternating
            roles.
        """
        fixed_messages = []
        allowed_roles = ["system", "user", "assistant"]
        previous_role = None
        for msg in messages:
            role = msg["role"]
            assert role in allowed_roles, f"Role {role} not allowed"
            if role == previous_role:
                filler_role = "assistant" if role == "user" else "user"
                fixed_messages.append(
                    {
                        "role": filler_role,
                        "content": filler_messages[filler_role],
                    }
                )
            fixed_messages.append(msg)
            previous_role = msg["role"]

        return fixed_messages

    def __getattr__(self, name):
        if name == "_lazy_module":
            return super()
        self._load_litellm()
        return getattr(self._lazy_module, name)

    def _load_litellm(self):
        if self._lazy_module is not None:
            return

        self._lazy_module = importlib.import_module("litellm")

        self._lazy_module.suppress_debug_info = True
        self._lazy_module.set_verbose = False
        self._lazy_module.drop_params = True
        self._lazy_module._logging._disable_debugging()


litellm = LazyLiteLLM()

__all__ = [litellm]
