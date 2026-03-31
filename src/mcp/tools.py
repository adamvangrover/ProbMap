from typing import Callable, Dict, Any

class ToolRegistry:
    """
    ToolRegistry stores and manages the deterministic execution layer tools
    for the Model Context Protocol (MCP).
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """
        Registers a generic tool with a name.
        """
        if name in self._tools:
            raise ValueError(f"Tool {name} is already registered.")
        self._tools[name] = func

    def execute(self, name: str, **kwargs) -> Any:
        """
        Executes a registered tool securely.
        """
        if name not in self._tools:
            raise ValueError(f"Tool {name} is not registered.")

        try:
            return self._tools[name](**kwargs)
        except Exception as e:
            return {"error": str(e)}

# Instantiate a global registry
registry = ToolRegistry()

# Example Tool
def calculate_risk(portfolio_value: float, risk_factor: float) -> float:
    return portfolio_value * risk_factor

registry.register("calculate_risk", calculate_risk)
