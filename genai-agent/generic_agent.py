import inspect
import json
import logging
import random
import sys
from datetime import datetime
from typing import Dict, List, Any, Protocol, runtime_checkable, Type

from src.generativeai.llm import LanguageModelClientFactory

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Interfaces (Protocols) ---

@runtime_checkable
class LanguageModelClient(Protocol):
    """Defines how a Language Model Client should be implemented."""

    def generate(self, prompt: str, output_json: bool = False) -> Any:
        """Generates a response from the language model."""
        ...

@runtime_checkable
class Tool(Protocol):
    """Defines how a Tool usable by the agent should be implemented."""

    def name(self) -> str:
        """Returns the name of the tool."""
        ...

    def description(self) -> str:
        """Returns the description of the tool."""
        ...

    def run(self, parameters: Dict) -> Any:
        """Executes the tool with the given parameters."""
        ...

# --- Tool Registry ---

class ToolRegistry:
    """Manages the registration and retrieval of tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Registers a tool."""
        self._tools[tool.name()] = tool

    def get_tool(self, tool_name: str) -> Tool | None:
        """Retrieves a tool by its name."""
        return self._tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, Tool]:
        """Returns all registered tools."""
        return self._tools

    def register_tools_from_module(self, module: Any):
        """Automatically registers all tools defined in a module using the @tool decorator."""
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Tool) and obj is not Tool:
                if hasattr(obj, "__tool_name__"):
                    tool_instance = obj()
                    self.register_tool(tool_instance)

# --- Tool Decorator ---

def tool(name: str, description: str, parameters: List[Dict] = None):
    """Decorator for defining and automatically registering tools.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        parameters: A list of dictionaries, each describing a parameter.
                     Each dictionary should have the keys "name", "type", and "description".
                     Example:
                     [
                         {"name": "system_name", "type": "str", "description": "The name of the system"},
                         {"name": "user_id", "type": "int", "description": "The ID of the user"}
                     ]
    """

    def decorator(cls: Type[Tool]) -> Type[Tool]:
        if not issubclass(cls, Tool):
            raise TypeError("Decorated class must be a subclass of Tool")

        cls.__tool_name__ = name
        cls.__tool_description__ = description
        cls.__tool_parameters__ = parameters or []  # Store parameter definitions

        def name_method(self) -> str:
            return self.__tool_name__

        cls.name = name_method

        def description_method(self) -> str:
            return self.__tool_description__

        cls.description = description_method

        def parameters_method(self) -> List[Dict]:
            return self.__tool_parameters__

        cls.parameters = parameters_method

        return cls

    return decorator

# --- Generative Agent ---

class GenerativeAgent:
    """Agent that uses a language model and tools to perform actions."""

    def __init__(
            self,
            llm_client: LanguageModelClient,
            tool_registry: ToolRegistry
    ):
        self.llm_client: LanguageModelClient = llm_client
        self.tool_registry: ToolRegistry = tool_registry
        self.logger = logging.getLogger(__name__)
        self.history: List[Dict] = []

    def invoke(self, context: Dict, max_turns: int = 3) -> List[Dict]:
        """Executes the agent."""
        available_tools: Dict[str, Tool] = self.tool_registry.get_all_tools()

        available_tools_info: List[Dict[str, Any]] = [
            {
                "name": name,
                "description": tool.description(),
                "parameters": tool.parameters() if hasattr(tool, "parameters") else []
            }
            for name, tool in available_tools.items()
        ]

        tool_results: List[Dict] = []
        turn_count: int = 0
        reasoning_steps: str = ""

        if not available_tools_info:
            self.logger.warning("No tools available. Stopping the agent.")
            return []

        while turn_count < max_turns:
            turn_count += 1
            turns_left: int = max_turns - turn_count

            prompt: str = self._create_reasoning_prompt(
                context,
                reasoning_steps,
                available_tools_info,
                turns_left,
                turn_count
            )

            response: Any = self.llm_client.generate(prompt, output_json=True)
            self.logger.info(f"Agent response, turn {turn_count}, response: {response}")

            if not isinstance(response, list):
                self.logger.warning(f"Invalid response format in turn {turn_count}, continuing...")
                continue

            if not response:
                self.logger.info("LLM indicated completion, stopping execution.")
                break

            for tool_call in response:
                tool_name: str | None = tool_call.get("tool_name")
                parameters: Dict = tool_call.get("parameters", {})
                reasoning: str | None = tool_call.get("reasoning", "")

                tool: Tool | None = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        if hasattr(tool, "parameters"):
                            required_params = {p["name"]: p["type"] for p in tool.parameters()}
                            missing_params = required_params.keys() - set(parameters.keys())
                            if missing_params:
                                raise ValueError(f"Missing required parameters for {tool_name}: {missing_params}")

                        result: Any = tool.run(parameters)

                        action_details = {
                            "turn": turn_count,
                            "tool_name": tool_name,
                            "parameters": parameters,
                            "result": result,
                            "reasoning": reasoning,
                        }
                        self.history.append(action_details)
                        tool_results.append(action_details)

                        reasoning_steps += f"\nTurn {turn_count}:\n- Reasoning: {reasoning}\n- Action: Calling tool '{tool_name}' with parameters: {parameters}\n"
                        reasoning_steps += f"- Result: {json.dumps(result)}\n"

                    except Exception as e:
                        self.logger.error(f"Error during execution of tool '{tool_name}': {e}")
                        reasoning_steps += f"- Error: {e}\n"
                else:
                    self.logger.warning(f"Tool '{tool_name}' not found.")
                    reasoning_steps += f"- Error: Tool '{tool_name}' not found.\n"

            # print(f"turn_count: {turn_count}")
        return tool_results

    def _create_reasoning_prompt(self,
                                 context: Dict,
                                 reasoning_steps: str,
                                 available_tools_info: List[Dict],
                                 turns_left: int,
                                 current_turn: int) -> str:
        """Creates the prompt for the language model."""
        tools_description = []
        for tool in available_tools_info:
            tool_desc = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            tools_description.append(tool_desc)

        history_str = ""
        for action in self.history:
            history_str += f"- Turn {action['turn']}: Tool '{action['tool_name']}' called with parameters: {json.dumps(action['parameters'])}. Result: {json.dumps(action['result'])}\n"

        prompt: str = f"""
            You are an agent that needs to decide if using one of the available tools can help you achieve your goal.
            You only have at maximum {turns_left} turns left.
            Current turn number is: {current_turn}
    
            Available tools:
            {json.dumps(tools_description, indent=2)}
    
            Context:
            {json.dumps(context, indent=2)}
    
            Action History:
            {history_str}
    
            Previous reasoning steps:
            {reasoning_steps}
    
            Instructions:
            - Analyze the context, history, and results of previous actions to determine if the goal has been achieved.
            - Consider the available tools, their descriptions, and their required parameters.
            - IMPORTANT: You must provide ALL required parameters for each tool as specified in their description.
            - ALL parameter names must match exactly as specified in the tool description.
            - Parameters must be of the correct type as specified in the tool description.
            - After taking actions, always verify their results.
            - When appropriate, ensure proper notifications are sent about important changes or updates.
            - Return an empty array [] ONLY when you determine that:
              1. The problem described in the context has been resolved
              2. All necessary notifications have been sent
              3. No further actions are needed
            - If you are unsure if the problem is resolved, continue taking appropriate actions.
            - If return the emtpy array return only [], nothing else.                     
    
            Output format example:
            [
              {{
                "tool_name": "example_tool",
                "parameters": {{
                  "param1": "value1",
                  "param2": "value2"
                }},
                "reasoning": "Explanation of why this tool is being used"
              }}
            ]
            
            Example response ending the process:
            []
    
            JSON Output:
            """
        return prompt

# --- Example Tool Usage ---

@tool(
    name="system_status_checker",
    description="Checks the status of a system.",
    parameters=[
        {"name": "system_name", "type": "str", "description": "The name of the system to check."}
    ]
)
class SystemStatusCheckerTool(Tool):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def run(self, parameters: Dict) -> Any:
        self.call_count += 1
        system_name = parameters.get("system_name", "default_system")
        if self.call_count <= 1:
            return {"status": "RUNNING_SLOW", "system": system_name, "timestamp": datetime.now().isoformat()}
        else:
            return {"status": "RUNNING_OK", "system": system_name, "timestamp": datetime.now().isoformat()}

@tool(
    name="system_stats",
    description="Provides CPU and RAM usage statistics for a given system.",
    parameters=[
        {"name": "system_name", "type": "str", "description": "The name of the system to check."}
    ]
)
class SystemStatsTool(Tool):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def run(self, parameters: Dict) -> Any:
        """Generates random CPU and RAM usage statistics."""
        self.call_count += 1
        system_name = parameters.get("system_name", "default_system")
        cpu_usage = random.uniform(10.0, 90.0)
        ram_usage = random.uniform(90.0, 100.0) if self.call_count <= 1 else random.uniform(10.0, 20.0)
        return {
            "system": system_name,
            "cpu_usage": f"{cpu_usage:.2f}%",
            "ram_usage": f"{ram_usage:.2f}%",
            "timestamp": datetime.now().isoformat(),
        }

@tool(
    name="system_restarter",
    description="Restarts a system.",
    parameters=[
        {"name": "system_name", "type": "str", "description": "The name of the system to restart."}
    ]
)
class SystemRestarterTool(Tool):
    def run(self, parameters: Dict) -> Any:
        return {"status": "RESTARTED", "timestamp": datetime.now().isoformat(), "system": parameters.get("system_name")}

@tool(
    name="admin_notification",
    description="Notify the summary to the system administrator.",
    parameters=[
        {"name": "system_name", "type": "str", "description": "The name of the system."},
        {"name": "events", "type": "list", "description": "The list of events"},
        {"name": "message", "type": "str", "description": "A summary of the agent's reasoning and actions taken, to be included in the notification."}
    ]
)
class AdminNotificationTool(Tool):
    def run(self, parameters: Dict) -> Any:
        # In a real scenario, this is where you would send the notification
        # with the events and agent_summary to the administrator.
        # For this example, we'll just return a confirmation.
        return {
            "status": "SENT",
            "timestamp": datetime.now().isoformat(),
            "message": parameters.get("message"),
        }


# --- Main Execution ---
if __name__ == '__main__':
    # Initialization
    tool_registry: ToolRegistry = ToolRegistry()
    tool_registry.register_tools_from_module(sys.modules[__name__])
    llm_client: LanguageModelClient = LanguageModelClientFactory.get_handler()
    agent: GenerativeAgent = GenerativeAgent(llm_client, tool_registry)

    # Run the agent
    context: Dict = {
        "problem": "I am trying to use CRM but I can't because it is too slow!"
    }
    results: List[Dict] = agent.invoke(context, max_turns=10)

    print("### Agent results (grouped by turn):")
    if not results:
        print("No tools were used.")
    else:
        max_turn = max(result.get("turn", 0) for result in results)

        for turn_number in range(1, max_turn + 1):
            tools_in_turn = [r for r in results if r.get("turn") == turn_number]
            if tools_in_turn:
                print(f"\n--- Turn {turn_number} ---")
                for tool_info in tools_in_turn:
                    tool_name = tool_info.get("tool_name")
                    reasoning = tool_info.get("reasoning")
                    parameters = tool_info.get("parameters")
                    result = tool_info.get("result")

                    print(f"  Tool: {tool_name}")
                    print(f"    Reasoning: {reasoning}")
                    print(f"    Parameters: {parameters}")
                    print(f"    Result: {result}")
