import inspect
import json
import logging
import random
import sys
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Dict, List, Any, Protocol, runtime_checkable, Type

from src.generativeai.llm import LanguageModelClientFactory

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Abstract Classes ---

class LanguageModelClient(ABC):
    """Defines how a Language Model Client should be implemented."""

    @abstractmethod
    def generate(self, prompt: str, output_json: bool = False) -> Any:
        """Generates a response from the language model."""
        pass

class Tool(ABC):
    """Defines how a Tool usable by the agent should be implemented."""

    @abstractmethod
    def run(self, parameters: Dict) -> Any:
        """Executes the tool with the given parameters."""
        pass

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
            tool_registry: ToolRegistry,
            system_prompt: str = None,
            additional_instructions: str = None,
            enabled_tools: List[str] = None
    ):
        self.llm_client: LanguageModelClient = llm_client
        self.tool_registry: ToolRegistry = tool_registry
        self.logger = logging.getLogger(__name__)
        self.history: List[Dict] = []
        self.system_prompt: str = system_prompt or "You are an agent "
        self.additional_instructions: str = additional_instructions or ""
        self.enabled_tools: List[str] = enabled_tools or list(self.tool_registry.get_all_tools().keys())

    def invoke(self, context: Dict, max_turns: int = 3) -> Dict:
        """
        Executes the agent and returns a JSON output summarizing the agent's actions and outcome.

        Args:
            context (Dict): The initial context for the agent.
            max_turns (int): The maximum number of turns the agent can take.

        Returns:
            Dict: A JSON output summarizing the agent's actions, the final status, and any errors encountered.
        """
        available_tools: Dict[str, Tool] = {
            name: tool
            for name, tool in self.tool_registry.get_all_tools().items()
            if name in self.enabled_tools
        }

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
        outcome = {
            "success": False,
            "output": "",
            "reasoning": "",
            "actions": [],
            "errors": []
        }

        if not available_tools_info:
            self.logger.warning("No tools available. Stopping the agent.")
            outcome["reasoning"] = "No tools available."
            return outcome

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
                error_message = f"Invalid response format in turn {turn_count}"
                self.logger.warning(error_message)
                outcome["errors"].append(error_message)
                continue

            if not response:
                self.logger.info("LLM indicated completion, stopping execution.")
                outcome["success"] = True
                outcome["reasoning"] = "Agent completed the task successfully."
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
                        outcome["actions"].append(action_details)

                        reasoning_steps += f"\nTurn {turn_count}:\n- Reasoning: {reasoning}\n- Action: Calling tool '{tool_name}' with parameters: {parameters}\n"
                        reasoning_steps += f"- Result: {json.dumps(result, default=json_serial)}\n"
                        outcome["output"] = json.dumps(result, default=json_serial)

                    except Exception as e:
                        error_message = f"Error during execution of tool '{tool_name}': {e}"
                        self.logger.error(error_message)
                        outcome["errors"].append(error_message)
                        reasoning_steps += f"- Error: {e}\n"
                else:
                    error_message = f"Tool '{tool_name}' not found."
                    self.logger.warning(error_message)
                    outcome["errors"].append(error_message)
                    reasoning_steps += f"- Error: Tool '{tool_name}' not found.\n"

        # Add final reasoning if the loop completes without a definitive outcome
        if not outcome["success"]:
            outcome["reasoning"] = "Agent reached the maximum number of turns without completing the task."

        return outcome

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
            history_str += f"- Turn {action['turn']}: Tool '{action['tool_name']}' called with parameters: {json.dumps(action['parameters'], default=json_serial)}. Result: {json.dumps(action['result'], default=json_serial)}\n"

        # Costruisci il prompt con system prompt e istruzioni aggiuntive
        prompt: str = ""
        if self.system_prompt:
            prompt += f"System Prompt:\n{self.system_prompt}\n\n"

        prompt += f"""
            Decide if using one of the available tools can help you achieve your goal.
            You only have at maximum {turns_left} turns left.
            Current turn number is: {current_turn}

            Available tools:
            {json.dumps(tools_description, indent=2)}

            Context:
            {json.dumps(context, indent=2, default=json_serial)}

            Action History:
            {history_str}

            Previous reasoning steps:
            {reasoning_steps}
            """

        if self.additional_instructions:
            prompt += f"\nAdditional Instructions:\n{self.additional_instructions}\n"

        prompt += """
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

    system_prompt = "You are an expert IT support agent. Your task is to use the provided tools to solve user problems efficiently and accurately."
    additional_instructions = "After any action, notify the administrator using the 'admin_notification' tool."


    agent: GenerativeAgent = GenerativeAgent(llm_client, tool_registry, system_prompt, additional_instructions,
                                             # enabled_tools=["system_status_checker", "admin_notification"] 
                                             )

    # Run the agent
    context: Dict = {
        "problem": "CRM System is slow"
    }
    outcome: Dict = agent.invoke(context, max_turns=10)

    print("### Agent results:")
    print(json.dumps(outcome, indent=2, default=json_serial))
