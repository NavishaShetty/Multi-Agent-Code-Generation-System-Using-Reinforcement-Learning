"""
Coder Agent - Writes clean Python code based on plans.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from agents.base_agent import BaseAgent
from communication.blackboard import Blackboard, MessageType


class CoderAgent(BaseAgent):
    """
    Coder agent that writes Python code based on plans and task descriptions.

    The Coder reads the plan from the blackboard (if available) and generates
    clean, working Python code.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(name="coder", api_key=api_key, model=model)

    def _get_system_prompt(self) -> str:
        return """You are a coding agent. Write clean, working Python code.

Rules:
1. Output ONLY Python code in ```python blocks
2. Include docstrings for functions
3. Handle edge cases mentioned in the plan
4. Keep the code simple and readable
5. Do NOT include any explanations outside the code block

Example output format:
```python
def function_name(params):
    \"\"\"Docstring.\"\"\"
    # implementation
    return result
```"""

    def generate_code(self, task: str, blackboard: Blackboard) -> str:
        """
        Generate code for the given task, using plan from blackboard if available.

        Args:
            task: The coding task description
            blackboard: Blackboard to read plan from and post code to

        Returns:
            The generated code
        """
        # Check if there's a plan available
        plan_msg = blackboard.get_latest_by_type(MessageType.PLAN)

        if plan_msg:
            prompt = f"""Task: {task}

Plan to follow:
{plan_msg.content}

Write the Python code implementing this plan. Output ONLY the code in a ```python block."""
        else:
            prompt = f"""Task: {task}

Write clean Python code to accomplish this task. Output ONLY the code in a ```python block."""

        response = self.call_llm(prompt)
        code = self.extract_code(response)

        # Post to blackboard
        blackboard.post(
            sender=self.name,
            content=code,
            message_type=MessageType.CODE
        )

        return code


if __name__ == "__main__":
    from communication.blackboard import Blackboard

    bb = Blackboard()
    task = "Write a function that reverses a string"
    bb.set_task(task)

    print("Testing CoderAgent...")
    try:
        agent = CoderAgent()
        code = agent.generate_code(task, bb)
        print(f"Generated code:\n{code}")
        print("\n✓ Coder agent working!")
    except Exception as e:
        print(f"Error (API key may not be set): {e}")
