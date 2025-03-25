# ClaudeGPT

A dual-agent autonomous system that combines Claude and GPT models to work collaboratively on tasks.

## Overview

ClaudeGPT implements a collaborative AI architecture where:
- Claude acts as the primary autonomous agent (the "Executor")
- GPT acts as a supportive assistant (the "Muse"/"Guide")

The system enables these agents to work together on tasks, with Claude making decisions and executing plans while GPT provides reflection, alternative viewpoints, and guidance.

Inspired by BabyAGI concepts, this system implements a task-based approach to goal achievement with continuous reflection and improvement through dual-agent collaboration.

## Features

- **Dual-agent architecture**: Leverages the strengths of both Claude and GPT models
- **Task-based execution**: Breaks down goals into manageable tasks
- **Memory system**: Maintains context and history across interactions
- **Tool support**: Extensible with custom tools for specific capabilities
- **Reflection and improvement**: Continuous feedback loop between agents
- **Callback system**: Hook into key events in the system's operation

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install anthropic openai python-dotenv
   ```
3. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   CLAUDE_MODEL=claude-3-opus-20240229
   GPT_MODEL=gpt-4-turbo
   CLAUDE_TEMPERATURE=0.7
   GPT_TEMPERATURE=0.7
   MAX_TOKENS=4000
   ```

## Usage

### Basic Usage

```python
import asyncio
from claudeGPT import ClaudeGPT

async def main():
    # Initialize the system
    system = ClaudeGPT()
    
    # Set a goal and context
    goal = "Create a marketing plan for a new product"
    context = "The product is a smart home device targeting middle-income families."
    
    await system.set_goal(goal, context)
    
    # Run the system for 3 interaction cycles
    await system.run(cycles=3)
    
    # Get a summary of the results
    print(system.get_summary())
    
    # Export the complete results
    results = system.export_results()

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating Custom Tools

You can extend the system with custom tools:

```python
from claudeGPT import ClaudeGPT, Tool

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get weather information for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City or location name"
                }
            }
        )
    
    async def execute(self, **kwargs):
        location = kwargs.get("location")
        # In a real implementation, call a weather API here
        return {
            "location": location,
            "temperature": 72,
            "conditions": "Partly cloudy"
        }

# Initialize system with the custom tool
system = ClaudeGPT(tools=[WeatherTool()])
```

### Using Callbacks

You can register callbacks to hook into key events:

```python
async def on_task_complete(task_id, result):
    print(f"Task {task_id} completed with result: {result[:100]}...")

async def on_reflection(agent_type, reflection):
    print(f"{agent_type.upper()} reflection: {reflection[:100]}...")

# Initialize system with callbacks
system = ClaudeGPT(
    callbacks={
        "on_task_complete": on_task_complete,
        "on_reflection": on_reflection
    }
)
```

## Examples

The repository includes several example scripts:

- `simple_example.py`: Basic usage of the system
- `custom_tools_example.py`: Example of creating and using custom tools
- `advanced_example.py`: Advanced usage with custom tools and callbacks

Run any example with:

```
python simple_example.py
```

## Architecture

The system consists of several key components:

- **ClaudeGPT**: Main orchestrator for the dual-agent system
- **ClaudeAgent**: The primary autonomous agent (the "Executor")
- **GPTAgent**: The supportive assistant agent (the "Guide"/"Muse")
- **TaskManager**: Manages task creation, prioritization, and execution
- **Memory**: Manages shared memory between agents
- **Tool**: Base class for tools that can be used by the agents

## License

MIT