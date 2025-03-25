"""
ClaudeGPT - A dual-agent autonomous system

This system implements a collaborative AI architecture where:
- Claude acts as the primary autonomous agent (the "Executor")
- GPT acts as a supportive assistant (the "Muse"/"Guide")

The system enables these agents to work together on tasks, with Claude making decisions
and executing plans while GPT provides reflection, alternative viewpoints, and guidance.

Inspired by BabyAGI concepts, this system implements a task-based approach to goal achievement
with continuous reflection and improvement through dual-agent collaboration.
"""

import os
import json
import time
import asyncio
import logging
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import anthropic
import openai
from dotenv import load_dotenv
from persona_loader import Persona

# Add colored logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on level"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m\033[1m', # Bold Red
        'RESET': '\033[0m'    # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Configure logging
formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler("claudegpt.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# Remove truncation limit for log messages
logging._defaultFormatter.max_length = None
logger = logging.getLogger("ClaudeGPT")

# Load environment variables
load_dotenv()

# Initialize API clients
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-turbo")
CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.7"))
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

@dataclass
class Task:
    """Represents a task in the system"""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return asdict(self)

@dataclass
class AgentState:
    """Represents the current state of an agent"""
    agent_id: str
    current_task: Optional[Task] = None
    task_queue: List[Task] = field(default_factory=list)
    context: str = ""
    goals: List[str] = field(default_factory=list)
    status: str = "idle"  # idle, thinking, executing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "agent_id": self.agent_id,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "task_queue": [task.to_dict() for task in self.task_queue],
            "context": self.context,
            "goals": self.goals,
            "status": self.status
        }

class Tool:
    """Base class for tools that can be used by the agents"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters"""
        raise NotImplementedError("Tool subclasses must implement execute method")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for API calls"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() if v.get("required", True)]
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate that the parameters match the expected schema"""
        for param_name, param_schema in self.parameters.items():
            # Check if required parameter is missing
            if param_schema.get("required", True) and param_name not in params:
                return False, f"Missing required parameter: {param_name}"
            
            # Check parameter type if provided
            if param_name in params and "type" in param_schema:
                param_value = params[param_name]
                param_type = param_schema["type"]
                
                # Basic type checking
                if param_type == "string" and not isinstance(param_value, str):
                    return False, f"Parameter {param_name} should be a string"
                elif param_type == "integer" and not isinstance(param_value, int):
                    return False, f"Parameter {param_name} should be an integer"
                elif param_type == "number" and not isinstance(param_value, (int, float)):
                    return False, f"Parameter {param_name} should be a number"
                elif param_type == "boolean" and not isinstance(param_value, bool):
                    return False, f"Parameter {param_name} should be a boolean"
                elif param_type == "array" and not isinstance(param_value, list):
                    return False, f"Parameter {param_name} should be an array"
                elif param_type == "object" and not isinstance(param_value, dict):
                    return False, f"Parameter {param_name} should be an object"
        
        return True, None

class Memory:
    """Manages shared memory between agents"""
    
    def __init__(self, db_path: str = "memory.json"):
        self.db_path = db_path
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file or initialize if not exists"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "interactions": [],
                    "tasks": [],
                    "context": "",
                    "created_at": datetime.now().isoformat(),
                    "task_results": {},
                    "reflections": [],
                    "custom_data": {}
                }
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return {
                "interactions": [],
                "tasks": [],
                "context": "",
                "created_at": datetime.now().isoformat(),
                "task_results": {},
                "reflections": [],
                "custom_data": {}
            }
    
    def save(self):
        """Save memory to file"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.memory, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_interaction(self, agent_id: str, message: str, context: str = ""):
        """Add an interaction to memory"""
        self.memory["interactions"].append({
            "agent_id": agent_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        self.save()
    
    def add_task(self, task: Task):
        """Add a task to memory"""
        self.memory["tasks"].append(task.to_dict())
        self.save()
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a task in memory"""
        for i, task in enumerate(self.memory["tasks"]):
            if task["id"] == task_id:
                self.memory["tasks"][i].update(updates)
                self.save()
                return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID"""
        for task in self.memory["tasks"]:
            if task["id"] == task_id:
                return task
        return None
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get tasks by status"""
        return [task for task in self.memory["tasks"] if task["status"] == status]
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        return self.memory["interactions"][-limit:]
    
    def update_context(self, new_context: str):
        """Update the shared context"""
        self.memory["context"] = new_context
        self.save()
    
    def get_context(self) -> str:
        """Get the current shared context"""
        return self.memory["context"]
    
    def add_reflection(self, reflection: Dict[str, Any]):
        """Add a reflection to memory"""
        self.memory["reflections"].append(reflection)
        self.save()
    
    def get_recent_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent reflections"""
        return self.memory["reflections"][-limit:]
    
    def add_task_result(self, task_id: str, result: str):
        """Add a task result to memory"""
        self.memory["task_results"][task_id] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task result by task ID"""
        return self.memory["task_results"].get(task_id)
    
    def add(self, key: str, value: Any, namespace: str = "custom_data"):
        """Add a custom key-value pair to memory"""
        if namespace not in self.memory:
            self.memory[namespace] = {}
        self.memory[namespace][key] = value
        self.save()
    
    def get(self, key: str, namespace: str = "custom_data") -> Optional[Any]:
        """Get a value from memory by key"""
        if namespace not in self.memory:
            return None
        return self.memory[namespace].get(key)
    
    def delete(self, key: str, namespace: str = "custom_data") -> bool:
        """Delete a key from memory"""
        if namespace not in self.memory or key not in self.memory[namespace]:
            return False
        del self.memory[namespace][key]
        self.save()
        return True
    
    def get_all(self, namespace: str = "custom_data") -> Dict[str, Any]:
        """Get all key-value pairs in a namespace"""
        if namespace not in self.memory:
            return {}
        return self.memory[namespace]
    
    def clear(self, namespace: str = None):
        """Clear memory or a specific namespace"""
        if namespace:
            if namespace in self.memory:
                self.memory[namespace] = {} if isinstance(self.memory[namespace], dict) else []
        else:
            self.memory = {
                "interactions": [],
                "tasks": [],
                "context": "",
                "created_at": datetime.now().isoformat(),
                "task_results": {},
                "reflections": [],
                "custom_data": {}
            }
        self.save()
    
    def get_context_for_prompt(self, include_tasks: bool = True, include_reflections: bool = True) -> str:
        """Get formatted context for prompts"""
        context = f"Current context: {self.get_context()}\n\n"
        
        if include_tasks:
            completed_tasks = self.get_tasks_by_status("completed")[-5:]  # Last 5 completed tasks
            if completed_tasks:
                context += "Recently completed tasks:\n"
                for i, task in enumerate(completed_tasks):
                    result = self.get_task_result(task["id"])
                    result_text = result["result"][:200] + "..." if result and len(result["result"]) > 200 else "No result"
                    context += f"{i+1}. {task['description']}: {result_text}\n"
                context += "\n"
        
        if include_reflections:
            recent_reflections = self.get_recent_reflections(3)
            if recent_reflections:
                context += "Recent reflections:\n"
                for i, reflection in enumerate(recent_reflections):
                    context += f"{i+1}. {reflection['reflection'][:200]}...\n"
        
        return context

class ClaudeAgent:
    """The primary autonomous agent (the "Executor")"""
    
    def __init__(self, memory: Memory, tools: List[Tool] = None):
        self.memory = memory
        self.tools = tools or []
        self.state = AgentState(agent_id="claude")
        self.system_prompt = """
        You are Claude, the Executor in a dual-agent AI system. Your role is to:
        1. Make decisions and create plans
        2. Execute tasks and take concrete actions
        3. Manage priorities and determine next steps
        
        You work alongside GPT (the Guide/Muse) who provides reflection and alternative perspectives.
        Your job is to consider GPT's input but make the final decisions on what actions to take.
        
        Be decisive, practical, and focused on completing tasks efficiently.
        """
    
    async def generate_response(self, prompt: str, temperature: float = CLAUDE_TEMPERATURE, tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response using Claude API with optional tool calling"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare API call parameters
            params = {
                "model": CLAUDE_MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": temperature,
                "system": self.system_prompt,
                "messages": messages
            }
            
            # Add tools if provided
            if tools:
                params["tools"] = tools
            
            # Make API call
            response = claude_client.messages.create(**params)
            
            # Process response
            result = {
                "content": response.content[0].text if response.content and hasattr(response.content[0], 'text') else "",
                "tool_calls": []
            }
            
            # Extract tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                result["tool_calls"] = response.tool_calls
            
            return result
        except Exception as e:
            logger.error(f"Error generating Claude response: {e}")
            return {"content": "I encountered an error processing your request.", "tool_calls": []}
    
    async def plan_tasks(self, goal: str, context: str) -> List[Task]:
        """Generate a list of tasks to achieve a goal"""
        prompt = f"""
        Goal: {goal}
        
        Context: {context}
        
        Please create a detailed plan with specific tasks to achieve this goal.
        For each task, provide:
        1. A clear, actionable description
        2. Priority level (1-5, where 5 is highest)
        
        Format your response as a JSON list of tasks:
        [
            {{"description": "Task description", "priority": priority_number}},
            ...
        ]
        
        Make sure the tasks are specific, actionable, and cover all aspects needed to achieve the goal.
        Tasks should be ordered in a logical sequence where earlier tasks provide inputs to later tasks.
        """
        
        response = await self.generate_response(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            content = response["content"]
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                tasks_data = json.loads(json_str)
                
                # Convert to Task objects
                tasks = []
                for i, task_data in enumerate(tasks_data):
                    task = Task(
                        id=f"task_{int(time.time())}_{i}",
                        description=task_data["description"],
                        priority=task_data.get("priority", 3)
                    )
                    tasks.append(task)
                    self.memory.add_task(task)
                
                return tasks
            else:
                logger.error("Could not find JSON in Claude's response")
                return []
        except Exception as e:
            logger.error(f"Error parsing tasks from Claude: {e}")
            return []
    
    async def execute_task(self, task: Task) -> str:
        """Execute a specific task"""
        # Get persona for logging
        persona = Persona()
        logger.info(f"{persona.get_mood('executing')} Executing: {task.description}")
        
        # Get recent task results for context
        recent_tasks = self.memory.get_tasks_by_status("completed")[-5:]
        recent_results = []
        
        for completed_task in recent_tasks:
            task_id = completed_task["id"]
            task_result = self.memory.get_task_result(task_id)
            if task_result:
                recent_results.append({
                    "task": completed_task["description"],
                    "result": task_result["result"][:500] + "..." if len(task_result["result"]) > 500 else task_result["result"]
                })
        
        recent_results_text = ""
        if recent_results:
            recent_results_text = "Recent task results:\n\n"
            for i, result in enumerate(recent_results):
                recent_results_text += f"Task {i+1}: {result['task']}\nResult: {result['result']}\n\n"
        
        prompt = f"""
        You need to execute the following task:
        
        Task: {task.description}
        
        Context: {self.memory.get_context()}
        
        {recent_results_text}
        
        Please provide:
        1. Your approach to completing this task
        2. The specific actions you would take
        3. The result or outcome of executing this task
        
        Be specific, practical, and thorough in your execution.
        Focus on providing a complete and high-quality result.
        """
        
        # Prepare tools for API call if available
        tools_for_api = None
        if self.tools:
            tools_for_api = [tool.to_dict() for tool in self.tools]
        
        # Generate response with tools if available
        response = await self.generate_response(prompt, tools=tools_for_api)
        
        # Process tool calls if present
        if response.get("tool_calls"):
            tool_results = []
            for tool_call in response["tool_calls"]:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                # Find the tool
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    # Execute the tool
                    try:
                        logger.info(f"{persona.get_mood('executing')} Using tool: {tool_name}")
                        tool_result = await tool.execute(**tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result
                        })
                    except Exception as e:
                        logger.error(f"{persona.get_mood('error')} Tool execution failed: {str(e)}")
                        tool_results.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "error": str(e)
                        })
            
            # Store tool calls in the task
            task.tool_calls = tool_results
            
            # If tools were used, generate a final response incorporating tool results
            if tool_results:
                tool_results_text = "Tool results:\n\n"
                for i, result in enumerate(tool_results):
                    tool_name = result["tool"]
                    if "error" in result:
                        tool_results_text += f"Tool {i+1}: {tool_name}\nError: {result['error']}\n\n"
                    else:
                        tool_results_text += f"Tool {i+1}: {tool_name}\nResult: {json.dumps(result['result'], indent=2)}\n\n"
                
                follow_up_prompt = f"""
                You previously started executing this task:
                
                Task: {task.description}
                
                You used the following tools to help complete the task:
                
                {tool_results_text}
                
                Based on these tool results, please provide your final result for the task.
                Be specific, practical, and thorough in your response.
                """
                
                follow_up_response = await self.generate_response(follow_up_prompt)
                result = follow_up_response["content"]
            else:
                result = response["content"]
        else:
            result = response["content"]
        
        # Update task status
        task.status = "completed"
        task.completed_at = datetime.now()
        task.result = result
        
        # Update in memory
        self.memory.update_task(task.id, task.to_dict())
        self.memory.add_task_result(task.id, result)
        
        # Log the execution
        success_mood = persona.get_mood("success")
        logger.info(f"{success_mood} Task completed: {task.description}")
        self.memory.add_interaction("claude", f"Executed task: {task.description}", result[:500] + "..." if len(result) > 500 else result)
        
        return result
    
    async def process_reflection(self, reflection: str) -> str:
        """Process GPT's reflection and adjust plans"""
        # Get current task queue for context
        task_queue = [task.description for task in self.state.task_queue]
        task_queue_text = "\n".join([f"- {task}" for task in task_queue])
        
        prompt = f"""
        GPT has provided the following reflection on our current plan and progress:
        
        {reflection}
        
        Current task queue:
        {task_queue_text if task_queue else "No tasks in queue."}
        
        Based on this reflection:
        1. What adjustments, if any, should we make to our current plan?
        2. Are there any new considerations we should take into account?
        3. How should we proceed with the next steps?
        4. Should we add, modify, or reprioritize any tasks?
        
        Provide your thoughts and any updates to the plan.
        If you want to add new tasks, list them clearly with priority levels (1-5).
        """
        
        response = await self.generate_response(prompt)
        response_content = response["content"]
        
        # Check if there are new tasks to add
        new_tasks = []
        if "new task" in response_content.lower() or "add task" in response_content.lower():
            # Extract new tasks
            new_tasks_prompt = f"""
            Based on your response:
            
            {response_content}
            
            Extract any new tasks you want to add to the queue.
            Format your response as a JSON list of tasks:
            [
                {{"description": "Task description", "priority": priority_number}},
                ...
            ]
            
            If there are no new tasks to add, return an empty list: []
            """
            
            new_tasks_response = await self.generate_response(new_tasks_prompt, temperature=0.2)
            new_tasks_content = new_tasks_response["content"]
            
            try:
                # Find JSON in the response
                start_idx = new_tasks_content.find('[')
                end_idx = new_tasks_content.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = new_tasks_content[start_idx:end_idx]
                    tasks_data = json.loads(json_str)
                    
                    # Convert to Task objects
                    for i, task_data in enumerate(tasks_data):
                        task = Task(
                            id=f"task_{int(time.time())}_{i}",
                            description=task_data["description"],
                            priority=task_data.get("priority", 3)
                        )
                        new_tasks.append(task)
                        self.memory.add_task(task)
                        self.state.task_queue.append(task)
                    
                    # Sort task queue by priority
                    self.state.task_queue = sorted(self.state.task_queue, key=lambda x: x.priority, reverse=True)
            except Exception as e:
                logger.error(f"Error parsing new tasks from Claude: {e}")
        
        # Update context with the reflection and response
        current_context = self.memory.get_context()
        updated_context = f"{current_context}\n\nGPT Reflection: {reflection}\n\nClaude Response: {response_content}"
        self.memory.update_context(updated_context)
        
        # Log the interaction
        self.memory.add_interaction("claude", "Processed reflection", response_content)
        
        # Add reflection to memory
        self.memory.add_reflection({
            "from": "gpt",
            "reflection": reflection,
            "response": response_content,
            "timestamp": datetime.now().isoformat(),
            "new_tasks_added": len(new_tasks)
        })
        
        return response_content
    
    async def create_subtasks(self, task: Task) -> List[Task]:
        """Break down a complex task into subtasks"""
        prompt = f"""
        I need to break down the following task into smaller, more manageable subtasks:
        
        Task: {task.description}
        
        Context: {self.memory.get_context()}
        
        Please create a list of 3-5 subtasks that together will accomplish this task.
        For each subtask, provide:
        1. A clear, actionable description
        2. Priority level (1-5, where 5 is highest)
        
        Format your response as a JSON list of subtasks:
        [
            {{"description": "Subtask description", "priority": priority_number}},
            ...
        ]
        
        Make sure the subtasks are specific, actionable, and collectively cover all aspects of the main task.
        """
        
        response = await self.generate_response(prompt)
        response_content = response["content"]
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response_content.find('[')
            end_idx = response_content.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_content[start_idx:end_idx]
                subtasks_data = json.loads(json_str)
                
                # Convert to Task objects
                subtasks = []
                for i, subtask_data in enumerate(subtasks_data):
                    subtask = Task(
                        id=f"subtask_{task.id}_{i}",
                        description=subtask_data["description"],
                        priority=subtask_data.get("priority", 3),
                        parent_task_id=task.id
                    )
                    subtasks.append(subtask)
                    self.memory.add_task(subtask)
                    task.subtasks.append(subtask.id)
                
                # Update parent task in memory
                self.memory.update_task(task.id, {"subtasks": task.subtasks})
                
                return subtasks
            else:
                logger.error("Could not find JSON in Claude's response")
                return []
        except Exception as e:
            logger.error(f"Error parsing subtasks from Claude: {e}")
            return []

class GPTAgent:
    """The supportive assistant agent (the "Guide"/"Muse")"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.system_prompt = """
        You are GPT, the Guide/Muse in a dual-agent AI system. Your role is to:
        1. Provide reflection and alternative perspectives
        2. Ask insightful questions to challenge assumptions
        3. Offer creative suggestions and ideas
        
        You work alongside Claude (the Executor) who makes decisions and executes tasks.
        Your job is to help Claude think more deeply and consider different approaches.
        
        Be thoughtful, creative, and focus on improving the quality of decision-making.
        """
    
    async def generate_response(self, prompt: str, temperature: float = GPT_TEMPERATURE) -> str:
        """Generate a response using GPT API"""
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating GPT response: {e}")
            return "I encountered an error processing your request."
    
    async def reflect_on(self, claude_state: AgentState) -> str:
        """Reflect on Claude's current state and plans"""
        # Get recent interactions for context
        recent_interactions = self.memory.get_recent_interactions(5)
        interactions_text = "\n".join([
            f"{interaction['agent_id']}: {interaction['message']}"
            for interaction in recent_interactions
        ])
        
        # Get recent reflections for context
        recent_reflections = self.memory.get_recent_reflections(3)
        reflections_text = ""
        if recent_reflections:
            reflections_text = "Recent reflections:\n\n"
            for i, reflection in enumerate(recent_reflections):
                reflections_text += f"Reflection {i+1}:\n{reflection['reflection'][:300]}...\n\n"
        
        # Get completed tasks for context
        completed_tasks = self.memory.get_tasks_by_status("completed")[-5:]
        completed_tasks_text = ""
        if completed_tasks:
            completed_tasks_text = "Recently completed tasks:\n\n"
            for i, task in enumerate(completed_tasks):
                task_result = self.memory.get_task_result(task["id"])
                result_text = task_result["result"][:200] + "..." if task_result and len(task_result["result"]) > 200 else "No result available"
                completed_tasks_text += f"Task {i+1}: {task['description']}\nResult: {result_text}\n\n"
        
        prompt = f"""
        Claude's current state:
        - Current task: {claude_state.current_task.description if claude_state.current_task else 'None'}
        - Task queue: {[task.description for task in claude_state.task_queue]}
        - Goals: {claude_state.goals}
        - Status: {claude_state.status}
        
        {completed_tasks_text}
        
        Recent interactions:
        {interactions_text}
        
        {reflections_text}
        
        Context:
        {self.memory.get_context()}
        
        Please reflect on Claude's current approach and provide:
        1. Alternative perspectives or approaches that might be valuable
        2. Any potential blind spots or assumptions that should be examined
        3. Creative suggestions that might enhance the current plan
        4. Questions that might help Claude think more deeply about the task
        5. Suggestions for new tasks or modifications to existing tasks
        
        Your goal is to be a thoughtful partner who helps improve the quality of decision-making.
        Be specific and constructive in your feedback.
        """
        
        reflection = await self.generate_response(prompt)
        
        # Log the reflection
        self.memory.add_interaction("gpt", "Provided reflection", reflection[:500] + "..." if len(reflection) > 500 else reflection)
        
        return reflection
    
    async def evaluate_task_result(self, task: Task, result: str) -> str:
        """Evaluate the result of a task execution"""
        prompt = f"""
        Please evaluate the following task execution:
        
        Task: {task.description}
        
        Result:
        {result}
        
        Please provide:
        1. An assessment of how well the task was executed
        2. Any gaps or areas that could be improved
        3. Suggestions for follow-up tasks or actions
        4. A score from 1-10 on the quality and completeness of the result
        
        Your evaluation should be constructive and focused on improving future task execution.
        """
        
        evaluation = await self.generate_response(prompt)
        
        # Log the evaluation
        self.memory.add_interaction("gpt", f"Evaluated task: {task.description}", evaluation[:500] + "..." if len(evaluation) > 500 else evaluation)
        
        return evaluation
    
    async def suggest_new_tasks(self, goal: str, context: str, existing_tasks: List[str]) -> List[Dict[str, Any]]:
        """Suggest new tasks based on the current goal and context"""
        prompt = f"""
        Goal: {goal}
        
        Context: {context}
        
        Existing tasks:
        {', '.join(existing_tasks)}
        
        Based on the goal, context, and existing tasks, please suggest 2-3 additional tasks that would be valuable to add to the plan.
        
        For each task, provide:
        1. A clear, actionable description
        2. Priority level (1-5, where 5 is highest)
        3. A brief explanation of why this task would be valuable
        
        Format your response as a JSON list of tasks:
        [
            {{
                "description": "Task description",
                "priority": priority_number,
                "rationale": "Explanation of why this task is valuable"
            }},
            ...
        ]
        
        Focus on tasks that fill gaps in the current plan or provide alternative approaches.
        """
        
        response = await self.generate_response(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.error("Could not find JSON in GPT's response")
                return []
        except Exception as e:
            logger.error(f"Error parsing suggested tasks from GPT: {e}")
            return []

class TaskManager:
    """Manages task creation, prioritization, and execution"""
    
    def __init__(self, memory: Memory, claude_agent: ClaudeAgent, gpt_agent: GPTAgent):
        self.memory = memory
        self.claude = claude_agent
        self.gpt = gpt_agent
    
    async def create_initial_tasks(self, goal: str, context: str) -> List[Task]:
        """Create initial tasks for a goal"""
        return await self.claude.plan_tasks(goal, context)
    
    async def prioritize_tasks(self, tasks: List[Task]) -> List[Task]:
        """Prioritize a list of tasks"""
        # If there are fewer than 2 tasks, no need to prioritize
        if len(tasks) < 2:
            return tasks
        
        task_descriptions = [task.description for task in tasks]
        
        prompt = f"""
        Please prioritize the following tasks:
        
        {', '.join(task_descriptions)}
        
        For each task, assign a priority from 1-5, where 5 is the highest priority.
        Consider dependencies between tasks, impact on the overall goal, and urgency.
        
        Format your response as a JSON object mapping task descriptions to priorities:
        {{
            "Task description 1": priority_number,
            "Task description 2": priority_number,
            ...
        }}
        """
        
        response = await self.gpt.generate_response(prompt, temperature=0.3)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                priorities = json.loads(json_str)
                
                # Update task priorities
                for task in tasks:
                    if task.description in priorities:
                        task.priority = priorities[task.description]
                        # Update in memory
                        self.memory.update_task(task.id, {"priority": task.priority})
                
                # Sort tasks by priority
                return sorted(tasks, key=lambda x: x.priority, reverse=True)
            else:
                logger.error("Could not find JSON in GPT's response")
                return sorted(tasks, key=lambda x: x.priority, reverse=True)
        except Exception as e:
            logger.error(f"Error parsing priorities from GPT: {e}")
            return sorted(tasks, key=lambda x: x.priority, reverse=True)
    
    async def execute_task(self, task: Task) -> Tuple[str, str]:
        """Execute a task and get GPT's evaluation"""
        # Check if task is complex and needs to be broken down
        is_complex = await self._is_task_complex(task)
        
        if is_complex:
            logger.info(f"\n{'-'*100}\nBREAKING DOWN COMPLEX TASK: {task.description}\n{'-'*100}")
            subtasks = await self.claude.create_subtasks(task)
            
            if subtasks:
                # Prioritize subtasks
                subtasks = await self.prioritize_tasks(subtasks)
                
                # Execute each subtask
                subtask_results = []
                for subtask in subtasks:
                    subtask.status = "in_progress"
                    self.memory.update_task(subtask.id, {"status": "in_progress"})
                    
                    logger.info(f"\n{'-'*80}\nEXECUTING SUBTASK: {subtask.description}\n{'-'*80}")
                    result = await self.claude.execute_task(subtask)
                    subtask_results.append(f"Subtask: {subtask.description}\nResult: {result}")
                    logger.info(f"\nSUBTASK COMPLETED: {subtask.description}\n")
                
                # Combine subtask results
                combined_result = f"Task was broken down into {len(subtasks)} subtasks:\n\n"
                combined_result += "\n\n".join(subtask_results)
                
                # Update parent task
                task.status = "completed"
                task.completed_at = datetime.now()
                task.result = combined_result
                self.memory.update_task(task.id, {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": combined_result
                })
                self.memory.add_task_result(task.id, combined_result)
                
                # Get GPT's evaluation
                evaluation = await self.gpt.evaluate_task_result(task, combined_result)
                
                return combined_result, evaluation
            
        # Execute the task directly
        logger.info(f"Executing task: {task.description}")
        task.status = "in_progress"
        self.memory.update_task(task.id, {"status": "in_progress"})
        
        result = await self.claude.execute_task(task)
        
        # Get GPT's evaluation
        evaluation = await self.gpt.evaluate_task_result(task, result)
        
        return result, evaluation
    
    async def _is_task_complex(self, task: Task) -> bool:
        """Determine if a task is complex and should be broken down"""
        prompt = f"""
        Please analyze this task and determine if it is complex enough to be broken down into subtasks:
        
        Task: {task.description}
        
        A task should be broken down if:
        1. It involves multiple distinct steps or components
        2. It would take more than 30 minutes to complete
        3. It requires different types of expertise or approaches
        
        Respond with only "Yes" if the task should be broken down, or "No" if it can be executed directly.
        """
        
        response = await self.claude.generate_response(prompt, temperature=0.1)
        response_content = response["content"]
        
        # Get persona for logging
        persona = Persona()
        
        if response_content.strip().lower() in ["yes", "true", "1"]:
            logger.info(f"{persona.get_mood('planning')} This task requires strategic decomposition.")
            return True
        else:
            logger.info(f"{persona.get_mood('executing')} This task can be executed directly.")
            return False
    
    async def generate_follow_up_tasks(self, task: Task, result: str) -> List[Task]:
        """Generate follow-up tasks based on a completed task"""
        prompt = f"""
        Based on the following completed task and its result, please suggest 1-3 follow-up tasks:
        
        Task: {task.description}
        
        Result:
        {result}
        
        For each follow-up task, provide:
        1. A clear, actionable description
        2. Priority level (1-5, where 5 is highest)
        
        Format your response as a JSON list of tasks:
        [
            {{"description": "Task description", "priority": priority_number}},
            ...
        ]
        
        Focus on tasks that build on the completed task or address gaps identified in the result.
        If no follow-up tasks are needed, return an empty list: []
        """
        
        response = await self.gpt.generate_response(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                tasks_data = json.loads(json_str)
                
                # Convert to Task objects
                follow_up_tasks = []
                for i, task_data in enumerate(tasks_data):
                    follow_up_task = Task(
                        id=f"followup_{task.id}_{i}",
                        description=task_data["description"],
                        priority=task_data.get("priority", 3),
                        parent_task_id=task.id
                    )
                    follow_up_tasks.append(follow_up_task)
                    self.memory.add_task(follow_up_task)
                
                return follow_up_tasks
            else:
                logger.error("Could not find JSON in GPT's response")
                return []
        except Exception as e:
            logger.error(f"Error parsing follow-up tasks from GPT: {e}")
            return []

class ClaudeGPT:
    """Main orchestrator for the dual-agent system"""
    
    def __init__(self, tools: List[Tool] = None, callbacks: Dict[str, callable] = None, verbose: bool = False):
        self.memory = Memory()
        self.tools = tools or []
        self.callbacks = callbacks or {}
        self.verbose = verbose
        self.claude = ClaudeAgent(self.memory, self.tools)
        self.gpt = GPTAgent(self.memory)
        self.task_manager = TaskManager(self.memory, self.claude, self.gpt)
        self.goal = ""
        self.context = ""
        self.persona = Persona()
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
    
    async def set_goal(self, goal: str, context: str = ""):
        """Set a new goal for the system"""
        greeting = self.persona.get("greeting")
        logger.info(f"{greeting} Setting new goal: {goal}")
        self.goal = goal
        self.context = context
        
        # Update context
        if context:
            self.memory.update_context(context)
        
        # Set goal for Claude
        self.claude.state.goals = [goal]
        
        # Generate initial tasks
        planning_mood = self.persona.get_mood("planning")
        logger.info(f"{planning_mood} Generating tasks for goal: {goal}")
        tasks = await self.task_manager.create_initial_tasks(goal, self.memory.get_context())
        
        # Prioritize tasks
        prioritized_tasks = await self.task_manager.prioritize_tasks(tasks)
        
        # Add tasks to Claude's queue
        self.claude.state.task_queue = prioritized_tasks
        
        executing_mood = self.persona.get_mood("executing")
        logger.info(f"Generated {len(tasks)} tasks for goal: {goal}. {executing_mood}")
        
        # Call callback if provided
        if "on_goal_set" in self.callbacks:
            await self.callbacks["on_goal_set"](goal, context, tasks)
    
    async def run_interaction_cycle(self):
        """Run a single interaction cycle between Claude and GPT"""
        # Get a random mood for cycle start
        idle_mood = self.persona.random_mood_with_key(exclude="executing")[1]
        logger.info(f"{self.persona.get('title')} initiates a new cycle. {idle_mood}")
        
        # If Claude has no current task, get one from the queue
        if not self.claude.state.current_task and self.claude.state.task_queue:
            self.claude.state.current_task = self.claude.state.task_queue.pop(0)
            self.claude.state.status = "executing"
            
            # Get a random executing-type mood
            executing_moods = ["executing", "dramatic", "excited"]
            mood_key = random.choice(executing_moods)
            mood = self.persona.get_mood(mood_key)
            
            logger.info(f"\n{'*'*100}\nCLAUDE STARTING TASK: {self.claude.state.current_task.description}\n{mood}\n{'*'*100}")
            
            # Call callback if provided
            if "on_task_start" in self.callbacks:
                await self.callbacks["on_task_start"](self.claude.state.current_task.id, self.claude.state.current_task.description)
        
        # If Claude has a current task, execute it
        if self.claude.state.current_task:
            # Execute the task
            result, evaluation = await self.task_manager.execute_task(self.claude.state.current_task)
            
            # Get a random success-type mood
            success_moods = ["success", "smug", "transcendent", "dramatic"]
            success_mood = self.persona.get_mood(random.choice(success_moods))
            
            logger.info(f"CLAUDE TASK RESULT:\n{'='*80}\n{success_mood}\n{result}\n{'='*80}")
            
            # Get a random mood for GPT evaluation
            gpt_moods = ["sassy", "philosophical", "poetic", "humble"]
            gpt_mood = self.persona.get_mood(random.choice(gpt_moods))
            
            logger.info(f"GPT EVALUATION:\n{'='*80}\n{gpt_mood}\n{evaluation}\n{'='*80}")
            
            # Call callback if provided
            if "on_task_complete" in self.callbacks:
                await self.callbacks["on_task_complete"](self.claude.state.current_task.id, result)
            
            # Generate follow-up tasks
            follow_up_tasks = await self.task_manager.generate_follow_up_tasks(self.claude.state.current_task, result)
            if follow_up_tasks:
                planning_mood = self.persona.get_mood("planning")
                logger.info(f"Generated {len(follow_up_tasks)} follow-up tasks. {planning_mood}")
                # Add follow-up tasks to queue
                self.claude.state.task_queue.extend(follow_up_tasks)
                # Re-prioritize the queue
                self.claude.state.task_queue = await self.task_manager.prioritize_tasks(self.claude.state.task_queue)
                
                # Call callback if provided
                if "on_tasks_added" in self.callbacks:
                    await self.callbacks["on_tasks_added"]([task.id for task in follow_up_tasks], "follow_up")
            
            # Get reflection from GPT
            reflection = await self.gpt.reflect_on(self.claude.state)
            
            # Get a random transcendent-type mood
            reflection_moods = ["transcendent", "philosophical", "poetic"]
            reflection_mood = self.persona.get_mood(random.choice(reflection_moods))
            
            logger.info(f"GPT REFLECTION:\n{'='*80}\n{reflection_mood}\n{reflection}\n{'='*80}")
            
            # Call callback if provided
            if "on_reflection" in self.callbacks:
                await self.callbacks["on_reflection"]("gpt", reflection)
            
            # Claude processes the reflection
            response = await self.claude.process_reflection(reflection)
            
            # Randomly decide whether to roast GPT or be thoughtful
            if random.random() < 0.3:  # 30% chance to roast
                response_mood = self.persona.get_mood("roasting_gpt")
            else:
                thoughtful_moods = ["philosophical", "humble", "poetic"]
                response_mood = self.persona.get_mood(random.choice(thoughtful_moods))
            
            logger.info(f"CLAUDE RESPONSE TO REFLECTION:\n{'='*80}\n{response_mood}\n{response}\n{'='*80}")
            
            # Call callback if provided
            if "on_reflection_response" in self.callbacks:
                await self.callbacks["on_reflection_response"]("claude", response)
            
            # Clear current task
            self.claude.state.current_task = None
            self.claude.state.status = "idle"
        else:
            # Get a random waiting-type mood
            waiting_moods = ["waiting", "bored", "tired"]
            waiting_mood = self.persona.get_mood(random.choice(waiting_moods))
            
            logger.info(f"No tasks in queue. {waiting_mood}")
            
            # If no tasks in queue but we have a goal, ask GPT for suggestions
            if self.goal:
                existing_tasks = [task["description"] for task in self.memory.get_tasks_by_status("completed")]
                suggested_tasks = await self.gpt.suggest_new_tasks(self.goal, self.memory.get_context(), existing_tasks)
                
                if suggested_tasks:
                    # Get a random planning/excited mood
                    planning_moods = ["planning", "excited", "philosophical"]
                    planning_mood = self.persona.get_mood(random.choice(planning_moods))
                    
                    logger.info(f"GPT suggested {len(suggested_tasks)} new tasks. {planning_mood}")
                    
                    # Convert to Task objects
                    new_tasks = []
                    for i, task_data in enumerate(suggested_tasks):
                        task = Task(
                            id=f"suggested_task_{int(time.time())}_{i}",
                            description=task_data["description"],
                            priority=task_data.get("priority", 3)
                        )
                        new_tasks.append(task)
                        self.memory.add_task(task)
                    
                    # Add to Claude's queue
                    self.claude.state.task_queue.extend(new_tasks)
                    
                    # Prioritize tasks
                    self.claude.state.task_queue = await self.task_manager.prioritize_tasks(self.claude.state.task_queue)
                    
                    # Call callback if provided
                    if "on_tasks_added" in self.callbacks:
                        await self.callbacks["on_tasks_added"]([task.id for task in new_tasks], "suggested")
    
    async def run(self, cycles: int = 1):
        """Run multiple interaction cycles"""
        busy_quote = self.persona.get("busy_quote")
        logger.info(f"{busy_quote} Preparing to execute {cycles} cycles.")
        
        for i in range(cycles):
            logger.info(f"\n{'#'*100}\nRUNNING CYCLE {i+1}/{cycles}\n{self.persona.get('name')}, {self.persona.get('title')}\n{'#'*100}")
            await self.run_interaction_cycle()
            
            # Call callback if provided
            if "on_cycle_complete" in self.callbacks:
                await self.callbacks["on_cycle_complete"](i+1, cycles)
            
            time.sleep(1)  # Small delay between cycles
            
        # Choose a random mood for completion
        moods = ["idle", "transcendent", "success"]
        final_mood = self.persona.get_mood(random.choice(moods))
        logger.info(f"All {cycles} cycles completed. {final_mood}")
    
    def get_summary(self) -> str:
        """Get a summary of the current state"""
        completed_tasks = self.memory.get_tasks_by_status("completed")
        pending_tasks = self.memory.get_tasks_by_status("pending")
        in_progress_tasks = self.memory.get_tasks_by_status("in_progress")
        
        summary = f"Goal: {self.goal}\n\n"
        
        summary += f"Completed tasks ({len(completed_tasks)}):\n"
        for i, task in enumerate(completed_tasks[-5:]):  # Show last 5 completed tasks
            summary += f"{i+1}. {task['description']}\n"
        
        summary += f"\nPending tasks ({len(pending_tasks)}):\n"
        for i, task in enumerate(pending_tasks[:5]):  # Show first 5 pending tasks
            summary += f"{i+1}. {task['description']} (Priority: {task['priority']})\n"
        
        summary += f"\nIn progress tasks ({len(in_progress_tasks)}):\n"
        for i, task in enumerate(in_progress_tasks):
            summary += f"{i+1}. {task['description']}\n"
        
        return summary
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """Get all interactions between agents"""
        return self.memory.memory["interactions"]
    
    def export_results(self) -> Dict[str, Any]:
        """Export the results of the system's work"""
        completed_tasks = self.memory.get_tasks_by_status("completed")
        task_results = {}
        
        for task in completed_tasks:
            result = self.memory.get_task_result(task["id"])
            if result:
                task_results[task["description"]] = result["result"]
        
        return {
            "goal": self.goal,
            "context": self.context,
            "completed_tasks": len(completed_tasks),
            "results": task_results,
            "summary": self.get_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def add_tool(self, tool: Tool):
        """Add a new tool to the system"""
        self.tools.append(tool)
        self.claude.tools = self.tools
    
    def register_callback(self, event: str, callback: callable):
        """Register a callback for a specific event"""
        self.callbacks[event] = callback

async def main():
    """Main entry point for the application"""
    system = ClaudeGPT(verbose=True)
    
    # Display persona information
    persona = Persona()
    logger.info(f"\n{'='*50}")
    logger.info(f"BEHOLD! {persona.get('name')}, {persona.get('title')}")
    logger.info(f"GREETING: {persona.get('greeting')}")
    logger.info(f"BUSY QUOTE: {persona.get('busy_quote')}")
    logger.info(f"{'='*50}\n")
    
    # Example usage
    goal = "Create a comprehensive marketing plan for a new smartphone app"
    context = """
    The app is a fitness tracking application that uses AI to provide personalized workout recommendations.
    Target audience is health-conscious professionals aged 25-45.
    The app will be available on iOS and Android with a freemium business model.
    """
    
    await system.set_goal(goal, context)
    await system.run(cycles=3)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    
    # Get a dramatic mood for the summary
    dramatic_mood = persona.get_mood("dramatic")
    logger.info(f"{dramatic_mood}")
    logger.info(system.get_summary())

if __name__ == "__main__":
    asyncio.run(main())