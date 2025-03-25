"""
Utility functions for the ClaudeGPT system
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
import asyncio

logger = logging.getLogger("ClaudeGPT.utils")

class PerformanceMonitor:
    """Monitor and track performance metrics for the ClaudeGPT system"""
    
    def __init__(self):
        self.metrics = {
            "task_execution_times": [],
            "api_call_times": {
                "claude": [],
                "gpt": []
            },
            "tool_execution_times": {},
            "total_tokens": {
                "claude": {"input": 0, "output": 0},
                "gpt": {"input": 0, "output": 0}
            }
        }
    
    def record_task_time(self, task_id: str, duration: float):
        """Record the execution time of a task"""
        self.metrics["task_execution_times"].append({
            "task_id": task_id,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def record_api_call(self, model: str, duration: float, input_tokens: int, output_tokens: int):
        """Record API call metrics"""
        if model.lower() in ["claude", "anthropic"]:
            key = "claude"
        else:
            key = "gpt"
            
        self.metrics["api_call_times"][key].append({
            "duration": duration,
            "timestamp": time.time()
        })
        
        self.metrics["total_tokens"][key]["input"] += input_tokens
        self.metrics["total_tokens"][key]["output"] += output_tokens
    
    def record_tool_execution(self, tool_name: str, duration: float):
        """Record tool execution time"""
        if tool_name not in self.metrics["tool_execution_times"]:
            self.metrics["tool_execution_times"][tool_name] = []
            
        self.metrics["tool_execution_times"][tool_name].append({
            "duration": duration,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        task_times = [item["duration"] for item in self.metrics["task_execution_times"]]
        
        claude_times = [item["duration"] for item in self.metrics["api_call_times"]["claude"]]
        gpt_times = [item["duration"] for item in self.metrics["api_call_times"]["gpt"]]
        
        tool_avg_times = {}
        for tool_name, times in self.metrics["tool_execution_times"].items():
            if times:
                tool_avg_times[tool_name] = sum(item["duration"] for item in times) / len(times)
            else:
                tool_avg_times[tool_name] = 0
        
        return {
            "avg_task_time": sum(task_times) / len(task_times) if task_times else 0,
            "total_tasks": len(self.metrics["task_execution_times"]),
            "avg_claude_call_time": sum(claude_times) / len(claude_times) if claude_times else 0,
            "avg_gpt_call_time": sum(gpt_times) / len(gpt_times) if gpt_times else 0,
            "total_claude_calls": len(self.metrics["api_call_times"]["claude"]),
            "total_gpt_calls": len(self.metrics["api_call_times"]["gpt"]),
            "total_tokens": self.metrics["total_tokens"],
            "tool_avg_times": tool_avg_times
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Performance metrics exported to {file_path}")

class DebugLogger:
    """Enhanced logging for debugging the ClaudeGPT system"""
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.DEBUG):
        self.logger = logging.getLogger("ClaudeGPT.debug")
        self.logger.setLevel(level)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_api_request(self, api: str, prompt: str, params: Dict[str, Any]):
        """Log API request details"""
        self.logger.debug(f"API Request to {api}:")
        self.logger.debug(f"Parameters: {json.dumps(params, indent=2)}")
        self.logger.debug(f"Prompt: {prompt[:200]}...")  # Log first 200 chars
    
    def log_api_response(self, api: str, response: Any):
        """Log API response details"""
        self.logger.debug(f"API Response from {api}:")
        if hasattr(response, 'model_dump_json'):
            # For Pydantic models
            self.logger.debug(response.model_dump_json())
        elif isinstance(response, dict):
            self.logger.debug(json.dumps(response, indent=2))
        else:
            self.logger.debug(str(response))
    
    def log_task_execution(self, task_id: str, description: str, status: str):
        """Log task execution details"""
        self.logger.info(f"Task {task_id} ({status}): {description}")
    
    def log_tool_execution(self, tool_name: str, params: Dict[str, Any], result: Any):
        """Log tool execution details"""
        self.logger.debug(f"Tool Execution: {tool_name}")
        self.logger.debug(f"Parameters: {json.dumps(params, indent=2)}")
        self.logger.debug(f"Result: {json.dumps(result, indent=2) if isinstance(result, dict) else str(result)}")
    
    def log_memory_update(self, key: str, value: Any):
        """Log memory updates"""
        self.logger.debug(f"Memory Update - Key: {key}")
        if isinstance(value, dict):
            self.logger.debug(f"Value: {json.dumps(value, indent=2)}")
        else:
            self.logger.debug(f"Value: {str(value)[:200]}...")  # Log first 200 chars

async def rate_limit(calls_per_minute: int = 60):
    """Rate limiting utility to prevent API throttling"""
    delay = 60 / calls_per_minute
    await asyncio.sleep(delay)

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to a maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"