"""
Test script for the ClaudeGPT system
"""

import asyncio
import unittest
import logging
from unittest.mock import patch, MagicMock
from claudeGPT import ClaudeGPT, Task, Memory, Tool

# Configure logging
logging.basicConfig(level=logging.ERROR)

class MockTool(Tool):
    """Mock tool for testing"""
    
    def __init__(self):
        super().__init__(
            name="mock_tool",
            description="A mock tool for testing",
            parameters={
                "param1": {
                    "type": "string",
                    "description": "Test parameter"
                }
            }
        )
    
    async def execute(self, **kwargs):
        return {"result": f"Executed with {kwargs.get('param1', 'default')}"}

class TestClaudeGPT(unittest.TestCase):
    """Test cases for ClaudeGPT system"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_claude_response = {
            "content": [{"type": "text", "text": "This is a mock Claude response"}],
            "id": "mock-id",
            "model": "claude-3-opus-20240229",
            "role": "assistant",
            "type": "message"
        }
        
        self.mock_gpt_response = {
            "choices": [{
                "message": {
                    "content": "This is a mock GPT response",
                    "role": "assistant"
                }
            }]
        }
    
    @patch('anthropic.Anthropic')
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai, mock_anthropic):
        """Test system initialization"""
        system = ClaudeGPT()
        self.assertIsNotNone(system)
        self.assertIsInstance(system.memory, Memory)
        self.assertEqual(len(system.tools), 0)
    
    @patch('anthropic.Anthropic')
    @patch('openai.OpenAI')
    async def test_set_goal(self, mock_openai, mock_anthropic):
        """Test setting a goal"""
        # Setup mocks
        mock_anthropic_instance = mock_anthropic.return_value
        mock_anthropic_instance.messages.create.return_value = self.mock_claude_response
        
        # Create system and set goal
        system = ClaudeGPT()
        await system.set_goal("Test goal", "Test context")
        
        # Verify
        self.assertEqual(system.goal, "Test goal")
        self.assertEqual(system.context, "Test context")
        self.assertTrue(mock_anthropic_instance.messages.create.called)
    
    @patch('anthropic.Anthropic')
    @patch('openai.OpenAI')
    async def test_task_execution(self, mock_openai, mock_anthropic):
        """Test task execution"""
        # Setup mocks
        mock_anthropic_instance = mock_anthropic.return_value
        mock_anthropic_instance.messages.create.return_value = self.mock_claude_response
        
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.chat.completions.create.return_value = self.mock_gpt_response
        
        # Create system with mock tool
        system = ClaudeGPT(tools=[MockTool()])
        await system.set_goal("Test goal", "Test context")
        
        # Create and execute a task
        task = Task(
            id="task-1",
            description="Test task",
            priority=5,
            status="pending"
        )
        
        result = await system._execute_task(task)
        
        # Verify
        self.assertIsNotNone(result)
        self.assertTrue(mock_anthropic_instance.messages.create.called)
    
    @patch('anthropic.Anthropic')
    @patch('openai.OpenAI')
    async def test_tool_execution(self, mock_openai, mock_anthropic):
        """Test tool execution"""
        # Create tool and system
        tool = MockTool()
        system = ClaudeGPT(tools=[tool])
        
        # Execute tool
        result = await system._execute_tool("mock_tool", {"param1": "test_value"})
        
        # Verify
        self.assertEqual(result["result"], "Executed with test_value")
    
    @patch('anthropic.Anthropic')
    @patch('openai.OpenAI')
    async def test_memory_operations(self, mock_openai, mock_anthropic):
        """Test memory operations"""
        system = ClaudeGPT()
        
        # Add to memory
        system.memory.add("test_key", "test_value")
        
        # Retrieve from memory
        value = system.memory.get("test_key")
        
        # Verify
        self.assertEqual(value, "test_value")
        
        # Test memory context generation
        context = system.memory.get_context()
        self.assertIn("test_key", context)
        self.assertIn("test_value", context)

async def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestClaudeGPT)
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    asyncio.run(run_tests())