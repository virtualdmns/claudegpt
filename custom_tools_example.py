"""
Example of creating and using custom tools with ClaudeGPT
"""

import asyncio
import logging
import json
from datetime import datetime
from claudeGPT import ClaudeGPT, Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClaudeGPT.custom_tools")

# Define a simple calculator tool
class CalculatorTool(Tool):
    """Tool to perform basic calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform basic mathematical calculations",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Mathematical operation to perform (add, subtract, multiply, divide)"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            }
        )
    
    async def execute(self, **kwargs):
        """Execute the calculation"""
        operation = kwargs.get("operation", "").lower()
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        
        result = None
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero"}
            result = a / b
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }

# Define a text analysis tool
class TextAnalysisTool(Tool):
    """Tool to analyze text"""
    
    def __init__(self):
        super().__init__(
            name="analyze_text",
            description="Analyze text for word count, sentiment, and key phrases",
            parameters={
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                },
                "include_sentiment": {
                    "type": "boolean",
                    "description": "Whether to include sentiment analysis",
                    "default": True
                }
            }
        )
    
    async def execute(self, **kwargs):
        """Analyze the provided text"""
        text = kwargs.get("text", "")
        include_sentiment = kwargs.get("include_sentiment", True)
        
        # Simple word count
        words = text.split()
        word_count = len(words)
        
        # Simple sentence count
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Very basic sentiment analysis
        sentiment = None
        if include_sentiment:
            positive_words = ["good", "great", "excellent", "positive", "happy", "best", "love", "wonderful"]
            negative_words = ["bad", "terrible", "negative", "sad", "worst", "hate", "awful", "poor"]
            
            positive_count = sum(1 for word in words if word.lower() in positive_words)
            negative_count = sum(1 for word in words if word.lower() in negative_words)
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        # Simple key phrase extraction (just most common words)
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?;:()"\'')
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 words
        key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "key_phrases": [{"word": word, "count": count} for word, count in key_phrases]
        }
        
        if include_sentiment:
            result["sentiment"] = sentiment
        
        return result

async def run_custom_tools_example():
    """Run an example with custom tools"""
    logger.info("Initializing ClaudeGPT system with custom tools")
    
    # Initialize system with custom tools
    system = ClaudeGPT(
        tools=[CalculatorTool(), TextAnalysisTool()]
    )
    
    # Define a goal that will likely use the tools
    goal = "Create a financial literacy guide for high school students"
    context = """
    The guide should cover basic concepts like budgeting, saving, investing, and understanding credit.
    It should include practical examples and calculations to illustrate financial concepts.
    The language should be accessible and engaging for teenagers.
    The guide will be distributed digitally and as a printed booklet.
    """
    
    # Set the goal and run the system
    logger.info(f"Setting goal: {goal}")
    await system.set_goal(goal, context)
    
    # Run 5 interaction cycles
    logger.info("Running 5 interaction cycles")
    await system.run(cycles=5)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(system.get_summary())
    
    # Export the results
    results = system.export_results()
    with open("financial_literacy_guide.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Custom tools example completed. Results exported to financial_literacy_guide.json")

if __name__ == "__main__":
    asyncio.run(run_custom_tools_example())