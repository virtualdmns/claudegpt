"""
Simple example usage of the ClaudeGPT system
"""

import asyncio
import logging
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import ClaudeGPT
from claudeGPT import ClaudeGPT, ColoredFormatter

# Configure logging with colors and no truncation
logger = logging.getLogger("ClaudeGPT.simple_example")
# The root logger configuration from claudeGPT.py will be inherited

async def run_simple_example():
    """Run a simple example with ClaudeGPT"""
    logger.info("Initializing ClaudeGPT system")
    
    # Initialize system
    system = ClaudeGPT()
    
    # Define a simple goal and context
    goal = "Create a content marketing strategy for a new online bookstore"
    context = """
    The bookstore specializes in rare and collectible books, as well as new releases.
    Target audience includes book collectors, avid readers, and gift shoppers.
    The store has a small physical location but wants to grow its online presence.
    Budget for marketing is limited, so organic and cost-effective strategies are preferred.
    """
    
    # Set the goal and run the system
    logger.info(f"Setting goal: {goal}")
    await system.set_goal(goal, context)
    
    # Run 3 interaction cycles
    logger.info("Running 3 interaction cycles")
    await system.run(cycles=3)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(system.get_summary())
    
    logger.info("Simple example completed")

if __name__ == "__main__":
    asyncio.run(run_simple_example())