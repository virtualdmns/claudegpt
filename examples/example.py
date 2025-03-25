"""
Example usage of the ClaudeGPT system
"""

import asyncio
import logging
from claudeGPT import ClaudeGPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClaudeGPT.example")

async def run_example():
    """Run an example with ClaudeGPT"""
    logger.info("Initializing ClaudeGPT system")
    system = ClaudeGPT()
    
    # Define a goal and context
    goal = "Design a sustainable urban garden for a community center"
    context = """
    The community center is located in an urban neighborhood with limited green space.
    The garden should be accessible to people of all ages and abilities.
    The budget is $5,000 for initial setup.
    The garden should include edible plants, flowers, and spaces for community gatherings.
    Sustainability and low maintenance are key priorities.
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
    
    logger.info("Example completed")

if __name__ == "__main__":
    asyncio.run(run_example())