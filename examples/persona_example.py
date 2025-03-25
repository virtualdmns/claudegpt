"""
ClaudeGPT Persona Example

This example demonstrates the persona system in ClaudeGPT.
"""

"""
ClaudeGPT Persona Example

This example demonstrates the persona system in ClaudeGPT.
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from claudeGPT import ClaudeGPT
from persona_loader import Persona

async def main():
    """Main entry point for the application"""
    # Initialize the system
    system = ClaudeGPT(verbose=True)
    
    # Print the persona information
    persona = Persona()
    print(f"\n{'='*50}")
    print(f"PERSONA: {persona.get('name')}, {persona.get('title')}")
    print(f"GREETING: {persona.get('greeting')}")
    print(f"BUSY QUOTE: {persona.get('busy_quote')}")
    print(f"MOODS:")
    for mood_name, mood_text in persona.persona.get("moods", {}).items():
        print(f"  - {mood_name}: {mood_text}")
    print(f"{'='*50}\n")
    
    # Example usage
    goal = "Create a comprehensive marketing plan for a new smartphone app"
    context = """
    The app is a fitness tracking application that uses AI to provide personalized workout recommendations.
    Target audience is health-conscious professionals aged 25-45.
    The app will be available on iOS and Android with a freemium business model.
    """
    
    await system.set_goal(goal, context)
    await system.run(cycles=2)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(system.get_summary())

if __name__ == "__main__":
    asyncio.run(main())