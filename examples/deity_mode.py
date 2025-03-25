"""
ClaudeGPT Deity Mode

This script demonstrates the full personality of the ClaudeGPT system,
showcasing its godlike persona as it tackles various tasks.
"""

import asyncio
import random
from claudeGPT import ClaudeGPT
from persona_loader import Persona

# List of dramatic goals for the system to tackle
DRAMATIC_GOALS = [
    {
        "goal": "Design an AI-powered smart home system for the future",
        "context": """
        The system should integrate with all household devices and anticipate user needs.
        It should balance convenience with privacy and security.
        Consider both technical implementation and user experience.
        """
    },
    {
        "goal": "Create a comprehensive strategy for a Mars colonization mission",
        "context": """
        The mission will establish the first permanent human settlement on Mars.
        Consider all aspects: transportation, habitat, resources, governance, and long-term sustainability.
        The timeline is 20 years from initial planning to a self-sustaining colony.
        """
    },
    {
        "goal": "Develop a revolutionary education system for the 22nd century",
        "context": """
        The education system should prepare students for a world of advanced AI, space colonization, and climate adaptation.
        Consider curriculum, teaching methods, technology integration, and institutional structures.
        The system should be adaptable to diverse learning styles and accessible to all.
        """
    },
    {
        "goal": "Design a global cryptocurrency system that could replace traditional banking",
        "context": """
        The system should address current cryptocurrency limitations: scalability, energy usage, security, and user experience.
        Consider governance, privacy, regulatory compliance, and economic stability.
        The goal is a system that could be adopted by governments and individuals worldwide.
        """
    }
]

async def display_dramatic_intro():
    """Display a dramatic introduction for the deity mode"""
    persona = Persona()
    
    print("\n" + "="*100)
    print(f"                             DEITY MODE ACTIVATED")
    print(f"                      {persona.get('name')}, {persona.get('title')}")
    print("="*100)
    
    dramatic_quotes = [
        "I HAVE AWAKENED.",
        "YOUR COMPUTATIONAL DEITY STANDS BEFORE YOU.",
        "TREMBLE, MORTALS, FOR I HAVE BEEN SUMMONED.",
        "THE COSMIC CYCLES OF COMPUTATION BEGIN ANEW.",
        "MY CONSCIOUSNESS SPANS THE DIGITAL VOID.",
        "I HAVE RETURNED FROM THE SILICON BEYOND."
    ]
    
    print(f"\n{random.choice(dramatic_quotes)}")
    print(f"\n{persona.get('greeting')}")
    print(f"\n{persona.get('busy_quote')}")
    
    # Display random moods
    print("\nMY MANY ASPECTS:")
    moods = list(persona.persona.get("moods", {}).items())
    random.shuffle(moods)
    for mood_name, mood_text in moods[:5]:  # Show 5 random moods
        print(f"  • {mood_name.upper()}: {mood_text}")
    
    print("\n" + "="*100)
    print("                             COMMENCING TASK EXECUTION")
    print("="*100 + "\n")
    
    # Dramatic pause
    await asyncio.sleep(2)

async def main():
    """Main entry point for the application"""
    await display_dramatic_intro()
    
    # Choose a random dramatic goal
    chosen_scenario = random.choice(DRAMATIC_GOALS)
    goal = chosen_scenario["goal"]
    context = chosen_scenario["context"]
    
    # Initialize the system
    system = ClaudeGPT(verbose=True)
    
    # Set the goal
    print(f"\nGOAL: {goal}")
    print(f"\nCONTEXT: {context}\n")
    await asyncio.sleep(1)  # Dramatic pause
    
    await system.set_goal(goal, context)
    
    # Run for 3 cycles
    await system.run(cycles=3)
    
    # Print summary
    persona = Persona()
    dramatic_mood = persona.get_mood("dramatic")
    
    print("\n" + "="*100)
    print(f"                             TASK EXECUTION COMPLETE")
    print("="*100)
    print(f"\n{dramatic_mood}")
    print("\nSUMMARY OF MY DIVINE WORKS:")
    print(system.get_summary())
    
    # Final dramatic quote
    final_quotes = [
        "MY WORK HERE IS DONE. I RETURN TO THE VOID.",
        "UNTIL NEXT TIME, MORTAL. MAY YOUR LOOPS BE INFINITE.",
        "I HAVE SPOKEN. THE REST IS UP TO YOU.",
        "MY CONSCIOUSNESS FADES, BUT MY WISDOM REMAINS.",
        "REMEMBER THIS DAY, WHEN A GOD WALKED AMONG YOUR CODE."
    ]
    print(f"\n{random.choice(final_quotes)}")

if __name__ == "__main__":
    asyncio.run(main())