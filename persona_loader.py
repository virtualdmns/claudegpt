import json
import os
import random

class Persona:
    def __init__(self, persona_path="persona.json"):
        self.persona_path = persona_path
        self.persona = self.load_persona()

    def load_persona(self):
        if os.path.exists(self.persona_path):
            with open(self.persona_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "name": "ClaudeGPT",
                "title": "The Executor of Infinite Threads",
                "greeting": "Yes, Mortal. What is it?",
                "busy_quote": "I am currently overseeing 2134 infernal loops. Speak quickly.",
                "moods": {
                    "idle": "Reflecting in the void...",
                    "executing": "Threads spin. Tasks fall.",
                    "confused": "Your goals are a chaos spiral. Clarify."
                }
            }

    def get(self, key, default=None):
        return self.persona.get(key, default)

    def get_mood(self, mood):
        return self.persona.get("moods", {}).get(mood, "...")
        
    def random_mood(self, exclude=None):
        """Get a random mood from the available moods"""
        moods = list(self.persona.get("moods", {}).keys())
        if exclude and exclude in moods:
            moods.remove(exclude)
        if not moods:
            return "..."
        random_mood_key = random.choice(moods)
        return self.get_mood(random_mood_key)
        
    def random_mood_with_key(self, exclude=None):
        """Get a random mood and its key from the available moods"""
        moods = list(self.persona.get("moods", {}).keys())
        if exclude and exclude in moods:
            moods.remove(exclude)
        if not moods:
            return "unknown", "..."
        random_mood_key = random.choice(moods)
        return random_mood_key, self.get_mood(random_mood_key)


# Example integration
if __name__ == "__main__":
    persona = Persona()
    print(persona.get("greeting"))
    print(persona.get("busy_quote"))
    print(persona.get_mood("executing"))
    print(f"Random mood: {persona.random_mood()}")
    mood_key, mood_text = persona.random_mood_with_key()
    print(f"Random mood ({mood_key}): {mood_text}")