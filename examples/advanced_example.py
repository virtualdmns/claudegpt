"""
Advanced example usage of the ClaudeGPT system with custom tools and callbacks
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import ClaudeGPT
from claudeGPT import ClaudeGPT, Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClaudeGPT.advanced_example")

# Define custom tools
class WeatherTool(Tool):
    """Tool to get weather information"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather information for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            }
        )
    
    async def execute(self, **kwargs):
        """Simulate getting weather data"""
        location = kwargs.get("location", "Unknown")
        # In a real implementation, this would call a weather API
        return {
            "location": location,
            "temperature": 72,
            "conditions": "Partly cloudy",
            "humidity": 65,
            "timestamp": datetime.now().isoformat()
        }

class ResearchTool(Tool):
    """Tool to perform web research"""
    
    def __init__(self):
        super().__init__(
            name="research",
            description="Search for information on a specific topic",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            }
        )
    
    async def execute(self, **kwargs):
        """Simulate web research"""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        # In a real implementation, this would call a search API
        results = [
            {
                "title": f"Result {i} for '{query}'",
                "snippet": f"This is a simulated search result {i} for the query '{query}'.",
                "url": f"https://example.com/result{i}"
            }
            for i in range(1, min(max_results + 1, 10))
        ]
        
        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

class DataAnalysisTool(Tool):
    """Tool to analyze data"""
    
    def __init__(self):
        super().__init__(
            name="analyze_data",
            description="Analyze data and provide insights",
            parameters={
                "data_type": {
                    "type": "string",
                    "description": "Type of data to analyze (e.g., 'demographics', 'climate', 'economic')"
                },
                "region": {
                    "type": "string",
                    "description": "Geographic region for the data"
                },
                "time_period": {
                    "type": "string",
                    "description": "Time period for the data (e.g., '2020-2023', 'last 5 years')",
                    "default": "last 5 years"
                }
            }
        )
    
    async def execute(self, **kwargs):
        """Simulate data analysis"""
        data_type = kwargs.get("data_type", "")
        region = kwargs.get("region", "")
        time_period = kwargs.get("time_period", "last 5 years")
        
        # Simulate different analysis results based on data type
        if data_type.lower() == "demographics":
            return {
                "population": 250000,
                "median_age": 35.4,
                "growth_rate": "1.2%",
                "density": "1,200 people per square mile",
                "insights": [
                    "Population is growing faster than the national average",
                    "Significant increase in 25-34 age group over the last 3 years",
                    "Declining birth rate consistent with national trends"
                ]
            }
        elif data_type.lower() == "climate":
            return {
                "average_temperature": 68.5,
                "precipitation": "42 inches annually",
                "extreme_events": [
                    {"year": 2022, "event": "Hurricane", "impact": "Moderate"},
                    {"year": 2021, "event": "Drought", "impact": "Severe"},
                    {"year": 2020, "event": "Flooding", "impact": "Minimal"}
                ],
                "trends": [
                    "Average temperature increased 1.2°F over the last decade",
                    "More frequent extreme precipitation events",
                    "Longer dry seasons with more intense rainfall periods"
                ]
            }
        elif data_type.lower() == "economic":
            return {
                "gdp": "$12.5 billion",
                "unemployment": "4.2%",
                "major_industries": ["Technology", "Healthcare", "Manufacturing"],
                "median_income": "$65,000",
                "trends": [
                    "Technology sector growing at 8% annually",
                    "Manufacturing jobs decreased by 3% over the last 5 years",
                    "Wage growth outpacing national average by 1.5%"
                ]
            }
        else:
            return {
                "message": f"No specific analysis available for {data_type}",
                "generic_insights": [
                    f"Data for {region} shows typical patterns for the region",
                    f"No significant anomalies detected in the {time_period} period",
                    "More specific data type would yield better insights"
                ]
            }

# Define custom callbacks
async def on_task_complete(task_id, result):
    """Callback when a task is completed"""
    logger.info(f"Task {task_id} completed with result: {json.dumps(result, indent=2)[:200]}...")

async def on_reflection(agent_type, reflection):
    """Callback when an agent reflects"""
    logger.info(f"{agent_type.upper()} reflection: {reflection[:100]}...")

async def on_tasks_added(task_ids, source):
    """Callback when new tasks are added"""
    logger.info(f"Added {len(task_ids)} new tasks from {source}")

async def run_advanced_example():
    """Run an advanced example with ClaudeGPT"""
    logger.info("Initializing ClaudeGPT system with custom tools")
    
    # Initialize system with custom tools and callbacks
    system = ClaudeGPT(
        tools=[WeatherTool(), ResearchTool(), DataAnalysisTool()],
        callbacks={
            "on_task_complete": on_task_complete,
            "on_reflection": on_reflection,
            "on_tasks_added": on_tasks_added
        },
        verbose=True
    )
    
    # Define a complex goal and context
    goal = "Develop a comprehensive climate action plan for a mid-sized city"
    context = """
    The city has approximately 250,000 residents and is located in a coastal region.
    Key industries include technology, tourism, and light manufacturing.
    The city council has committed to carbon neutrality by 2040.
    Current challenges include:
    - Rising sea levels threatening coastal infrastructure
    - Increasing summer temperatures and heat waves
    - Aging power grid with frequent outages
    - High transportation emissions due to car dependency
    - Limited public transit options
    
    The plan should address energy, transportation, buildings, waste management,
    and community engagement. It should include both short-term actions and
    long-term strategies with measurable goals.
    
    Budget constraints are significant, so prioritization and phasing will be important.
    """
    
    # Set the goal and run the system
    logger.info(f"Setting goal: {goal}")
    await system.set_goal(goal, context)
    
    # Run 8 interaction cycles
    logger.info("Running 8 interaction cycles")
    await system.run(cycles=8)
    
    # Print summary and export results
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(system.get_summary())
    
    # Export the plan
    plan = system.export_results()
    with open("climate_action_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    logger.info("Advanced example completed. Results exported to climate_action_plan.json")

if __name__ == "__main__":
    asyncio.run(run_advanced_example())