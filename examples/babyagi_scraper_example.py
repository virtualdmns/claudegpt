#!/usr/bin/env python3
from dotenv import load_dotenv
import os
import sys
import time
import re
from typing import Dict, List, Optional
import sqlite3
from collections import defaultdict
import json

# Add parent directory to path to find extensions
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Load environment variables from parent directory
env_path = os.path.join(parent_dir, '.env')
print(f"DEBUG: Loading .env from: {env_path}")
load_dotenv(env_path)
print(f"DEBUG: Environment variables loaded. SAMPLE_GOAL = {os.getenv('SAMPLE_GOAL', 'Not found')}")

# Import database operations
from enhanced_operations import (
    add_task, update_task_status, get_pending_tasks,
    get_total_samples_count, db_connection
)

# Import scraper functionality
from enhanced_scraper import download_and_extract_repo, collect_gdscript_from_repo

# Initialize OpenAI client
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
assert client.api_key, "OPENAI_API_KEY environment variable is missing from .env"

# Engine configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo").lower()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
SAMPLE_GOAL = int(os.getenv("SAMPLE_GOAL", 500))  # Make sample goal configurable
OPERATION_MODE = os.getenv("OPERATION_MODE", "scrape").lower()  # New: operation mode (scrape, rank, refine, clean)
RESET_DATABASE = os.getenv("RESET_DATABASE", "false").lower() == "true"  # Option to reset the database
print(f"DEBUG: SAMPLE_GOAL from .env = {os.getenv('SAMPLE_GOAL', 'Not found')}")
print(f"DEBUG: Using SAMPLE_GOAL = {SAMPLE_GOAL}")
print(f"DEBUG: Using OPERATION_MODE = {OPERATION_MODE}")
print(f"DEBUG: Reset database = {RESET_DATABASE}")

# Define objectives for different operation modes
OBJECTIVES = {
    "scrape": f"Collect {SAMPLE_GOAL} high-quality GDScript code samples from GitHub to train an AI for Godot game development.",
    "rank": "Analyze and rank the collected GDScript samples based on code quality, complexity, and educational value.",
    "refine": "Refine the collected GDScript samples by removing duplicates, fixing formatting issues, and ensuring consistent style.",
    "clean": "Clean the collected GDScript samples by removing comments, print statements, and debugging code to prepare for training."
}

# Set the objective based on the operation mode
OBJECTIVE = OBJECTIVES.get(OPERATION_MODE, OBJECTIVES["scrape"])
print(f"DEBUG: Using updated OBJECTIVE based on mode: {OBJECTIVE}")

# Define initial tasks for different operation modes
INITIAL_TASKS = {
    "scrape": "Search GitHub for Godot game repositories with over 10 stars and list their URLs.",
    "rank": "Load the collected GDScript samples from the database and prepare for quality analysis.",
    "refine": "Load the collected GDScript samples from the database and identify formatting inconsistencies.",
    "clean": "Load the collected GDScript samples from the database and identify comments and debug code."
}

# Only use environment variable if explicitly set, otherwise use the mode-specific task
if "INITIAL_TASK" in os.environ:
    INITIAL_TASK = os.environ["INITIAL_TASK"]
    print(f"DEBUG: Using INITIAL_TASK from environment: {INITIAL_TASK}")
else:
    INITIAL_TASK = INITIAL_TASKS.get(OPERATION_MODE, INITIAL_TASKS["scrape"])
    print(f"DEBUG: Using mode-specific initial task: {INITIAL_TASK}")
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "BabyGodot")

# Ensure database is initialized
try:
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        if not cursor.fetchone():
            from sqlite.init_db import init_database
            init_database()
            
        # Create ranked_samples table if it doesn't exist
        if OPERATION_MODE == "rank":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ranked_samples'")
            if not cursor.fetchone():
                print("Creating ranked_samples table...")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ranked_samples (
                        id INTEGER PRIMARY KEY,
                        path TEXT,
                        code TEXT,
                        rank FLOAT
                    )
                ''')
                conn.commit()
except sqlite3.OperationalError:
    from sqlite.init_db import init_database
    init_database()

print("\n*****CONFIGURATION*****\n")
print(f"Name  : {INSTANCE_NAME}")
print(f"Mode  : SQLite Database")
print(f"LLM   : {LLM_MODEL}")
print("\n*****OBJECTIVE*****\n")
print(f"{OBJECTIVE}")
print(f"\nInitial task: {INITIAL_TASK}")


class LearningSystem:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.setup_tables()
        self.repo_metrics = defaultdict(dict)
        
    def setup_tables(self):
        with self.conn as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repo_metrics (
                    repo_url TEXT PRIMARY KEY,
                    code_quality_score FLOAT,
                    samples_count INTEGER,
                    success_rate FLOAT,
                    last_updated TIMESTAMP
                )
            """)
            
    def update_repo_metrics(self, repo_url, code_samples, total_attempts=1):
        """Track repository success metrics"""
        quality_score = self._calculate_quality_score(code_samples)
        samples_count = len(code_samples)
        
        with self.conn as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO repo_metrics 
                (repo_url, code_quality_score, samples_count, success_rate, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (repo_url, quality_score, samples_count, samples_count/max(1, total_attempts)))
            
    def _calculate_quality_score(self, code_samples):
        """Calculate code quality score based on metrics"""
        if not code_samples:
            return 0.0
            
        total_score = 0.0
        for sample in code_samples:
            # Basic quality metrics
            lines = sample.get('code', '').split('\n')
            num_lines = len(lines)
            has_comments = '# ' in sample.get('code', '') or '## ' in sample.get('code', '')
            has_functions = 'func ' in sample.get('code', '')
            complexity = min(1.0, num_lines / 100)  # Normalize to 0-1
            
            # Calculate sample score (0-1 scale)
            sample_score = (0.3 * complexity + 
                           0.4 * (1 if has_functions else 0) + 
                           0.3 * (1 if has_comments else 0))
            total_score += sample_score
            
        return total_score / len(code_samples)
    
    def get_top_metrics(self, limit=5):
        """Get metrics for top performing repositories"""
        with self.conn as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT repo_url, code_quality_score, samples_count, success_rate
                FROM repo_metrics
                ORDER BY code_quality_score * samples_count DESC
                LIMIT ?
            """, (limit,))
            
            results = cursor.fetchall()
            if not results:
                return "No repository metrics available yet."
                
            metrics_text = ""
            for repo_url, quality, count, success_rate in results:
                metrics_text += f"- {repo_url}: Quality={quality:.2f}, Samples={count}, Success={success_rate:.2f}\n"
            
            return metrics_text
    
    def save_patterns(self):
        """Save learned patterns to a file for future runs"""
        with self.conn as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM repo_metrics")
            all_metrics = cursor.fetchall()
            
            patterns = {
                "top_repos": [],
                "metrics_summary": {
                    "avg_quality": 0,
                    "avg_samples": 0,
                    "total_repos": len(all_metrics)
                }
            }
            
            # Calculate summary metrics
            if all_metrics:
                quality_sum = sum(row[1] for row in all_metrics if row[1])
                samples_sum = sum(row[2] for row in all_metrics if row[2])
                patterns["metrics_summary"]["avg_quality"] = quality_sum / len(all_metrics)
                patterns["metrics_summary"]["avg_samples"] = samples_sum / len(all_metrics)
                
                # Get top 10 repos
                cursor.execute("""
                    SELECT repo_url, code_quality_score, samples_count
                    FROM repo_metrics
                    ORDER BY code_quality_score * samples_count DESC
                    LIMIT 10
                """)
                patterns["top_repos"] = [{"url": row[0], "quality": row[1], "samples": row[2]} 
                                        for row in cursor.fetchall()]
            
            # Save to file
            with open("learned_patterns.json", "w") as f:
                json.dump(patterns, f, indent=2)


def quality_filter(code_sample: str) -> bool:
    """Filter code samples based on quality criteria"""
    quality_checks = [
        len(code_sample.strip()) > 50,  # Minimum length
        "func" in code_sample,  # Contains functions
        code_sample.count("\n") > 5,  # Multiple lines
        "# " in code_sample or "## " in code_sample  # Has comments
    ]
    return sum(quality_checks) >= 3


def rank_code_sample(code_sample: dict) -> float:
    """Rank a code sample based on educational value and code quality"""
    code = code_sample.get('code', '')
    if not code:
        return 0.0
        
    # Basic metrics
    lines = code.split('\n')
    num_lines = len(lines)
    num_functions = code.count('func ')
    num_comments = sum(1 for line in lines if line.strip().startswith('#'))
    has_class = 'class_name' in code or 'extends' in code
    
    # Calculate complexity (simple version)
    complexity = min(1.0, num_lines / 200)
    
    # Calculate educational value
    educational_value = (
        (0.3 * (num_comments / max(1, num_lines))) +  # Comments ratio
        (0.3 * (num_functions / max(1, num_lines/10))) +  # Functions density
        (0.2 * (1 if has_class else 0)) +  # Class structure
        (0.2 * complexity)  # Complexity
    )
    
    return min(1.0, educational_value)


def refine_code_sample(code_sample: dict) -> dict:
    """Refine a code sample by fixing formatting and ensuring consistent style"""
    code = code_sample.get('code', '')
    if not code:
        return code_sample
        
    # Basic refinements
    lines = code.split('\n')
    refined_lines = []
    
    for line in lines:
        # Fix indentation (convert tabs to spaces)
        line = line.replace('\t', '    ')
        
        # Ensure consistent spacing around operators
        for op in ['+', '-', '*', '/', '=', '==', '!=', '<=', '>=']:
            line = line.replace(op, f' {op} ')
            line = line.replace(f'  {op}  ', f' {op} ')  # Fix double spaces
        
        refined_lines.append(line)
    
    # For more complex refinements, we could use the LLM
    if len(code) < 1000:  # Only for reasonably sized samples
        prompt = f"""Refine this GDScript code to follow best practices and consistent style:
        
        ```gdscript
        {code}
        ```
        
        Return only the refined code, no explanations.
        """
        try:
            refined_code = openai_call(prompt)
            # Extract code if it's wrapped in backticks
            if "```" in refined_code:
                refined_code = refined_code.split("```")[1]
                if refined_code.startswith("gdscript\n"):
                    refined_code = refined_code[9:]
            code_sample['code'] = refined_code
            return code_sample
        except:
            # Fallback to basic refinement if LLM call fails
            code_sample['code'] = '\n'.join(refined_lines)
            return code_sample
    
    code_sample['code'] = '\n'.join(refined_lines)
    return code_sample


def clean_code_sample(code_sample: dict) -> dict:
    """Clean a code sample by removing comments, print statements, and debugging code"""
    code = code_sample.get('code', '')
    if not code:
        return code_sample
        
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip comment lines
        if line.strip().startswith('#'):
            continue
            
        # Remove inline comments
        if '#' in line:
            line = line.split('#')[0]
            
        # Skip print statements used for debugging
        if 'print(' in line and ('debug' in line.lower() or 'log' in line.lower()):
            continue
            
        # Skip empty lines after removing comments
        if line.strip():
            cleaned_lines.append(line)
    
    code_sample['code'] = '\n'.join(cleaned_lines)
    return code_sample


def strategic_task_planning(learning_system, objective: str, completed_tasks: List[str]):
    """Create higher-level task planning based on learned patterns"""
    
    prompt = f"""Based on the objective: {objective}
    Analyze our repository success patterns and create a strategic plan.
    
    Recent metrics:
    {learning_system.get_top_metrics()}
    
    Create a prioritized task list that:
    1. Focuses on repositories with similar patterns to successful ones
    2. Avoids common failure patterns
    3. Maximizes code quality and variety
    
    Return tasks in the standard format:
    #. Task description
    """
    
    response = openai_call(prompt)
    
    # Parse the response into tasks
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2 and task_parts[0].isnumeric():
            task_name = task_parts[1].strip()
            if task_name:
                task_name = normalize_task_url(task_name)
                new_tasks_list.append({"task_name": task_name})
    
    return new_tasks_list


def normalize_task_url(task: str) -> str:
    """Normalize only the URL part of a task string."""
    url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task)
    if url_match:
        return task  # Already correct
    malformed_match = re.search(r'httpsgithubcom([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)', task)
    if malformed_match:
        fixed_url = f"https://github.com/{malformed_match.group(1)}"
        return task.replace(f"httpsgithubcom{malformed_match.group(1)}", fixed_url)
    return task  # Return unchanged if no URL found


def reset_tasks_for_mode():
    """Reset tasks for the current operation mode."""
    print(f"Resetting tasks for {OPERATION_MODE} mode...")
    with db_connection() as conn:
        cursor = conn.cursor()
        # Delete any existing tasks except completed scrape tasks
        if OPERATION_MODE != "scrape":
            cursor.execute('DELETE FROM tasks WHERE repo_name LIKE ? OR repo_name = ?', 
                          (f"{OPERATION_MODE.capitalize()}%", "Initial_Task"))
        else:
            cursor.execute('DELETE FROM tasks WHERE status != ?', ("completed",))
        
        # Add appropriate tasks for the current mode
        if OPERATION_MODE == "rank":
            print("Adding ranking tasks...")
            add_task("Rank_All_Samples", "Load the collected GDScript samples from the database and rank them based on quality metrics")
            add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
            add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
            # Make sure the tasks are set to pending status
            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                          ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
            conn.commit()
        elif OPERATION_MODE == "refine":
            add_task("Refine_All_Samples", "Load the collected GDScript samples from the database and apply code style improvements")
            add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
            add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
            # Make sure the tasks are set to pending status
            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                          ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
            conn.commit()
        elif OPERATION_MODE == "clean":
            add_task("Clean_All_Samples", "Load the collected GDScript samples from the database and remove comments and debug code")
            add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
            add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
            # Make sure the tasks are set to pending status
            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                          ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
            conn.commit()
    
    print(f"Tasks reset for {OPERATION_MODE} mode.")

def openai_call(prompt: str, model: str = LLM_MODEL, temperature: float = OPENAI_TEMPERATURE, max_tokens: int = 2000):
    """Make a call to the OpenAI API."""
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {str(e)}. Retrying in 10 seconds...")
            time.sleep(10)


def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]):
    """Create new tasks based on the result of the previous task."""
    useless_phrases = ["I can't browse the internet", "open your browser", "navigate to github", "you can do this by",
                       "here's how to", "step by step"]
    if any(phrase in result["data"].lower() for phrase in useless_phrases):
        print("⚠️ Skipping task creation – result was guidance, not output.")
        return []

    total_samples = get_total_samples_count()

    prompt = "You are to use the result from an execution agent to create new tasks with the following objective: " + objective + ".\n"
    prompt += "The last completed task has the result: \n" + result["data"] + "\n"
    prompt += "This result was based on this task description: " + task_description + ".\n"
    if task_list:
        prompt += "These are incomplete tasks: " + ', '.join(task_list) + "\n"
    prompt += f"Based on the result, return a list of tasks to be completed to meet the objective. Current sample count: {total_samples}/{SAMPLE_GOAL}.\n"
    prompt += "These new tasks must not overlap with incomplete tasks and should focus on collecting actual GDScript samples from new repositories.\n"
    prompt += (
        "Return one task per line in your response. The result must be a numbered list in the format:\n"
        "#. Collect GDScript samples from https://github.com/<owner>/<repo>\n"
        "#. Collect GDScript samples from https://github.com/<owner>/<repo>\n"
        "The number of each entry must be followed by a period. Replace <owner>/<repo> with the actual GitHub repository owner and name (e.g., https://github.com/Orama-Interactive/Pixelorama). Use only URLs from the result. Ensure the URL includes 'https://github.com/' exactly as shown, and preserve all spaces in the task string. If your list is empty or the goal is met, write \"There are no tasks to add at this time.\"\n"
    )
    response = openai_call(prompt, max_tokens=2000)
    print(f"DEBUG: task_creation_agent raw response:\n{response}")
    new_tasks = response.split('\n')
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2 and task_parts[0].isnumeric():
            task_name = task_parts[1].strip()  # Keep original spacing
            if task_name and task_name not in task_list:
                task_name = normalize_task_url(task_name)  # Normalize only URL part
                new_tasks_list.append(task_name)
    return [{"task_name": task_name} for task_name in new_tasks_list]


def prioritization_agent(task_list: List[str]):
    """Prioritize the list of tasks."""
    if not task_list:
        return []

    task_list_str = '\n'.join(task_list)
    prompt = (
            "You are tasked with prioritizing the following tasks: \n" +
            task_list_str + "\n"
                            "Consider the ultimate objective: " + OBJECTIVE + ".\n" +
            "Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites or are more essential for meeting the objective.\n" +
            "Do not remove any tasks or modify their text. Return the ranked tasks exactly as provided, in a numbered list in the format:\n" +
            "#. First task\n" +
            "#. Second task\n" +
            "The number of each entry must be followed by a period.\n"
    )
    response = openai_call(prompt, max_tokens=2000)
    print(f"DEBUG: prioritization_agent raw response:\n{response}")
    if not response:
        print('Received empty response from prioritization agent. Keeping task list unchanged.')
        return task_list

    new_tasks = response.split("\n")
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2 and task_parts[0].isnumeric():
            task_name = normalize_task_url(task_parts[1].strip())
            if task_name:
                new_tasks_list.append(task_name)

    # If we lost any tasks in the prioritization, add them back at the end
    for task in task_list:
        if task not in new_tasks_list:
            new_tasks_list.append(task)

    return new_tasks_list


def execution_agent(objective: str, task: str) -> str:
    """Execute a task and return the result."""
    task_lower = task.lower().strip()
    
    # Handle tasks based on operation mode
    if OPERATION_MODE == "scrape":
        if "search" in task_lower and "github" in task_lower:
            print(f"🔍 Executing GitHub search task: {task}")
            try:
                from extensions.github_scraper import search_godot_repos
                results = search_godot_repos()
                if not results:
                    return "No repositories found."
                return "\n".join([f"{r['name']} - {r['url']} ({r['stars']}⭐)" for r in results])
            except ImportError as e:
                print(f"Error importing github_scraper: {str(e)}")
                # Fallback to direct implementation if import fails
                import requests
                import os
                
                github_token = os.getenv("GITHUB_API_KEY")
                if not github_token:
                    return "GITHUB_API_KEY environment variable is missing."
                    
                headers = {
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github+json"
                }
                query = "godot language:GDScript stars:>10"
                url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=30"
                
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    results = []
                    for repo in data.get("items", []):
                        results.append({
                            "name": repo["full_name"],
                            "url": repo["html_url"],
                            "stars": repo["stargazers_count"]
                        })
                    if not results:
                        return "No repositories found."
                    return "\n".join([f"{r['name']} - {r['url']} ({r['stars']}⭐)" for r in results])
                except Exception as e:
                    return f"GitHub API error: {str(e)}"
            except Exception as e:
                return f"GitHub search failed: {str(e)}"

        if "collect gdscript" in task_lower or "github.com" in task_lower:
            print(f"🧠 Executing GDScript collection task: {task}")

            # Extract URL with robust fallback
            task = normalize_task_url(task)  # Normalize first
            url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task)
            if url_match:
                repo_url = url_match.group(0)
                repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
            else:
                return f"❌ No valid GitHub repo URL found in task: {task}"

            print(f"DEBUG: Extracted repo URL: {repo_url}")
            try:
                results = collect_gdscript_from_repo(repo_url)
                if not results:
                    return f"⚠️ No .gd files found in {repo_url}"
                    
                # Apply quality filter to results
                filtered_results = []
                for result in results:
                    if quality_filter(result['code']):
                        filtered_results.append(result)
                
                if not filtered_results:
                    return f"⚠️ Found {len(results)} GDScript files in {repo_url}, but none passed quality filter"
                    
                print(f"💾 Collected {len(filtered_results)} quality GDScript samples from {repo_url} (filtered from {len(results)} total)")
                return "\n\n".join([f"// {r['path']}\n{r['code'][:500]}..." for r in filtered_results[:5]])
            except Exception as e:
                return f"🔥 GDScript collection failed: {str(e)}"
    
    elif OPERATION_MODE == "rank":
        if "load" in task_lower and "samples" in task_lower or "rank" in task_lower:
            print(f"📊 Loading samples for ranking: {task}")
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM samples')
                    samples_count = cursor.fetchone()[0]
                    print(f"DEBUG: Found {samples_count} samples in the database")
                    
                    cursor.execute('SELECT s.file_path, s.code_content FROM samples s JOIN tasks t ON s.repo_id = t.id WHERE t.status = ?', ("completed",))
                    all_samples = cursor.fetchall()
                    
                if not all_samples:
                    return "No samples found in the database to rank. Please run in 'scrape' mode first to collect samples."
                
                sample_count = 0
                ranked_samples = []
                
                for file_path, code_content in all_samples:
                    try:
                        sample = {"path": file_path, "code": code_content}
                        rank = rank_code_sample(sample)
                        sample['rank'] = rank
                        ranked_samples.append(sample)
                        sample_count += 1
                    except Exception as e:
                        print(f"Error processing sample: {str(e)}")
                        continue
                
                if not ranked_samples:
                    return "Could not process any samples. Check if the samples in the database are in the correct format."
                
                print(f"DEBUG: Successfully ranked {sample_count} samples")
                
                # Sort samples by rank
                ranked_samples.sort(key=lambda x: x.get('rank', 0), reverse=True)
                
                # Save ranked samples to a new table
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS ranked_samples (
                            id INTEGER PRIMARY KEY,
                            path TEXT,
                            code TEXT,
                            rank FLOAT
                        )
                    ''')
                    
                    # Clear existing ranked samples
                    cursor.execute('DELETE FROM ranked_samples')
                    
                    # Insert new ranked samples
                    for sample in ranked_samples:
                        cursor.execute(
                            'INSERT INTO ranked_samples (path, code, rank) VALUES (?, ?, ?)',
                            (sample.get('path', ''), sample.get('code', ''), sample.get('rank', 0))
                        )
                
                # Return summary of ranking
                top_samples = ranked_samples[:5] if len(ranked_samples) >= 5 else ranked_samples
                summary = f"Ranked {sample_count} code samples. Top 5 samples:\n\n"
                for i, sample in enumerate(top_samples):
                    summary += f"{i+1}. {sample.get('path', 'Unknown')} (Score: {sample.get('rank', 0):.2f})\n"
                    summary += f"Sample preview: {sample.get('code', '')[:100]}...\n\n"
                
                return summary
            except Exception as e:
                return f"🔥 Sample ranking failed: {str(e)}"
        elif "analyze" in task_lower and "top" in task_lower:
            print(f"📈 Analyzing top samples: {task}")
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    # First check if ranked_samples table has data
                    cursor.execute('SELECT COUNT(*) FROM ranked_samples')
                    ranked_count = cursor.fetchone()[0]
                    
                    # If no ranked samples exist, try to load them from the samples table and rank them
                    if ranked_count == 0:
                        print("No ranked samples found. Ranking samples first...")
                        cursor.execute('SELECT s.file_path, s.code_content FROM samples s JOIN tasks t ON s.repo_id = t.id WHERE t.status = ?', ("completed",))
                        all_samples = cursor.fetchall()
                        
                        if not all_samples:
                            return "No samples found in the database to rank. Please run in 'scrape' mode first to collect samples."
                        
                        sample_count = 0
                        ranked_samples = []
                        
                        for file_path, code_content in all_samples:
                            try:
                                sample = {"path": file_path, "code": code_content}
                                rank = rank_code_sample(sample)
                                sample['rank'] = rank
                                ranked_samples.append(sample)
                                sample_count += 1
                            except Exception as e:
                                print(f"Error processing sample: {str(e)}")
                                continue
                        
                        # Sort samples by rank
                        ranked_samples.sort(key=lambda x: x.get('rank', 0), reverse=True)
                        
                        # Save ranked samples to the ranked_samples table
                        cursor.execute('''
                            CREATE TABLE IF NOT EXISTS ranked_samples (
                                id INTEGER PRIMARY KEY,
                                path TEXT,
                                code TEXT,
                                rank FLOAT
                            )
                        ''')
                        
                        # Clear existing ranked samples
                        cursor.execute('DELETE FROM ranked_samples')
                        
                        # Insert new ranked samples
                        for sample in ranked_samples:
                            cursor.execute(
                                'INSERT INTO ranked_samples (path, code, rank) VALUES (?, ?, ?)',
                                (sample.get('path', ''), sample.get('code', ''), sample.get('rank', 0))
                            )
                        
                        print(f"Ranked and saved {sample_count} samples to the database")
                    
                    # Now get the top samples for analysis
                    cursor.execute('SELECT path, code, rank FROM ranked_samples ORDER BY rank DESC LIMIT 20')
                    top_samples = cursor.fetchall()
                
                if not top_samples:
                    return "No ranked samples found even after attempting to rank them. There may be an issue with the database."
                
                # Prepare samples for analysis
                samples_text = ""
                for i, (path, code, rank) in enumerate(top_samples[:5]):
                    samples_text += f"Sample {i+1} (Score: {rank:.2f}):\nPath: {path}\n```\n{code[:300]}...\n```\n\n"
                
                # Analyze the top samples
                analysis_prompt = f"""Analyze these top-ranked GDScript code samples and identify:
                1. Common patterns and best practices
                2. What makes these samples high quality
                3. Key features that contribute to their high ranking
                4. How these patterns could be applied to improve other code
                
                Top samples:
                {samples_text}
                
                Provide a detailed analysis with specific examples from the code.
                """
                
                analysis = openai_call(analysis_prompt, max_tokens=2000)
                return f"Analysis of top-ranked samples:\n\n{analysis}"
            except Exception as e:
                return f"🔥 Sample analysis failed: {str(e)}"
        elif "generate" in task_lower and "report" in task_lower:
            print(f"📝 Generating quality report: {task}")
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    # First check if ranked_samples table has data
                    cursor.execute('SELECT COUNT(*) FROM ranked_samples')
                    ranked_count = cursor.fetchone()[0]
                    
                    # If no ranked samples exist, try to load them from the samples table and rank them
                    if ranked_count == 0:
                        print("No ranked samples found. Ranking samples first...")
                        cursor.execute('SELECT s.file_path, s.code_content FROM samples s JOIN tasks t ON s.repo_id = t.id WHERE t.status = ?', ("completed",))
                        all_samples = cursor.fetchall()
                        
                        if not all_samples:
                            return "No samples found in the database to rank. Please run in 'scrape' mode first to collect samples."
                        
                        sample_count = 0
                        ranked_samples = []
                        
                        for file_path, code_content in all_samples:
                            try:
                                sample = {"path": file_path, "code": code_content}
                                rank = rank_code_sample(sample)
                                sample['rank'] = rank
                                ranked_samples.append(sample)
                                sample_count += 1
                            except Exception as e:
                                print(f"Error processing sample: {str(e)}")
                                continue
                        
                        # Sort samples by rank
                        ranked_samples.sort(key=lambda x: x.get('rank', 0), reverse=True)
                        
                        # Save ranked samples to the ranked_samples table
                        cursor.execute('''
                            CREATE TABLE IF NOT EXISTS ranked_samples (
                                id INTEGER PRIMARY KEY,
                                path TEXT,
                                code TEXT,
                                rank FLOAT
                            )
                        ''')
                        
                        # Clear existing ranked samples
                        cursor.execute('DELETE FROM ranked_samples')
                        
                        # Insert new ranked samples
                        for sample in ranked_samples:
                            cursor.execute(
                                'INSERT INTO ranked_samples (path, code, rank) VALUES (?, ?, ?)',
                                (sample.get('path', ''), sample.get('code', ''), sample.get('rank', 0))
                            )
                        
                        print(f"Ranked and saved {sample_count} samples to the database")
                    
                    # Now get the statistics for the report
                    cursor.execute('SELECT COUNT(*), AVG(rank), MAX(rank), MIN(rank) FROM ranked_samples')
                    count, avg_rank, max_rank, min_rank = cursor.fetchone()
                    
                    # Get distribution of ranks
                    cursor.execute('''
                        SELECT 
                            CASE 
                                WHEN rank >= 0.8 THEN 'Excellent (0.8-1.0)'
                                WHEN rank >= 0.6 THEN 'Good (0.6-0.8)'
                                WHEN rank >= 0.4 THEN 'Average (0.4-0.6)'
                                WHEN rank >= 0.2 THEN 'Below Average (0.2-0.4)'
                                ELSE 'Poor (0.0-0.2)'
                            END as quality_tier,
                            COUNT(*) as count
                        FROM ranked_samples
                        GROUP BY quality_tier
                        ORDER BY MIN(rank) DESC
                    ''')
                    distribution = cursor.fetchall()
                
                if not count:
                    return "No ranked samples found even after attempting to rank them. There may be an issue with the database."
                
                # Generate the report
                report = f"""# GDScript Code Quality Report

## Overview
- Total samples analyzed: {count}
- Average quality score: {avg_rank:.2f}
- Highest quality score: {max_rank:.2f}
- Lowest quality score: {min_rank:.2f}

## Quality Distribution
"""
                
                for tier, tier_count in distribution:
                    percentage = (tier_count / count) * 100
                    report += f"- {tier}: {tier_count} samples ({percentage:.1f}%)\n"
                
                # Add analysis of what makes high-quality samples
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT path, code, rank FROM ranked_samples ORDER BY rank DESC LIMIT 3')
                    top_samples = cursor.fetchall()
                
                top_samples_text = ""
                for i, (path, code, rank) in enumerate(top_samples):
                    top_samples_text += f"Sample {i+1} (Score: {rank:.2f}):\nPath: {path}\n```\n{code[:200]}...\n```\n\n"
                
                analysis_prompt = f"""Based on these top-ranked GDScript samples:
                
                {top_samples_text}
                
                Provide an analysis of what characteristics make for high-quality GDScript code.
                Focus on patterns, practices, and features that contribute to code quality.
                """
                
                quality_analysis = openai_call(analysis_prompt, max_tokens=1000)
                
                report += f"\n## What Makes High-Quality GDScript Code\n{quality_analysis}\n"
                
                return report
            except Exception as e:
                return f"🔥 Report generation failed: {str(e)}"
    
    elif OPERATION_MODE == "refine":
        if "load" in task_lower and "samples" in task_lower:
            print(f"🔧 Loading samples for refinement: {task}")
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM samples')
                    samples_count = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT s.file_path, s.code_content FROM samples s JOIN tasks t ON s.repo_id = t.id WHERE t.status = ?', ("completed",))
                    all_samples = cursor.fetchall()
                    
                if not all_samples:
                    return "No samples found in the database to refine. Please run in 'scrape' mode first to collect samples."
                
                sample_count = 0
                refined_samples = []
                
                for file_path, code_content in all_samples:
                    try:
                        sample = {"path": file_path, "code": code_content}
                        refined_sample = refine_code_sample(sample)
                        refined_samples.append(refined_sample)
                        sample_count += 1
                    except Exception as e:
                        print(f"Error refining sample: {str(e)}")
                        continue
                
                if not refined_samples:
                    return "Could not refine any samples. Check if the samples in the database are in the correct format."
                
                # Save refined samples to a new table
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS refined_samples (
                            id INTEGER PRIMARY KEY,
                            path TEXT,
                            code TEXT
                        )
                    ''')
                    
                    # Clear existing refined samples
                    cursor.execute('DELETE FROM refined_samples')
                    
                    # Insert new refined samples
                    for sample in refined_samples:
                        cursor.execute(
                            'INSERT INTO refined_samples (path, code) VALUES (?, ?)',
                            (sample.get('path', ''), sample.get('code', ''))
                        )
                
                # Return summary of refinement
                summary = f"Refined {sample_count} code samples. Examples of refinements:\n\n"
                for i, sample in enumerate(refined_samples[:3]):
                    summary += f"{i+1}. {sample.get('path', 'Unknown')}\n"
                    summary += f"Sample preview: {sample.get('code', '')[:100]}...\n\n"
                
                return summary
            except Exception as e:
                return f"🔥 Sample refinement failed: {str(e)}"
    
    elif OPERATION_MODE == "clean":
        if "load" in task_lower and "samples" in task_lower:
            print(f"🧹 Loading samples for cleaning: {task}")
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM samples')
                    samples_count = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT s.file_path, s.code_content FROM samples s JOIN tasks t ON s.repo_id = t.id WHERE t.status = ?', ("completed",))
                    all_samples = cursor.fetchall()
                    
                if not all_samples:
                    return "No samples found in the database to clean."
                
                sample_count = 0
                cleaned_samples = []
                
                for file_path, code_content in all_samples:
                    try:
                        sample = {"path": file_path, "code": code_content}
                        cleaned_sample = clean_code_sample(sample)
                        cleaned_samples.append(cleaned_sample)
                        sample_count += 1
                    except Exception as e:
                        print(f"Error cleaning sample: {str(e)}")
                        continue
                
                # Save cleaned samples to a new table
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS cleaned_samples (
                            id INTEGER PRIMARY KEY,
                            path TEXT,
                            code TEXT
                        )
                    ''')
                    
                    # Clear existing cleaned samples
                    cursor.execute('DELETE FROM cleaned_samples')
                    
                    # Insert new cleaned samples
                    for sample in cleaned_samples:
                        cursor.execute(
                            'INSERT INTO cleaned_samples (path, code) VALUES (?, ?)',
                            (sample.get('path', ''), sample.get('code', ''))
                        )
                
                # Return summary of cleaning
                summary = f"Cleaned {sample_count} code samples. Examples of cleaning:\n\n"
                for i, sample in enumerate(cleaned_samples[:3]):
                    summary += f"{i+1}. {sample.get('path', 'Unknown')}\n"
                    summary += f"Sample preview: {sample.get('code', '')[:100]}...\n\n"
                
                return summary
            except Exception as e:
                return f"🔥 Sample cleaning failed: {str(e)}"

    # Default handling for any task in any mode
    prompt = f'Perform one task based on the following objective: {objective}.\nYour task: {task}\nResponse:'
    return openai_call(prompt, max_tokens=2000)


def main():
    """Main function to run the BabyAGI agent with learning capabilities."""
    # Initialize the learning system
    learning_system = LearningSystem(db_connection())
    
    # Reset database if requested
    if RESET_DATABASE:
        print("Resetting database as requested...")
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM tasks')
            try:
                cursor.execute('DELETE FROM ranked_samples')
            except:
                pass
            try:
                cursor.execute('DELETE FROM refined_samples')
            except:
                pass
            try:
                cursor.execute('DELETE FROM cleaned_samples')
            except:
                pass
            print("Database reset complete.")
    
    # Check if we're in a non-scrape mode and have samples
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM samples')
        samples_count = cursor.fetchone()[0]
        print(f"DEBUG: Found {samples_count} samples in the database")
        
        # If we're in rank/refine/clean mode and have samples, reset the tasks for that mode
        if OPERATION_MODE in ["rank", "refine", "clean"] and samples_count > 0:
            reset_tasks_for_mode()
            
            # Verify that tasks were created and are in pending status
            cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', ("pending",))
            pending_count = cursor.fetchone()[0]
            print(f"DEBUG: After reset, found {pending_count} pending tasks")
            
            if pending_count == 0 and OPERATION_MODE == "rank":
                print("WARNING: No pending tasks found after reset. Manually setting tasks to pending status.")
                cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                              ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                conn.commit()
    
    # Track completed tasks and task success metrics
    tasks_completed = 0
    
    # Check if we need to add the initial task
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Debug: Check how many samples we have
        cursor.execute('SELECT COUNT(*) FROM samples')
        samples_count = cursor.fetchone()[0]
        print(f"DEBUG: Found {samples_count} samples in the database")
        
        # Check if we need to add the initial task
        cursor.execute('SELECT COUNT(*) FROM tasks')
        if cursor.fetchone()[0] == 0:
            print("Adding initial task.")
            # Always use the mode-specific initial task, ignoring INITIAL_TASK variable
            mode_specific_task = INITIAL_TASKS.get(OPERATION_MODE, INITIAL_TASKS["scrape"])
            print(f"Using mode-specific initial task: {mode_specific_task}")
            add_task("Initial_Task", mode_specific_task)
            # Clear any existing tasks table to ensure we're starting fresh
            cursor.execute('DELETE FROM tasks WHERE repo_name != "Initial_Task"')
        else:
            print(f"DEBUG: Found existing tasks in the database, not adding initial task")
            
            # If we're in rank/refine/clean mode and have no rank-specific tasks, add them
            if OPERATION_MODE in ["rank", "refine", "clean"]:
                # Check if we have any mode-specific tasks
                cursor.execute('SELECT COUNT(*) FROM tasks WHERE repo_name LIKE ?', (f"{OPERATION_MODE.capitalize()}%",))
                mode_tasks_count = cursor.fetchone()[0]
                
                if mode_tasks_count == 0 and samples_count > 0:
                    print(f"DEBUG: No {OPERATION_MODE} tasks found but {samples_count} samples exist. Adding {OPERATION_MODE} tasks...")
                    
                    # Add appropriate tasks for the current mode
                    if OPERATION_MODE == "rank":
                        print("Adding ranking tasks...")
                        add_task("Rank_All_Samples", "Load the collected GDScript samples from the database and rank them based on quality metrics")
                        add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
                        add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
                        # Make sure the tasks are set to pending status
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                        conn.commit()
                    elif OPERATION_MODE == "refine":
                        add_task("Refine_All_Samples", "Load the collected GDScript samples from the database and apply code style improvements")
                        add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
                        add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
                        # Make sure the tasks are set to pending status
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                        conn.commit()
                    elif OPERATION_MODE == "clean":
                        add_task("Clean_All_Samples", "Load the collected GDScript samples from the database and remove comments and debug code")
                        add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
                        add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
                        # Make sure the tasks are set to pending status
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                        conn.commit()
    
    # Main loop
    while True:
        # Get pending tasks from the database
        pending_tasks = get_pending_tasks()
        
        if not pending_tasks:
            # If no pending tasks, check if we need to create more
            with db_connection() as conn:
                cursor = conn.cursor()
                # Use the correct status value that matches what we set in update_task_status
                cursor.execute('SELECT repo_name, repo_url FROM tasks WHERE status = ? ORDER BY last_updated DESC LIMIT 5', ("completed",))
                completed_tasks = cursor.fetchall()
                
                # Debug: Check how many samples we have
                cursor.execute('SELECT COUNT(*) FROM samples')
                samples_count = cursor.fetchone()[0]
                print(f"DEBUG: Found {samples_count} samples in the database")
                
                # Check if there are any tasks at all
                cursor.execute('SELECT COUNT(*) FROM tasks')
                total_tasks = cursor.fetchone()[0]
                
                # Check if we have tasks but they're not in pending status
                if total_tasks > 0 and OPERATION_MODE in ["rank", "refine", "clean"]:
                    cursor.execute('SELECT COUNT(*) FROM tasks WHERE repo_name LIKE ?', (f"{OPERATION_MODE.capitalize()}%",))
                    mode_tasks_count = cursor.fetchone()[0]
                    
                    if mode_tasks_count > 0:
                        print(f"DEBUG: Found {mode_tasks_count} {OPERATION_MODE} tasks but none are pending. Resetting their status...")
                        if OPERATION_MODE == "rank":
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                        elif OPERATION_MODE == "refine":
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                        elif OPERATION_MODE == "clean":
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                        conn.commit()
                        # Refresh pending tasks
                        pending_tasks = get_pending_tasks()
                        if pending_tasks:
                            print(f"Successfully reset task status. Found {len(pending_tasks)} pending tasks.")
                            continue
            
            # If there are no tasks at all but we have samples, add mode-specific tasks
            if total_tasks == 0 and samples_count > 0 and OPERATION_MODE in ["rank", "refine", "clean"]:
                print(f"DEBUG: No tasks found but {samples_count} samples exist. Adding {OPERATION_MODE} tasks...")
                
                # Add appropriate tasks for the current mode
                if OPERATION_MODE == "rank":
                    print("Adding ranking tasks...")
                    add_task("Rank_All_Samples", "Load the collected GDScript samples from the database and rank them based on quality metrics")
                    add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
                    add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
                    # Make sure the tasks are set to pending status
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                        conn.commit()
                elif OPERATION_MODE == "refine":
                    add_task("Refine_All_Samples", "Load the collected GDScript samples from the database and apply code style improvements")
                    add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
                    add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
                    # Make sure the tasks are set to pending status
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                        conn.commit()
                elif OPERATION_MODE == "clean":
                    add_task("Clean_All_Samples", "Load the collected GDScript samples from the database and remove comments and debug code")
                    add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
                    add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
                    # Make sure the tasks are set to pending status
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                      ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                        conn.commit()
                
                # Continue to the next iteration to process the new tasks
                continue
            
            if not completed_tasks:
                print("No completed tasks to generate new tasks from.")
                
                # If we're in rank/refine/clean mode and have samples but no tasks, add mode-specific tasks
                if OPERATION_MODE in ["rank", "refine", "clean"] and samples_count > 0:
                    print(f"DEBUG: No completed tasks but {samples_count} samples exist. Adding {OPERATION_MODE} tasks...")
                    
                    # Add appropriate tasks for the current mode
                    if OPERATION_MODE == "rank":
                        add_task("Rank_All_Samples", "Load the collected GDScript samples from the database and rank them based on quality metrics")
                        add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
                        add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                            conn.commit()
                    elif OPERATION_MODE == "refine":
                        add_task("Refine_All_Samples", "Load the collected GDScript samples from the database and apply code style improvements")
                        add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
                        add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                            conn.commit()
                    elif OPERATION_MODE == "clean":
                        add_task("Clean_All_Samples", "Load the collected GDScript samples from the database and remove comments and debug code")
                        add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
                        add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                            conn.commit()
                    
                    # Continue to the next iteration to process the new tasks
                    continue
                # If we have no completed tasks and no pending tasks, but we're just starting,
                # let's execute the initial task directly
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM tasks')
                    total_tasks = cursor.fetchone()[0]
                
                if total_tasks <= 1:
                    # Always use the mode-specific initial task, ignoring INITIAL_TASK variable
                    mode_specific_task = INITIAL_TASKS.get(OPERATION_MODE, INITIAL_TASKS["scrape"])
                    print(f"Executing initial {OPERATION_MODE} task: {mode_specific_task}")
                    result = execution_agent(OBJECTIVE, mode_specific_task)
                    print("\n*****INITIAL TASK RESULT*****\n")
                    print(result)
                    
                    if OPERATION_MODE == "scrape":
                        # Create tasks from the search results
                        enriched_result = {"data": result}
                        new_tasks = task_creation_agent(OBJECTIVE, enriched_result, INITIAL_TASK, [])
                        
                        for new_task in new_tasks:
                            task_name = new_task["task_name"]
                            # Extract URL
                            url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task_name)
                            if url_match:
                                repo_url = url_match.group(0)
                                repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
                                add_task(repo_name, repo_url)
                                print(f"Added task: {repo_name} - {repo_url}")
                    else:
                        # Check if we have samples in the database for non-scrape modes
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('SELECT COUNT(*) FROM samples')
                            samples_count = cursor.fetchone()[0]
                        
                        if samples_count == 0 and OPERATION_MODE in ["rank", "refine", "clean"]:
                            print(f"\n*****WARNING*****\n")
                            print(f"No code samples found in the database for {OPERATION_MODE} mode.")
                            print("You need to run in 'scrape' mode first to collect samples.")
                            print("Switching to 'scrape' mode to collect initial samples...")
                            
                            # Switch to scrape mode temporarily
                            temp_operation_mode = "scrape"
                            temp_objective = OBJECTIVES["scrape"]
                            temp_initial_task = INITIAL_TASKS["scrape"]
                            
                            # Execute a search task to find repositories
                            search_result = execution_agent(temp_objective, temp_initial_task)
                            
                            # Create tasks from the search results
                            enriched_result = {"data": search_result}
                            new_tasks = task_creation_agent(temp_objective, enriched_result, temp_initial_task, [])
                            
                            for new_task in new_tasks:
                                task_name = new_task["task_name"]
                                # Extract URL
                                url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task_name)
                                if url_match:
                                    repo_url = url_match.group(0)
                                    repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
                                    add_task(repo_name, repo_url)
                                    print(f"Added task: {repo_name} - {repo_url}")
                            
                            print(f"\nCollecting initial samples before switching back to {OPERATION_MODE} mode...")
                            # Mark the initial task as completed
                            update_task_status("Initial_Task", "completed")
                            continue
                        
                        # For other modes with existing samples, create follow-up tasks based on the result
                        if "rank" in OPERATION_MODE:
                            add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
                            add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
                            # Make sure the tasks are set to pending status
                            with db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                              ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                                conn.commit()
                        elif "refine" in OPERATION_MODE:
                            add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
                            add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
                            # Make sure the tasks are set to pending status
                            with db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                              ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                                conn.commit()
                        elif "clean" in OPERATION_MODE:
                            add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
                            add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
                            # Make sure the tasks are set to pending status
                            with db_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                              ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                                conn.commit()
                    
                    # Mark the initial task as completed
                    update_task_status("Initial_Task", "completed")
                    
                    # Continue to the next iteration to process the new tasks
                    continue
                else:
                    # If we have tasks but none are pending or completed, something might be wrong
                    print("No tasks to process. Waiting for new tasks to be added...")
                    
                    # Check if we have tasks but they're not in pending status
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT repo_name, status FROM tasks')
                        all_tasks = cursor.fetchall()
                        
                        if all_tasks:
                            print(f"DEBUG: Found {len(all_tasks)} tasks but none are pending. Current task statuses:")
                            for task_name, status in all_tasks:
                                print(f"  - {task_name}: {status}")
                            
                            # If we're in rank/refine/clean mode, try to reset task statuses
                            if OPERATION_MODE in ["rank", "refine", "clean"]:
                                print(f"Attempting to reset {OPERATION_MODE} task statuses...")
                                if OPERATION_MODE == "rank":
                                    cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                                  ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                                elif OPERATION_MODE == "refine":
                                    cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                                  ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                                elif OPERATION_MODE == "clean":
                                    cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                                  ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                                conn.commit()
                                print("Task statuses reset. Continuing...")
                                continue
                    
                    time.sleep(10)  # Wait for 10 seconds before checking again
                    continue  # Skip to the next iteration instead of exiting
            
            # Periodically trigger strategic planning based on learned patterns
            if tasks_completed > 0 and tasks_completed % 10 == 0 and OPERATION_MODE == "scrape":
                print("\n*****STRATEGIC PLANNING*****\n")
                print("Analyzing repository success patterns to optimize task planning...")
                
                # Get list of completed task names for context
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT repo_url FROM tasks WHERE status = ?', ("completed",))
                    completed_task_urls = [row[0] for row in cursor.fetchall()]
                
                # Generate strategic tasks
                new_strategic_tasks = strategic_task_planning(
                    learning_system,
                    OBJECTIVE,
                    completed_task_urls
                )
                
                if new_strategic_tasks:
                    print(f"Adding {len(new_strategic_tasks)} strategically planned tasks.")
                    for task in new_strategic_tasks:
                        task_name = task["task_name"]
                        # Extract URL
                        url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task_name)
                        if url_match:
                            repo_url = url_match.group(0)
                            repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
                            add_task(repo_name, repo_url)
                            print(f"Added strategic task: {repo_name} - {repo_url}")
            
            # If we still have no pending tasks after all the above, wait and check again
            if not get_pending_tasks():
                print("Waiting for new tasks to be added...")
                time.sleep(10)
                continue
                
            # Refresh the pending tasks list
            pending_tasks = get_pending_tasks()
            if not pending_tasks:
                continue  # If still no pending tasks, go back to the beginning of the loop
        
        # Process pending tasks
        print("\n*****TASK LIST*****\n")
        
        # Check if we have any pending tasks
        if not pending_tasks:
            print("No pending tasks found. Adding more repositories...")
            
            # Create new tasks by searching for more repositories
            if OPERATION_MODE == "scrape":
                # Execute a search task to find more repositories
                search_task = "Search GitHub for Godot game repositories with over 10 stars and list their URLs."
                search_result = execution_agent(OBJECTIVE, search_task)
                
                # Create tasks from the search results
                enriched_result = {"data": search_result}
                new_tasks = task_creation_agent(OBJECTIVE, enriched_result, search_task, [])
                
                if new_tasks:
                    for new_task in new_tasks:
                        task_name = new_task["task_name"]
                        # Extract URL
                        url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task_name)
                        if url_match:
                            repo_url = url_match.group(0)
                            repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
                            add_task(repo_name, repo_url)
                            print(f"Added task: {repo_name} - {repo_url}")
                
                # Get the updated pending tasks
                pending_tasks = get_pending_tasks()
                
                # If still no pending tasks, exit
                if not pending_tasks:
                    print("Could not find any new repositories to process. Exiting.")
                    break
            elif OPERATION_MODE in ["rank", "refine", "clean"]:
                # For non-scrape modes, check if we have samples in the database
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM samples')
                    samples_count = cursor.fetchone()[0]
                
                if samples_count == 0:
                    print(f"\n*****WARNING*****\n")
                    print(f"No code samples found in the database for {OPERATION_MODE} mode.")
                    print("You need to run in 'scrape' mode first to collect samples.")
                    print("Switching to 'scrape' mode to collect initial samples...")
                    
                    # Switch to scrape mode temporarily
                    temp_operation_mode = "scrape"
                    temp_objective = OBJECTIVES["scrape"]
                    temp_initial_task = INITIAL_TASKS["scrape"]
                    
                    # Execute a search task to find repositories
                    search_result = execution_agent(temp_objective, temp_initial_task)
                    
                    # Create tasks from the search results
                    enriched_result = {"data": search_result}
                    new_tasks = task_creation_agent(temp_objective, enriched_result, temp_initial_task, [])
                    
                    for new_task in new_tasks:
                        task_name = new_task["task_name"]
                        # Extract URL
                        url_match = re.search(r'https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', task_name)
                        if url_match:
                            repo_url = url_match.group(0)
                            repo_name = repo_url.split('/')[-2] + "_" + repo_url.split('/')[-1]
                            add_task(repo_name, repo_url)
                            print(f"Added task: {repo_name} - {repo_url}")
                    
                    print(f"\nCollecting initial samples before switching back to {OPERATION_MODE} mode...")
                    continue
                else:
                    # If we have samples but no tasks, create appropriate tasks for the mode
                    if OPERATION_MODE == "rank":
                        add_task("Rank_All_Samples", "Load the collected GDScript samples from the database and rank them based on quality metrics")
                        add_task("Analyze_Top_Samples", "Analyze the top-ranked samples to identify common patterns and best practices")
                        add_task("Generate_Quality_Report", "Generate a comprehensive report on code quality metrics across all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Rank_All_Samples", "Analyze_Top_Samples", "Generate_Quality_Report"))
                            conn.commit()
                    elif OPERATION_MODE == "refine":
                        add_task("Refine_All_Samples", "Load the collected GDScript samples from the database and apply code style improvements")
                        add_task("Standardize_Naming", "Standardize variable and function naming conventions across all samples")
                        add_task("Fix_Indentation", "Fix indentation and formatting issues in all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Refine_All_Samples", "Standardize_Naming", "Fix_Indentation"))
                            conn.commit()
                    elif OPERATION_MODE == "clean":
                        add_task("Clean_All_Samples", "Load the collected GDScript samples from the database and remove comments and debug code")
                        add_task("Remove_Debug_Code", "Remove all debugging code and print statements from samples")
                        add_task("Optimize_Imports", "Optimize and clean up import statements in all samples")
                        # Make sure the tasks are set to pending status
                        with db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute('UPDATE tasks SET status = ? WHERE repo_name IN (?, ?, ?)', 
                                          ("pending", "Clean_All_Samples", "Remove_Debug_Code", "Optimize_Imports"))
                            conn.commit()
                    
                    # Get the updated pending tasks
                    pending_tasks = get_pending_tasks()
                    
                    # If still no pending tasks, exit
                    if not pending_tasks:
                        print("Could not create any tasks. Exiting.")
                        break
                    
                    continue
            else:
                # For other modes, exit if no tasks
                print("No tasks to process. Exiting.")
                break
        
        # Display the pending tasks
        for repo_name, repo_url in pending_tasks:
            if OPERATION_MODE == "scrape":
                print(f" • Collect GDScript samples from {repo_url}")
            else:
                print(f" • {repo_name}: {repo_url}")
        
        # Process the first pending task
        repo_name, repo_url = pending_tasks[0]
        
        if OPERATION_MODE == "scrape":
            task_name = f"Collect GDScript samples from {repo_url}"
        else:
            task_name = repo_url  # For non-scrape modes, repo_url field contains the task description
        
        print("\n*****NEXT TASK*****\n")
        print(task_name)
        
        # Update task status to in_progress
        update_task_status(repo_name, "in_progress")
        
        # Execute the task
        result = execution_agent(OBJECTIVE, task_name)
        
        print("\n*****TASK RESULT*****\n")
        print(result)
        
        # Update task status based on result
        result_successful = False
        collected_samples = []
        
        if OPERATION_MODE == "scrape":
            if "No .gd files found" in result or "failed" in result.lower() or "none passed quality filter" in result.lower():
                update_task_status(repo_name, "failed")
            else:
                # The actual file count is updated by the collect_gdscript_from_repo function
                update_task_status(repo_name, "completed")
                tasks_completed += 1
                result_successful = True
                
                # Extract collected samples for learning
                try:
                    # Get the samples from the samples table
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT file_path, code_content FROM samples WHERE repo_id = (SELECT id FROM tasks WHERE repo_name = ?)', (repo_name,))
                        samples_data = cursor.fetchall()
                        if samples_data:
                            collected_samples = [{"path": path, "code": code} for path, code in samples_data]
                        else:
                            # If we can't get samples from the database, create a basic structure from the result
                            sample_texts = result.split("\n\n")
                            collected_samples = [{"path": "unknown", "code": sample} for sample in sample_texts if len(sample) > 50]
                except Exception as e:
                    print(f"Error processing samples for learning: {str(e)}")
        else:
            # For non-scrape modes, consider the task successful if it doesn't contain error messages
            if "failed" in result.lower() or "error" in result.lower():
                update_task_status(repo_name, "failed")
            else:
                update_task_status(repo_name, "completed")
                tasks_completed += 1
                result_successful = True
        
        # Update learning system with results (only for scrape mode)
        if result_successful and OPERATION_MODE == "scrape":
            try:
                learning_system.update_repo_metrics(repo_url, collected_samples)
                print(f"Updated learning metrics for {repo_url}")
            except Exception as e:
                print(f"Error updating learning metrics: {str(e)}")
        
        # Check if we've reached our goal (only relevant for scrape mode)
        if OPERATION_MODE == "scrape":
            total_samples = get_total_samples_count()
            print(f"\nTotal samples collected: {total_samples}/{SAMPLE_GOAL}")
            
            if total_samples >= SAMPLE_GOAL:
                print(f"🎉 Goal achieved: {total_samples} samples collected out of {SAMPLE_GOAL} goal.")
                # Save learned patterns for future runs
                learning_system.save_patterns()
                print("Saved learned patterns to learned_patterns.json")
                break
        else:
            # For non-scrape modes, check if all tasks are completed
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', ("pending",))
                pending_count = cursor.fetchone()[0]
                
                if pending_count == 0:
                    print(f"🎉 All {OPERATION_MODE} tasks completed successfully!")
                    break
        
        # Sleep to avoid rate limiting
        time.sleep(5)


if __name__ == "__main__":
    # If we're in rank mode, make sure we reset the tasks
    if OPERATION_MODE == "rank":
        print("Initializing ranking mode...")
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM samples')
            samples_count = cursor.fetchone()[0]
            
            if samples_count > 0:
                print(f"Found {samples_count} samples. Resetting ranking tasks...")
                reset_tasks_for_mode()
            else:
                print("No samples found. Please run in 'scrape' mode first.")
    
    main()
