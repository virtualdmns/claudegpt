"""
Visualization tools for the ClaudeGPT system
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

class InteractionVisualizer:
    """Visualize interactions and performance of the ClaudeGPT system"""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualizer"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_task_execution_times(self, performance_data: Dict[str, Any], save: bool = True):
        """Plot task execution times"""
        task_times = performance_data.get("task_execution_times", [])
        if not task_times:
            print("No task execution data available")
            return
        
        # Extract data
        task_ids = [item["task_id"] for item in task_times]
        durations = [item["duration"] for item in task_times]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(task_ids, durations)
        plt.xlabel("Task ID")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Task Execution Times")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"task_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename)
            print(f"Saved task execution times plot to {filename}")
        else:
            plt.show()
    
    def plot_api_call_distribution(self, performance_data: Dict[str, Any], save: bool = True):
        """Plot distribution of API calls between Claude and GPT"""
        api_calls = performance_data.get("api_call_times", {})
        if not api_calls:
            print("No API call data available")
            return
        
        # Extract data
        claude_calls = len(api_calls.get("claude", []))
        gpt_calls = len(api_calls.get("gpt", []))
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.pie([claude_calls, gpt_calls], labels=["Claude", "GPT"], autopct='%1.1f%%')
        plt.title("Distribution of API Calls")
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"api_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename)
            print(f"Saved API call distribution plot to {filename}")
        else:
            plt.show()
    
    def plot_token_usage(self, performance_data: Dict[str, Any], save: bool = True):
        """Plot token usage by model"""
        token_data = performance_data.get("total_tokens", {})
        if not token_data:
            print("No token usage data available")
            return
        
        # Extract data
        models = list(token_data.keys())
        input_tokens = [token_data[model].get("input", 0) for model in models]
        output_tokens = [token_data[model].get("output", 0) for model in models]
        
        # Create plot
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, input_tokens, width, label='Input Tokens')
        ax.bar(x + width/2, output_tokens, width, label='Output Tokens')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Token Count')
        ax.set_title('Token Usage by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"token_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename)
            print(f"Saved token usage plot to {filename}")
        else:
            plt.show()
    
    def plot_tool_usage(self, performance_data: Dict[str, Any], save: bool = True):
        """Plot tool usage statistics"""
        tool_times = performance_data.get("tool_execution_times", {})
        if not tool_times:
            print("No tool execution data available")
            return
        
        # Extract data
        tools = list(tool_times.keys())
        counts = [len(tool_times[tool]) for tool in tools]
        avg_times = [sum(item["duration"] for item in tool_times[tool]) / len(tool_times[tool]) for tool in tools]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Tool usage count
        ax1.bar(tools, counts)
        ax1.set_xlabel('Tool')
        ax1.set_ylabel('Number of Executions')
        ax1.set_title('Tool Usage Count')
        ax1.set_xticklabels(tools, rotation=45)
        
        # Average execution time
        ax2.bar(tools, avg_times)
        ax2.set_xlabel('Tool')
        ax2.set_ylabel('Average Execution Time (seconds)')
        ax2.set_title('Average Tool Execution Time')
        ax2.set_xticklabels(tools, rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"tool_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename)
            print(f"Saved tool usage plot to {filename}")
        else:
            plt.show()
    
    def generate_interaction_graph(self, interactions: List[Dict[str, Any]], save: bool = True):
        """Generate a graph of agent interactions"""
        if not interactions:
            print("No interaction data available")
            return
        
        try:
            import networkx as nx
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for interaction in interactions:
                source = interaction.get("source", "unknown")
                target = interaction.get("target", "unknown")
                message_type = interaction.get("type", "message")
                
                G.add_node(source)
                G.add_node(target)
                G.add_edge(source, target, type=message_type)
            
            # Plot graph
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, 
                                  node_color=['skyblue' if node == 'Claude' else 'lightgreen' if node == 'GPT' else 'lightgray' for node in G.nodes()],
                                  node_size=1000)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos)
            
            plt.title("Agent Interaction Graph")
            plt.axis('off')
            
            if save:
                filename = os.path.join(self.output_dir, f"interaction_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(filename)
                print(f"Saved interaction graph to {filename}")
            else:
                plt.show()
                
        except ImportError:
            print("NetworkX library is required for interaction graphs. Install with: pip install networkx")
    
    def create_dashboard(self, performance_data: Dict[str, Any], save: bool = True):
        """Create a comprehensive dashboard of system performance"""
        if not performance_data:
            print("No performance data available")
            return
        
        # Create a multi-panel figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("ClaudeGPT System Performance Dashboard", fontsize=16)
        
        # Task execution times
        if "task_execution_times" in performance_data and performance_data["task_execution_times"]:
            ax1 = fig.add_subplot(2, 2, 1)
            task_times = performance_data["task_execution_times"]
            task_ids = [item["task_id"] for item in task_times]
            durations = [item["duration"] for item in task_times]
            
            ax1.bar(task_ids, durations)
            ax1.set_xlabel("Task ID")
            ax1.set_ylabel("Execution Time (seconds)")
            ax1.set_title("Task Execution Times")
            ax1.set_xticklabels(task_ids, rotation=45)
        
        # API call distribution
        if "api_call_times" in performance_data:
            ax2 = fig.add_subplot(2, 2, 2)
            api_calls = performance_data["api_call_times"]
            claude_calls = len(api_calls.get("claude", []))
            gpt_calls = len(api_calls.get("gpt", []))
            
            ax2.pie([claude_calls, gpt_calls], labels=["Claude", "GPT"], autopct='%1.1f%%')
            ax2.set_title("Distribution of API Calls")
        
        # Token usage
        if "total_tokens" in performance_data:
            ax3 = fig.add_subplot(2, 2, 3)
            token_data = performance_data["total_tokens"]
            models = list(token_data.keys())
            input_tokens = [token_data[model].get("input", 0) for model in models]
            output_tokens = [token_data[model].get("output", 0) for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax3.bar(x - width/2, input_tokens, width, label='Input Tokens')
            ax3.bar(x + width/2, output_tokens, width, label='Output Tokens')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Token Count')
            ax3.set_title('Token Usage by Model')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models)
            ax3.legend()
        
        # Tool usage
        if "tool_execution_times" in performance_data and performance_data["tool_execution_times"]:
            ax4 = fig.add_subplot(2, 2, 4)
            tool_times = performance_data["tool_execution_times"]
            tools = list(tool_times.keys())
            counts = [len(tool_times[tool]) for tool in tools]
            
            ax4.bar(tools, counts)
            ax4.set_xlabel('Tool')
            ax4.set_ylabel('Number of Executions')
            ax4.set_title('Tool Usage Count')
            ax4.set_xticklabels(tools, rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        if save:
            filename = os.path.join(self.output_dir, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename)
            print(f"Saved performance dashboard to {filename}")
        else:
            plt.show()
    
    def export_html_report(self, performance_data: Dict[str, Any], interactions: List[Dict[str, Any]], output_file: str = None):
        """Export an HTML report of system performance and interactions"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Generate plots and save them
        self.plot_task_execution_times(performance_data, save=True)
        self.plot_api_call_distribution(performance_data, save=True)
        self.plot_token_usage(performance_data, save=True)
        self.plot_tool_usage(performance_data, save=True)
        
        if interactions:
            self.generate_interaction_graph(interactions, save=True)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ClaudeGPT System Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-card {{ 
                    background-color: #f5f5f5; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px; 
                    min-width: 200px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .visualization {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>ClaudeGPT System Performance Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                <div class="metrics">
        """
        
        # Add summary metrics
        summary = {
            "Total Tasks": len(performance_data.get("task_execution_times", [])),
            "Avg Task Time": f"{performance_data.get('avg_task_time', 0):.2f}s",
            "Total Claude Calls": len(performance_data.get("api_call_times", {}).get("claude", [])),
            "Total GPT Calls": len(performance_data.get("api_call_times", {}).get("gpt", [])),
            "Total Input Tokens": sum(data.get("input", 0) for data in performance_data.get("total_tokens", {}).values()),
            "Total Output Tokens": sum(data.get("output", 0) for data in performance_data.get("total_tokens", {}).values()),
            "Unique Tools Used": len(performance_data.get("tool_execution_times", {}))
        }
        
        for label, value in summary.items():
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Add visualizations
        for img_file in os.listdir(self.output_dir):
            if img_file.endswith(".png") and datetime.now().strftime('%Y%m%d') in img_file:
                img_path = os.path.join(self.output_dir, img_file)
                title = img_file.split('_')[0].replace('_', ' ').title()
                html_content += f"""
                <div class="visualization">
                    <h3>{title}</h3>
                    <img src="{img_path}" alt="{title}" style="max-width: 100%;">
                </div>
                """
        
        # Add task details if available
        if performance_data.get("task_execution_times"):
            html_content += """
            <div class="section">
                <h2>Task Details</h2>
                <table>
                    <tr>
                        <th>Task ID</th>
                        <th>Duration (s)</th>
                        <th>Timestamp</th>
                    </tr>
            """
            
            for task in performance_data["task_execution_times"]:
                timestamp = datetime.fromtimestamp(task["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                html_content += f"""
                    <tr>
                        <td>{task["task_id"]}</td>
                        <td>{task["duration"]:.2f}</td>
                        <td>{timestamp}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report exported to {output_file}")
        return output_file