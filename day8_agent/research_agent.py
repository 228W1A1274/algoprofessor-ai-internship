"""
Autonomous Research Agent
Day 8 - Agent that can autonomously research hospital information
"""

import os
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from tools.custom_tools import hospital_tools
import json

load_dotenv()


class HospitalResearchAgent:
    """
    Autonomous agent that can research and compile comprehensive reports
    about hospital operations
    """
    
    def __init__(self):
        """Initialize research agent"""
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        self.tools = hospital_tools
        self.agent_executor = self._create_agent()
    
    def _create_agent(self):
        """Create the research agent with tools"""
        
        tools = [
            Tool(
                name="SearchHospitalData",
                func=self.tools.rag_search_tool,
                description="Search all hospital data including doctors, patients, departments, appointments, procedures"
            ),
            Tool(
                name="AnalyzeStatistics",
                func=self.tools.data_analysis_tool,
                description="Analyze hospital statistics for occupancy, costs, or doctor information"
            ),
            Tool(
                name="Calculate",
                func=self.tools.calculator_tool,
                description="Perform mathematical calculations"
            )
        ]
        
        template = """You are an autonomous research agent for a hospital. 
        
Your job is to gather comprehensive information and create detailed reports.

You have access to these tools:
{tools}

Research Process:
1. Understand the research question
2. Plan what data you need to gather
3. Use tools to collect information
4. Analyze the data
5. Compile a comprehensive answer

Use this format:

Question: {input}
Thought: {agent_scratchpad}
Action: [tool name]
Action Input: [input for tool]
Observation: [result from tool]
... (repeat as needed)
Thought: I have gathered all necessary information
Final Answer: [comprehensive research report]

Begin your research!"""
        
        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def research(self, topic: str) -> str:
        """
        Conduct autonomous research on a topic
        
        Args:
            topic: Research topic or question
            
        Returns:
            Comprehensive research report
        """
        result = self.agent_executor.invoke({"input": topic})
        return result['output']
    
    def generate_hospital_report(self) -> dict:
        """
        Generate a comprehensive hospital operations report
        
        Returns:
            Dictionary with various reports
        """
        print("\n🔬 GENERATING COMPREHENSIVE HOSPITAL REPORT...")
        print("="*60 + "\n")
        
        reports = {}
        
        # Report 1: Doctor Overview
        print("📋 Researching Doctor Information...")
        reports['doctors'] = self.research(
            "Provide a comprehensive overview of all doctors including specializations, "
            "experience levels, and consultation fees"
        )
        
        # Report 2: Department Analysis
        print("\n📋 Analyzing Department Operations...")
        reports['departments'] = self.research(
            "Analyze all departments, their bed capacity, current occupancy rates, "
            "and identify which departments are over/under capacity"
        )
        
        # Report 3: Financial Analysis
        print("\n📋 Conducting Financial Analysis...")
        reports['financials'] = self.research(
            "Analyze medical procedure costs, calculate total potential revenue, "
            "and provide cost breakdown by department"
        )
        
        # Report 4: Patient Care Summary
        print("\n📋 Summarizing Patient Care...")
        reports['patients'] = self.research(
            "Summarize current patients, their diagnoses, attending physicians, "
            "and treatment status"
        )
        
        return reports
    
    def save_report(self, reports: dict, filename: str = "outputs/hospital_report.json"):
        """Save research reports to file"""
        os.makedirs("outputs", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(reports, f, indent=2)
        
        # Also create a text version
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w') as f:
            f.write("COMPREHENSIVE HOSPITAL OPERATIONS REPORT\n")
            f.write("="*60 + "\n\n")
            
            for section, content in reports.items():
                f.write(f"\n{section.upper()}\n")
                f.write("-"*60 + "\n")
                f.write(content + "\n\n")
        
        print(f"\n✅ Reports saved to:")
        print(f"   - {filename}")
        print(f"   - {text_filename}")


def main():
    """Test the research agent"""
    
    print("\n" + "="*60)
    print("AUTONOMOUS HOSPITAL RESEARCH AGENT")
    print("="*60 + "\n")
    
    # Create research agent
    agent = HospitalResearchAgent()
    
    # Test individual research
    print("\n--- Individual Research Test ---\n")
    
    query = "What is the current financial status of the hospital based on procedure costs and doctor fees?"
    print(f"Research Question: {query}\n")
    
    result = agent.research(query)
    print(f"\n📊 RESEARCH RESULT:\n{result}\n")
    
    # Generate comprehensive report
    print("\n\n--- Comprehensive Report Generation ---\n")
    user_input = input("Generate full hospital report? (y/n): ")
    
    if user_input.lower() == 'y':
        reports = agent.generate_hospital_report()
        agent.save_report(reports)
        
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    main()