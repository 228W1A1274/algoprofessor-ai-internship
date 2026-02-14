"""
ReAct Agent Implementation - FINAL WORKING VERSION
Day 8 - Understanding Agent Architecture
"""

import os
from dotenv import load_dotenv
from groq import Groq
import json
from tools.custom_tools import hospital_tools

# Load environment variables
load_dotenv()

class ReActAgent:
    """
    Simple ReAct (Reasoning + Acting) Agent Implementation
    
    This agent follows the pattern:
    1. THOUGHT - Reason about what to do
    2. ACTION - Execute a tool
    3. OBSERVATION - See the result
    4. Repeat until task is complete
    """
    
    def __init__(self, api_key: str):
        """Initialize agent with Groq API"""
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.tools = hospital_tools
        self.max_iterations = 5
        
        # Define available tools
        self.available_tools = {
            "rag_search": {
                "function": self.tools.rag_search_tool,
                "description": "Search hospital database for information about doctors, patients, departments, procedures"
            },
            "calculator": {
                "function": self.tools.calculator_tool,
                "description": "Calculate mathematical expressions like costs, totals, percentages"
            },
            "analyze_data": {
                "function": self.tools.data_analysis_tool,
                "description": "Analyze hospital data (demographics, conditions, costs, departments)"
            },
            "patient_statistics": {
                "function": self.tools.patient_statistics_tool,
                "description": "Get patient statistics and summaries"
            },
            "risk_assessment": {
                "function": self.tools.risk_assessment_tool,
                "description": "Assess patient cardiovascular risk levels"
            }
        }
    
    def think(self, query: str, context: str = "") -> dict:
        """Agent thinks about what to do next"""
        
        tools_description = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.available_tools.items()
        ])
        
        system_prompt = f"""You are a helpful hospital AI assistant that uses tools to answer questions.

Available Tools:
{tools_description}

You must respond in this EXACT format:

THOUGHT: [Your reasoning about what to do]
ACTION: [tool_name]
ACTION_INPUT: [input for the tool]

OR if you have enough information:

THOUGHT: [Your reasoning]
FINAL_ANSWER: [Complete answer to the user]

Example:
User: "What is the total cost of ECG and consultation with Dr. Sarah Johnson?"
THOUGHT: I need to find the costs of ECG procedure and Dr. Sarah Johnson's consultation fee
ACTION: rag_search
ACTION_INPUT: ECG cost and Dr. Sarah Johnson consultation fee

After getting results:
THOUGHT: I found ECG costs $50 and consultation is $200, now I need to calculate total
ACTION: calculator
ACTION_INPUT: 50 + 200

After calculation:
THOUGHT: I have the total cost now
FINAL_ANSWER: The total cost is $250 (ECG: $50 + Consultation: $200)"""

        user_message = f"{context}\n\nUser Question: {query}\n\nRespond with THOUGHT, ACTION, ACTION_INPUT or FINAL_ANSWER:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return self._parse_response(answer)
            
        except Exception as e:
            return {
                "error": str(e),
                "final_answer": f"Error: {str(e)}"
            }
    
    def _parse_response(self, response: str) -> dict:
        """Parse agent's response into structured format"""
        result = {}
        
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('THOUGHT:'):
                result['thought'] = line.replace('THOUGHT:', '').strip()
            elif line.startswith('ACTION:'):
                result['action'] = line.replace('ACTION:', '').strip()
            elif line.startswith('ACTION_INPUT:'):
                result['action_input'] = line.replace('ACTION_INPUT:', '').strip()
            elif line.startswith('FINAL_ANSWER:'):
                result['final_answer'] = line.replace('FINAL_ANSWER:', '').strip()
                # Get rest of the answer if multiline
                idx = lines.index(line)
                if idx < len(lines) - 1:
                    result['final_answer'] += ' ' + ' '.join(lines[idx+1:])
        
        return result
    
    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result"""
        if tool_name in self.available_tools:
            try:
                tool_function = self.available_tools[tool_name]["function"]
                result = tool_function(tool_input)
                return result
            except Exception as e:
                return f"Tool execution error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
    
    def run(self, query: str, verbose: bool = True) -> str:
        """Run the ReAct loop"""
        context = ""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"USER QUERY: {query}")
            print(f"{'='*60}\n")
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Agent thinks
            decision = self.think(query, context)
            
            if 'error' in decision:
                return decision['final_answer']
            
            # Print thought
            if 'thought' in decision and verbose:
                print(f"💭 THOUGHT: {decision['thought']}")
            
            # Check if final answer
            if 'final_answer' in decision:
                if verbose:
                    print(f"\n✅ FINAL ANSWER: {decision['final_answer']}")
                return decision['final_answer']
            
            # Execute action
            if 'action' in decision and 'action_input' in decision:
                if verbose:
                    print(f"🔧 ACTION: {decision['action']}")
                    print(f"📝 INPUT: {decision['action_input']}")
                
                observation = self.execute_tool(decision['action'], decision['action_input'])
                
                if verbose:
                    print(f"👁️ OBSERVATION: {observation[:200]}...")
                
                # Add to context
                context += f"\nObservation from {decision['action']}: {observation}\n"
            else:
                if verbose:
                    print("⚠️ No valid action found, ending loop")
                break
        
        return "Maximum iterations reached. Please try rephrasing your question."


def main():
    """Test the ReAct agent"""
    
    # Initialize agent
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        print("\n💡 Create a .env file with:")
        print("GROQ_API_KEY=your_api_key_here")
        return
    
    agent = ReActAgent(api_key)
    
    print("\n" + "="*60)
    print("HOSPITAL AI AGENT - ReAct Implementation")
    print("="*60)
    
    # Test queries
    test_queries = [
        "What is the average age of patients in the hospital?",
        "How many patients have hypertension?",
        "What is the total cost if I get an ECG Test and consult Dr. Sarah Johnson?",
        "Show me cardiovascular risk assessment statistics"
    ]
    
    for query in test_queries:
        print(f"\n\n{'#'*60}")
        answer = agent.run(query, verbose=True)
        print(f"\n{'#'*60}\n")
        
        user_input = input("Press Enter for next query (or 'q' to quit)...")
        if user_input.lower() == 'q':
            break


if __name__ == "__main__":
    main()