"""
Writer Agent - Day 9 Multi-Agent System
Specialized agent for creating high-quality content from research
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime

from agent_definitions import BaseAgent, WRITER_CONFIG, AgentStatus

load_dotenv()


class WriterAgent(BaseAgent):
    """
    Specialized Writer Agent
    
    Capabilities:
    - Transform research into engaging content
    - Multiple content formats (blog, article, report)
    - SEO optimization
    - Proper formatting and structure
    - Citation integration
    """
    
    def __init__(self, api_key: str):
        super().__init__(WRITER_CONFIG)
        
        # Initialize LLM
        self.client = Groq(api_key=api_key)
        self.model = self.config.model
        
        # Writing history
        self.drafts: List[Dict] = []
        self.current_content = ""
    
    def write_content(
        self,
        topic: str,
        research_data: Dict[str, Any],
        content_type: str = "blog_post",
        word_count: int = 1500,
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """
        Write content based on research
        
        Args:
            topic: Content topic
            research_data: Research findings from ResearchAgent
            content_type: Type of content (blog_post, article, report)
            word_count: Target word count
            tone: Writing tone (professional, casual, academic)
        
        Returns:
            Content dictionary with text, metadata
        """
        try:
            self.set_status(AgentStatus.WORKING)
            self.current_task = f"Write {content_type}: {topic}"
            
            print(f"\n{'='*60}")
            print(f"✍️ WRITER AGENT: Creating {content_type} on '{topic}'")
            print(f"   Target: {word_count} words | Tone: {tone}")
            print(f"{'='*60}\n")
            
            # Step 1: Analyze research data
            print("📝 Step 1: Analyzing research data...")
            summary = research_data.get('summary', '')
            facts = research_data.get('facts', [])
            sources = research_data.get('sources', [])
            
            # Step 2: Create outline
            print("📝 Step 2: Creating content outline...")
            outline = self._create_outline(topic, summary, facts, content_type)
            print(f"   Created {len(outline.get('sections', []))} sections")
            
            # Step 3: Write content
            print("📝 Step 3: Writing content...")
            content = self._write_from_outline(
                topic, outline, summary, facts, 
                content_type, word_count, tone
            )
            
            # Step 4: Add citations
            print("📝 Step 4: Adding citations...")
            content_with_citations = self._add_citations(content, sources)
            
            # Step 5: Format document
            print("📝 Step 5: Formatting document...")
            formatted_content = self._format_document(
                topic, content_with_citations, sources, content_type
            )
            
            # Create result
            result = {
                "topic": topic,
                "content_type": content_type,
                "word_count": len(formatted_content.split()),
                "tone": tone,
                "content": formatted_content,
                "outline": outline,
                "sources_count": len(sources),
                "created_at": datetime.now().isoformat()
            }
            
            self.drafts.append(result)
            self.current_content = formatted_content
            self.set_status(AgentStatus.COMPLETED)
            
            print(f"\n✅ Content created!")
            print(f"   - Word count: {result['word_count']} words")
            print(f"   - Sections: {len(outline.get('sections', []))}")
            print(f"   - Sources cited: {len(sources)}")
            
            return result
            
        except Exception as e:
            self.handle_error(e, "Writing task")
            return {"error": str(e)}
    
    def _create_outline(
        self, 
        topic: str, 
        summary: str, 
        facts: List[str],
        content_type: str
    ) -> Dict[str, Any]:
        """Create content outline using LLM"""
        
        prompt = f"""Create a detailed outline for a {content_type} about "{topic}".

Research Summary:
{summary}

Key Facts Available:
{chr(10).join([f"- {fact}" for fact in facts[:10]])}

Create an outline with:
1. Engaging title
2. Introduction hook
3. 4-5 main sections with subsections
4. Conclusion

Format as JSON with structure:
{{
    "title": "...",
    "introduction": "...",
    "sections": [
        {{"heading": "...", "key_points": ["...", "..."]}}
    ],
    "conclusion": "..."
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert content outliner. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=1000
            )
            
            outline_text = response.choices[0].message.content
            
            # Try to parse JSON
            import json
            import re
            
            # Extract JSON if wrapped in markdown
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', outline_text, re.DOTALL)
            if json_match:
                outline_text = json_match.group(1)
            
            outline = json.loads(outline_text)
            
            self.log_action("create_outline", outline)
            return outline
            
        except Exception as e:
            print(f"⚠️ Outline creation failed: {e}, using default outline")
            # Return default outline
            return {
                "title": f"Understanding {topic}",
                "introduction": f"An exploration of {topic}",
                "sections": [
                    {"heading": "Overview", "key_points": ["Introduction to topic"]},
                    {"heading": "Key Findings", "key_points": facts[:3]},
                    {"heading": "Analysis", "key_points": ["Detailed analysis"]},
                    {"heading": "Implications", "key_points": ["Future impact"]}
                ],
                "conclusion": "Summary of findings"
            }
    
    def _write_from_outline(
        self,
        topic: str,
        outline: Dict,
        summary: str,
        facts: List[str],
        content_type: str,
        word_count: int,
        tone: str
    ) -> str:
        """Write full content from outline"""
        
        # Prepare research context
        research_context = f"""
Research Summary:
{summary}

Key Facts:
{chr(10).join([f"- {fact}" for fact in facts[:15]])}
"""
        
        sections_text = "\n\n".join([
            f"## {section.get('heading', '')}\nKey points: {', '.join(section.get('key_points', []))}"
            for section in outline.get('sections', [])
        ])
        
        prompt = f"""Write a {content_type} about "{topic}" following this outline:

Title: {outline.get('title', topic)}

Sections to cover:
{sections_text}

Research Data:
{research_context}

Requirements:
- Target length: {word_count} words
- Tone: {tone}
- Include factual information from research
- Use clear headings and structure
- Write engaging, informative content
- Integrate facts naturally

Write the complete {content_type} now:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert {content_type} writer. Write in a {tone} tone."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=3000
            )
            
            content = response.choices[0].message.content
            self.log_action("write_content", {"word_count": len(content.split())})
            
            return content
            
        except Exception as e:
            print(f"⚠️ Content writing failed: {e}")
            return f"# {outline.get('title', topic)}\n\nContent generation failed."
    
    def _add_citations(self, content: str, sources: List[str]) -> str:
        """Add citations to content"""
        if not sources:
            return content
        
        # Simple citation addition - add references section
        citations_section = "\n\n---\n\n## References\n\n"
        for i, source in enumerate(sources[:10], 1):
            citations_section += f"{i}. {source}\n"
        
        return content + citations_section
    
    def _format_document(
        self,
        topic: str,
        content: str,
        sources: List[str],
        content_type: str
    ) -> str:
        """Format final document with metadata"""
        
        # Add metadata header
        header = f"""---
Title: {topic}
Type: {content_type}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: AI Content Writer
Sources: {len(sources)}
---

"""
        
        return header + content
    
    def revise_content(
        self,
        content: str,
        feedback: str,
        suggestions: List[str]
    ) -> str:
        """
        Revise content based on feedback
        
        Args:
            content: Original content
            feedback: Feedback from reviewer
            suggestions: List of specific suggestions
        
        Returns:
            Revised content
        """
        try:
            print(f"\n📝 REVISING CONTENT based on feedback...")
            print(f"   Feedback: {feedback[:100]}...")
            print(f"   Suggestions: {len(suggestions)}")
            
            suggestions_text = "\n".join([f"- {s}" for s in suggestions])
            
            prompt = f"""Revise this content based on the feedback and suggestions:

ORIGINAL CONTENT:
{content}

REVIEWER FEEDBACK:
{feedback}

SPECIFIC SUGGESTIONS:
{suggestions_text}

Please revise the content addressing all feedback and suggestions while maintaining quality and coherence.
Return the complete revised content:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert editor. Revise content based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=3000
            )
            
            revised = response.choices[0].message.content
            self.current_content = revised
            
            print(f"✅ Content revised!")
            
            return revised
            
        except Exception as e:
            print(f"❌ Revision failed: {e}")
            return content
    
    def export_content(self, filepath: str, content: str = None) -> None:
        """
        Export content to file
        
        Args:
            filepath: Path to save content
            content: Content to export (defaults to current_content)
        """
        content = content or self.current_content
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📄 Content exported to {filepath}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n✍️ WRITER AGENT - Testing\n")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        exit(1)
    
    # Create writer agent
    writer = WriterAgent(api_key)
    
    # Mock research data
    mock_research = {
        "topic": "Artificial Intelligence in Healthcare",
        "summary": """AI is revolutionizing healthcare through improved diagnostics, 
        personalized treatment plans, and operational efficiency. Machine learning models 
        can now detect diseases earlier than traditional methods, leading to better patient outcomes.""",
        "facts": [
            "AI can detect certain cancers with 95% accuracy",
            "Machine learning reduces diagnostic time by 40%",
            "AI-powered chatbots handle 60% of patient inquiries",
            "Predictive analytics reduce hospital readmissions by 30%"
        ],
        "sources": [
            "Smith, J. (2025). AI in Healthcare. Retrieved from https://example.com/ai-health",
            "Johnson, M. (2025). Machine Learning for Diagnosis. Retrieved from https://example.com/ml-diagnosis"
        ]
    }
    
    # Test content writing
    print("="*60)
    print("TEST: Writing Blog Post")
    print("="*60)
    
    result = writer.write_content(
        topic="AI in Healthcare: The Future is Now",
        research_data=mock_research,
        content_type="blog_post",
        word_count=800,
        tone="professional"
    )
    
    print(f"\n{'='*60}")
    print("WRITTEN CONTENT")
    print(f"{'='*60}\n")
    print(result.get('content', 'No content'))
    
    # Test export
    print("\n\n" + "="*60)
    print("TEST: Export Content")
    print("="*60 + "\n")
    
    os.makedirs("./outputs", exist_ok=True)
    writer.export_content("./outputs/blog_post.md")
    
    print("\n✅ All tests complete!")