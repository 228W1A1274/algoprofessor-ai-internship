"""
Reviewer Agent - Day 9 Multi-Agent System
Specialized agent for quality assurance and content review
"""

import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
import re

from agent_definitions import BaseAgent, REVIEWER_CONFIG, AgentStatus

load_dotenv()


class ReviewerAgent(BaseAgent):
    """
    Specialized Reviewer Agent
    
    Capabilities:
    - Content quality assessment
    - Factual accuracy verification
    - Grammar and style checking
    - Citation verification
    - Providing actionable feedback
    """
    
    def __init__(self, api_key: str):
        super().__init__(REVIEWER_CONFIG)
        
        # Initialize LLM
        self.client = Groq(api_key=api_key)
        self.model = self.config.model
        
        # Review history
        self.reviews: List[Dict] = []
        self.quality_standards = {
            "min_word_count": 500,
            "max_word_count": 3000,
            "min_sources": 2,
            "required_sections": ["introduction", "conclusion"],
            "grammar_threshold": 0.9  # 90% accuracy
        }
    
    def review_content(
        self,
        content: str,
        research_data: Dict[str, Any],
        content_type: str = "blog_post"
    ) -> Dict[str, Any]:
        """
        Comprehensive content review
        
        Args:
            content: Content to review
            research_data: Original research data
            content_type: Type of content
        
        Returns:
            Review results with feedback and approval status
        """
        try:
            self.set_status(AgentStatus.WORKING)
            self.current_task = f"Review {content_type}"
            
            print(f"\n{'='*60}")
            print(f"👁️ REVIEWER AGENT: Reviewing {content_type}")
            print(f"{'='*60}\n")
            
            # Run all checks
            print("📝 Step 1: Quality checks...")
            quality_check = self._quality_check(content)
            
            print("📝 Step 2: Factual verification...")
            factual_check = self._verify_facts(content, research_data)
            
            print("📝 Step 3: Structure analysis...")
            structure_check = self._check_structure(content, content_type)
            
            print("📝 Step 4: Citation verification...")
            citation_check = self._verify_citations(content, research_data)
            
            print("📝 Step 5: Grammar and style review...")
            style_check = self._check_style(content)
            
            print("📝 Step 6: Generating feedback...")
            feedback = self._generate_feedback(
                quality_check, factual_check, structure_check,
                citation_check, style_check
            )
            
            # Calculate overall score
            scores = [
                quality_check.get('score', 0),
                factual_check.get('score', 0),
                structure_check.get('score', 0),
                citation_check.get('score', 0),
                style_check.get('score', 0)
            ]
            overall_score = sum(scores) / len(scores)
            
            # Determine approval
            approved = overall_score >= 0.75  # 75% threshold
            
            # Compile review
            review_result = {
                "content_type": content_type,
                "overall_score": overall_score,
                "approved": approved,
                "checks": {
                    "quality": quality_check,
                    "factual": factual_check,
                    "structure": structure_check,
                    "citations": citation_check,
                    "style": style_check
                },
                "feedback": feedback,
                "suggestions": self._generate_suggestions(
                    quality_check, factual_check, structure_check,
                    citation_check, style_check
                ),
                "reviewed_at": datetime.now().isoformat()
            }
            
            self.reviews.append(review_result)
            self.set_status(AgentStatus.COMPLETED)
            
            status_emoji = "✅" if approved else "⚠️"
            print(f"\n{status_emoji} Review complete!")
            print(f"   Overall Score: {overall_score*100:.1f}%")
            print(f"   Status: {'APPROVED' if approved else 'NEEDS REVISION'}")
            print(f"   Suggestions: {len(review_result['suggestions'])}")
            
            return review_result
            
        except Exception as e:
            self.handle_error(e, "Review task")
            return {"error": str(e), "approved": False}
    
    def _quality_check(self, content: str) -> Dict[str, Any]:
        """Check basic quality metrics"""
        
        word_count = len(content.split())
        char_count = len(content)
        
        # Check word count
        word_count_ok = (
            self.quality_standards["min_word_count"] <= word_count <= 
            self.quality_standards["max_word_count"]
        )
        
        # Check for headings
        has_headings = bool(re.search(r'^#{1,6}\s+.+$', content, re.MULTILINE))
        
        # Check for paragraphs (multiple line breaks)
        paragraph_count = len(re.findall(r'\n\n', content))
        has_paragraphs = paragraph_count >= 3
        
        # Calculate score
        checks = [word_count_ok, has_headings, has_paragraphs]
        score = sum(checks) / len(checks)
        
        return {
            "score": score,
            "word_count": word_count,
            "word_count_ok": word_count_ok,
            "has_headings": has_headings,
            "has_paragraphs": has_paragraphs,
            "paragraph_count": paragraph_count
        }
    
    def _verify_facts(self, content: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify factual claims against research"""
        
        research_facts = research_data.get('facts', [])
        
        # Simple fact checking - see if research facts appear in content
        facts_used = 0
        for fact in research_facts:
            # Extract key terms from fact
            key_terms = [word for word in fact.split() if len(word) > 4]
            # Check if any key term appears in content
            if any(term.lower() in content.lower() for term in key_terms):
                facts_used += 1
        
        facts_percentage = facts_used / max(len(research_facts), 1)
        
        # Look for unsupported claims (numbers without context)
        potential_claims = re.findall(r'\d+%|\d+\s+(?:percent|million|billion|thousand)', content)
        
        return {
            "score": min(facts_percentage, 1.0),
            "facts_from_research": len(research_facts),
            "facts_used_in_content": facts_used,
            "usage_percentage": facts_percentage * 100,
            "potential_claims": len(potential_claims),
            "verification_needed": len(potential_claims) > facts_used
        }
    
    def _check_structure(self, content: str, content_type: str) -> Dict[str, Any]:
        """Check content structure"""
        
        # Find all headings
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        
        # Check for introduction
        intro_keywords = ['introduction', 'overview', 'background']
        has_intro = any(
            keyword in content[:500].lower() 
            for keyword in intro_keywords
        ) or len(headings) > 0
        
        # Check for conclusion
        conclusion_keywords = ['conclusion', 'summary', 'final', 'closing']
        has_conclusion = any(
            keyword in content[-500:].lower() 
            for keyword in conclusion_keywords
        ) or 'References' in content
        
        # Check for references section
        has_references = bool(re.search(r'#{1,6}\s*References', content, re.IGNORECASE))
        
        checks = [has_intro, has_conclusion, has_references, len(headings) >= 3]
        score = sum(checks) / len(checks)
        
        return {
            "score": score,
            "heading_count": len(headings),
            "headings": headings,
            "has_introduction": has_intro,
            "has_conclusion": has_conclusion,
            "has_references": has_references
        }
    
    def _verify_citations(self, content: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify citations are present and properly formatted"""
        
        sources = research_data.get('sources', [])
        
        # Check for references section
        has_references = bool(re.search(r'#{1,6}\s*References', content, re.IGNORECASE))
        
        # Count citation-like patterns [1], (Author, Year), etc.
        citation_patterns = re.findall(r'\[\d+\]|\([A-Z][a-z]+,\s*\d{4}\)', content)
        
        # Check if sources are listed
        sources_listed = 0
        for source in sources:
            if any(part in content for part in source.split()[:3]):
                sources_listed += 1
        
        score = 0.0
        if has_references:
            score += 0.5
        if sources_listed >= self.quality_standards["min_sources"]:
            score += 0.5
        
        return {
            "score": score,
            "has_references_section": has_references,
            "total_sources": len(sources),
            "sources_cited": sources_listed,
            "citation_marks": len(citation_patterns)
        }
    
    def _check_style(self, content: str) -> Dict[str, Any]:
        """Check grammar and style"""
        
        # Simple style checks
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        sentence_length_ok = 15 <= avg_sentence_length <= 25
        
        # Check for passive voice (simple detection)
        passive_indicators = re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', content)
        passive_ratio = len(passive_indicators) / max(len(sentences), 1)
        low_passive = passive_ratio < 0.3
        
        # Check for varied sentence starts
        first_words = [s.split()[0] for s in sentences if s.split()]
        unique_starts = len(set(first_words)) / max(len(first_words), 1)
        varied_starts = unique_starts > 0.5
        
        checks = [sentence_length_ok, low_passive, varied_starts]
        score = sum(checks) / len(checks)
        
        return {
            "score": score,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_ok": sentence_length_ok,
            "passive_voice_ratio": passive_ratio,
            "low_passive_voice": low_passive,
            "sentence_variety": unique_starts,
            "varied_starts": varied_starts
        }
    
    def _generate_feedback(self, *checks) -> str:
        """Generate comprehensive feedback using LLM"""
        
        checks_summary = "\n".join([
            f"- {check.get('score', 0)*100:.0f}% score: {list(check.keys())[0]} check"
            for check in checks if isinstance(check, dict)
        ])
        
        prompt = f"""Based on these content review checks:

{checks_summary}

Provide concise, constructive feedback (2-3 sentences) on:
1. Overall quality
2. Main strengths
3. Areas for improvement

Keep it professional and actionable."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional content reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return "Automated review complete. See detailed scores for areas of improvement."
    
    def _generate_suggestions(self, *checks) -> List[str]:
        """Generate specific suggestions for improvement"""
        
        suggestions = []
        
        quality, factual, structure, citations, style = checks
        
        # Quality suggestions
        if not quality.get('word_count_ok'):
            wc = quality.get('word_count', 0)
            if wc < self.quality_standards["min_word_count"]:
                suggestions.append(f"Expand content to at least {self.quality_standards['min_word_count']} words (currently {wc})")
            else:
                suggestions.append(f"Reduce content to under {self.quality_standards['max_word_count']} words (currently {wc})")
        
        if not quality.get('has_headings'):
            suggestions.append("Add section headings to improve readability")
        
        # Factual suggestions
        if factual.get('score', 0) < 0.5:
            suggestions.append("Include more facts from research to support claims")
        
        # Structure suggestions
        if not structure.get('has_introduction'):
            suggestions.append("Add a clear introduction section")
        
        if not structure.get('has_conclusion'):
            suggestions.append("Add a conclusion to summarize key points")
        
        if not structure.get('has_references'):
            suggestions.append("Add a References section with properly formatted citations")
        
        # Citation suggestions
        if citations.get('sources_cited', 0) < self.quality_standards["min_sources"]:
            suggestions.append(f"Cite at least {self.quality_standards['min_sources']} sources")
        
        # Style suggestions
        if not style.get('sentence_length_ok'):
            avg_len = style.get('avg_sentence_length', 0)
            if avg_len < 15:
                suggestions.append("Combine some short sentences for better flow")
            else:
                suggestions.append("Break up long sentences for better readability")
        
        if not style.get('low_passive_voice'):
            suggestions.append("Reduce passive voice usage for more engaging writing")
        
        return suggestions
    
    def quick_approve(self, content: str) -> bool:
        """
        Quick approval check without full review
        
        Args:
            content: Content to check
        
        Returns:
            True if meets basic standards
        """
        word_count = len(content.split())
        has_headings = bool(re.search(r'^#{1,6}\s+.+$', content, re.MULTILINE))
        has_references = 'References' in content or 'references' in content
        
        return (
            word_count >= self.quality_standards["min_word_count"] and
            has_headings and
            has_references
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n👁️ REVIEWER AGENT - Testing\n")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        exit(1)
    
    # Create reviewer
    reviewer = ReviewerAgent(api_key)
    
    # Mock content
    mock_content = """# AI in Healthcare: The Future is Now

## Introduction

Artificial intelligence is revolutionizing the healthcare industry. Machine learning models can now detect diseases with 95% accuracy, significantly improving patient outcomes.

## Current Applications

AI-powered diagnostic tools are being used in hospitals worldwide. These tools reduce diagnostic time by 40% and help doctors make more informed decisions.

## Benefits

The integration of AI in healthcare brings multiple benefits:
- Improved accuracy
- Faster diagnoses
- Better patient care

## Conclusion

AI is transforming healthcare for the better. As technology advances, we can expect even more improvements.

## References

1. Smith, J. (2025). AI in Healthcare. Retrieved from https://example.com
2. Johnson, M. (2025). Machine Learning for Diagnosis. Retrieved from https://example.com/ml
"""
    
    mock_research = {
        "facts": [
            "AI can detect certain cancers with 95% accuracy",
            "Machine learning reduces diagnostic time by 40%"
        ],
        "sources": [
            "Smith, J. (2025). AI in Healthcare.",
            "Johnson, M. (2025). Machine Learning for Diagnosis."
        ]
    }
    
    # Test review
    print("="*60)
    print("TEST: Content Review")
    print("="*60)
    
    review = reviewer.review_content(mock_content, mock_research, "blog_post")
    
    print(f"\n{'='*60}")
    print("REVIEW RESULTS")
    print(f"{'='*60}\n")
    print(f"Overall Score: {review['overall_score']*100:.1f}%")
    print(f"Status: {'✅ APPROVED' if review['approved'] else '⚠️ NEEDS REVISION'}")
    print(f"\nFeedback:\n{review['feedback']}")
    print(f"\nSuggestions ({len(review['suggestions'])}):")
    for i, suggestion in enumerate(review['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    print("\n✅ All tests complete!")
