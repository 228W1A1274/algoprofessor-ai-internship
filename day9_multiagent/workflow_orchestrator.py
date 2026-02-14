"""
Workflow Orchestrator - Day 9 Multi-Agent System
Coordinates agent activities and manages the complete workflow
"""

import os
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from agent_definitions import BaseAgent, ORCHESTRATOR_CONFIG, AgentStatus, AgentRole
from communication_protocol import MessageBus, Message, MessageType, MessagePriority, create_task_message, create_data_message, create_feedback_message
from research_agent import ResearchAgent
from writer_agent import WriterAgent
from reviewer_agent import ReviewerAgent

load_dotenv()


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowType(Enum):
    """Types of workflows"""
    SEQUENTIAL = "sequential"  # One agent after another
    PARALLEL = "parallel"      # Multiple agents simultaneously
    ITERATIVE = "iterative"    # With feedback loops


class WorkflowOrchestrator(BaseAgent):
    """
    Orchestrator Agent - Manages the entire multi-agent workflow
    
    Responsibilities:
    - Receive user requests
    - Plan workflow
    - Assign tasks to agents
    - Monitor progress
    - Handle failures
    - Compile final output
    """
    
    def __init__(self, api_key: str):
        super().__init__(ORCHESTRATOR_CONFIG)
        
        # Initialize message bus
        self.message_bus = MessageBus()
        
        # Initialize agents
        print("🚀 Initializing agents...")
        self.researcher = ResearchAgent(api_key)
        self.writer = WriterAgent(api_key)
        self.reviewer = ReviewerAgent(api_key)
        
        # Workflow state
        self.workflow_status = WorkflowStatus.PENDING
        self.current_workflow: Optional[Dict] = None
        self.workflow_history: List[Dict] = []
        
        # Configuration
        self.max_revision_cycles = 3
        self.timeout_seconds = 600  # 10 minutes
        
        print("✅ Orchestrator initialized with 3 agents")
    
    def create_content_workflow(
        self,
        topic: str,
        content_type: str = "blog_post",
        word_count: int = 1500,
        tone: str = "professional",
        research_depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Execute complete content creation workflow
        
        Workflow:
        1. Research → Gather information
        2. Write → Create content
        3. Review → Quality check
        4. Revise (if needed) → Improve content
        5. Final Approval
        
        Args:
            topic: Content topic
            content_type: Type of content to create
            word_count: Target word count
            tone: Writing tone
            research_depth: How deep to research (quick, standard, deep)
        
        Returns:
            Final content and metadata
        """
        try:
            self.set_status(AgentStatus.WORKING)
            self.workflow_status = WorkflowStatus.RUNNING
            
            workflow_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            print(f"\n{'='*60}")
            print(f"🎯 WORKFLOW ORCHESTRATOR: Starting Content Creation")
            print(f"{'='*60}")
            print(f"Workflow ID: {workflow_id}")
            print(f"Topic: {topic}")
            print(f"Type: {content_type}")
            print(f"Target: {word_count} words")
            print(f"{'='*60}\n")
            
            # Initialize workflow data
            workflow_data = {
                "workflow_id": workflow_id,
                "topic": topic,
                "content_type": content_type,
                "word_count": word_count,
                "tone": tone,
                "research_depth": research_depth,
                "started_at": datetime.now().isoformat(),
                "status": "running",
                "stages": {}
            }
            
            # =================================================================
            # STAGE 1: RESEARCH
            # =================================================================
            print(f"\n{'🔬 STAGE 1: RESEARCH':-^60}\n")
            
            research_start = datetime.now()
            research_result = self._execute_research(topic, research_depth)
            research_time = (datetime.now() - research_start).total_seconds()
            
            if "error" in research_result:
                return self._handle_workflow_failure(
                    "Research stage failed",
                    workflow_data,
                    research_result["error"]
                )
            
            workflow_data["stages"]["research"] = {
                "status": "completed",
                "time_seconds": research_time,
                "sources_found": research_result.get("sources_found",len(research_result.get("sources", [])))
            }
            
            # =================================================================
            # STAGE 2: WRITING
            # =================================================================
            print(f"\n{'✍️ STAGE 2: WRITING':-^60}\n")
            
            writing_start = datetime.now()
            writing_result = self._execute_writing(
                topic, research_result, content_type, word_count, tone
            )
            writing_time = (datetime.now() - writing_start).total_seconds()
            
            if "error" in writing_result:
                return self._handle_workflow_failure(
                    "Writing stage failed",
                    workflow_data,
                    writing_result["error"]
                )
            
            workflow_data["stages"]["writing"] = {
                "status": "completed",
                "time_seconds": writing_time,
                "word_count": writing_result.get("word_count", 0)
            }
            
            # =================================================================
            # STAGE 3: REVIEW & REVISION LOOP
            # =================================================================
            print(f"\n{'👁️ STAGE 3: REVIEW & REVISION':-^60}\n")
            
            revision_cycle = 0
            approved = False
            current_content = writing_result["content"]
            
            while not approved and revision_cycle < self.max_revision_cycles:
                print(f"\n📋 Review Cycle {revision_cycle + 1}/{self.max_revision_cycles}")
                
                review_start = datetime.now()
                review_result = self._execute_review(
                    current_content, research_result, content_type
                )
                review_time = (datetime.now() - review_start).total_seconds()
                
                if "error" in review_result:
                    print(f"⚠️ Review failed: {review_result['error']}")
                    break
                
                approved = review_result.get("approved", False)
                
                if approved:
                    print("✅ Content APPROVED!")
                    workflow_data["stages"][f"review_{revision_cycle+1}"] = {
                        "status": "approved",
                        "time_seconds": review_time,
                        "score": review_result.get("overall_score", 0)
                    }
                    break
                else:
                    print(f"⚠️ Content needs revision (Score: {review_result.get('overall_score', 0)*100:.1f}%)")
                    
                    workflow_data["stages"][f"review_{revision_cycle+1}"] = {
                        "status": "revision_needed",
                        "time_seconds": review_time,
                        "score": review_result.get("overall_score", 0),
                        "suggestions": len(review_result.get("suggestions", []))
                    }
                    
                    # Revise content
                    if revision_cycle < self.max_revision_cycles - 1:
                        print(f"\n📝 Revision {revision_cycle + 1}...")
                        revision_start = datetime.now()
                        
                        current_content = self._execute_revision(
                            current_content,
                            review_result.get("feedback", "Please improve quality"),
                            review_result.get("suggestions", [])
                        )
                        
                        revision_time = (datetime.now() - revision_start).total_seconds()
                        workflow_data["stages"][f"revision_{revision_cycle+1}"] = {
                            "status": "completed",
                            "time_seconds": revision_time
                        }
                
                revision_cycle += 1
            
            # =================================================================
            # FINAL COMPILATION
            # =================================================================
            print(f"\n{'📦 FINAL COMPILATION':-^60}\n")
            
            total_time = sum(
                stage.get("time_seconds", 0) 
                for stage in workflow_data["stages"].values()
            )
            
            final_output = {
                "workflow_id": workflow_id,
                "status": "completed" if approved else "completed_with_warnings",
                "topic": topic,
                "content_type": content_type,
                "content": current_content,
                "metadata": {
                    "word_count": len(current_content.split()),
                    "tone": tone,
                    "sources": research_result.get("sources", []),
                    "sources_count": len(research_result.get("sources", [])),
                    "research_depth": research_depth,
                    "revision_cycles": revision_cycle,
                    "final_approved": approved,
                    "final_score": review_result.get("overall_score", 0) if review_result else 0
                },
                "workflow_data": workflow_data,
                "timing": {
                    "total_seconds": total_time,
                    "total_minutes": total_time / 60,
                    "stages": {
                        name: stage.get("time_seconds", 0)
                        for name, stage in workflow_data["stages"].items()
                    }
                },
                "completed_at": datetime.now().isoformat()
            }
            
            self.workflow_history.append(final_output)
            self.workflow_status = WorkflowStatus.COMPLETED
            self.set_status(AgentStatus.COMPLETED)
            
            self._print_workflow_summary(final_output)
            
            return final_output
            
        except Exception as e:
            self.handle_error(e, "Workflow execution")
            return self._handle_workflow_failure("Unexpected error", {}, str(e))
    
    def _execute_research(self, topic: str, depth: str) -> Dict[str, Any]:
        """Execute research stage"""
        try:
            # Send task to researcher
            task_msg = create_task_message(
                from_agent=self.config.name,
                to_agent=self.researcher.config.name,
                task_description=f"Research topic: {topic}",
                task_data={"depth": depth}
            )
            self.message_bus.send(task_msg)
            
            # Execute research
            result = self.researcher.research(topic, depth)
            
            # Send completion message
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_writing(
        self,
        topic: str,
        research_data: Dict,
        content_type: str,
        word_count: int,
        tone: str
    ) -> Dict[str, Any]:
        """Execute writing stage"""
        try:
            # Send research data to writer
            data_msg = create_data_message(
                from_agent=self.researcher.config.name,
                to_agent=self.writer.config.name,
                data=research_data,
                description="Research findings"
            )
            self.message_bus.send(data_msg)
            
            # Execute writing
            result = self.writer.write_content(
                topic=topic,
                research_data=research_data,
                content_type=content_type,
                word_count=word_count,
                tone=tone
            )
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_review(
        self,
        content: str,
        research_data: Dict,
        content_type: str
    ) -> Dict[str, Any]:
        """Execute review stage"""
        try:
            # Send content to reviewer
            data_msg = create_data_message(
                from_agent=self.writer.config.name,
                to_agent=self.reviewer.config.name,
                data={"content": content},
                description="Content for review"
            )
            self.message_bus.send(data_msg)
            
            # Execute review
            result = self.reviewer.review_content(
                content=content,
                research_data=research_data,
                content_type=content_type
            )
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_revision(
        self,
        content: str,
        feedback: str,
        suggestions: List[str]
    ) -> str:
        """Execute revision stage"""
        try:
            # Send feedback to writer
            feedback_msg = create_feedback_message(
                from_agent=self.reviewer.config.name,
                to_agent=self.writer.config.name,
                feedback=feedback,
                suggestions=suggestions
            )
            self.message_bus.send(feedback_msg)
            
            # Execute revision
            revised_content = self.writer.revise_content(
                content=content,
                feedback=feedback,
                suggestions=suggestions
            )
            
            return revised_content
            
        except Exception as e:
            print(f"⚠️ Revision failed: {e}")
            return content  # Return original if revision fails
    
    def _handle_workflow_failure(
        self,
        reason: str,
        workflow_data: Dict,
        error: str
    ) -> Dict[str, Any]:
        """Handle workflow failure"""
        self.workflow_status = WorkflowStatus.FAILED
        self.set_status(AgentStatus.FAILED)
        
        print(f"\n❌ WORKFLOW FAILED: {reason}")
        print(f"   Error: {error}")
        
        return {
            "status": "failed",
            "reason": reason,
            "error": error,
            "workflow_data": workflow_data,
            "failed_at": datetime.now().isoformat()
        }
    
    def _print_workflow_summary(self, result: Dict) -> None:
        """Print workflow execution summary"""
        print(f"\n{'='*60}")
        print(f"✅ WORKFLOW COMPLETED")
        print(f"{'='*60}")
        print(f"Status: {result['status'].upper()}")
        print(f"Topic: {result['topic']}")
        print(f"Content Type: {result['content_type']}")
        print(f"\nMetrics:")
        print(f"  - Word Count: {result['metadata']['word_count']}")
        print(f"  - Sources Used: {result['metadata']['sources_count']}")
        print(f"  - Revision Cycles: {result['metadata']['revision_cycles']}")
        print(f"  - Final Score: {result['metadata']['final_score']*100:.1f}%")
        print(f"  - Approved: {'✅ Yes' if result['metadata']['final_approved'] else '⚠️ No'}")
        print(f"\nTiming:")
        print(f"  - Total Time: {result['timing']['total_minutes']:.2f} minutes")
        for stage, time in result['timing']['stages'].items():
            print(f"  - {stage}: {time:.1f}s")
        print(f"{'='*60}\n")
    
    def export_workflow(self, result: Dict, base_path: str = "./outputs") -> Dict[str, str]:
        """
        Export workflow results to files
        
        Args:
            result: Workflow result dictionary
            base_path: Base directory for outputs
        
        Returns:
            Dictionary of created file paths
        """
        import json
        
        os.makedirs(base_path, exist_ok=True)
        
        workflow_id = result["workflow_id"]
        
        # Save content
        content_path = f"{base_path}/{workflow_id}_content.md"
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(result["content"])
        
        # Save metadata
        metadata_path = f"{base_path}/{workflow_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "workflow_id": result["workflow_id"],
                "status": result["status"],
                "metadata": result["metadata"],
                "timing": result["timing"],
                "completed_at": result["completed_at"]
            }, f, indent=2)
        
        # Save full workflow data
        workflow_path = f"{base_path}/{workflow_id}_workflow.json"
        with open(workflow_path, 'w') as f:
            json.dump(result["workflow_data"], f, indent=2)
        
        # Export message history
        message_path = f"{base_path}/{workflow_id}_messages.json"
        self.message_bus.export_history(message_path)
        
        print(f"\n📁 Workflow exported:")
        print(f"  - Content: {content_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Workflow: {workflow_path}")
        print(f"  - Messages: {message_path}")
        
        return {
            "content": content_path,
            "metadata": metadata_path,
            "workflow": workflow_path,
            "messages": message_path
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n🎯 WORKFLOW ORCHESTRATOR - Testing\n")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found")
        exit(1)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(api_key)
    
    # Run workflow
    result = orchestrator.create_content_workflow(
        topic="The Future of Artificial Intelligence in 2025",
        content_type="blog_post",
        word_count=1000,
        tone="professional",
        research_depth="standard"
    )
    
    # Export results
    if result.get("status") != "failed":
        orchestrator.export_workflow(result)
    
    print("\n✅ Test complete!")