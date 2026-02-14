"""
Multi-Agent System - Day 9
Main executable for the complete content creation system

This system demonstrates a production-ready multi-agent architecture with:
- Real-time Google search
- Specialized agents (Researcher, Writer, Reviewer)
- Inter-agent communication
- Feedback loops and quality control
- Complete workflow orchestration
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

from workflow_orchestrator import WorkflowOrchestrator

# Load environment variables
load_dotenv()


def print_banner():
    """Print system banner"""
    print("\n" + "="*70)
    print(" " * 15 + "🤖 MULTI-AGENT CONTENT CREATION SYSTEM")
    print(" " * 20 + "Day 9 - Advanced AI Agents")
    print("="*70 + "\n")


def print_menu():
    """Print main menu"""
    print("\n" + "-"*70)
    print("MENU OPTIONS:")
    print("-"*70)
    print("1. Create Blog Post")
    print("2. Create Article")
    print("3. Create Research Report")
    print("4. Custom Content (Advanced)")
    print("5. View System Statistics")
    print("6. Exit")
    print("-"*70)


def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def create_blog_post(orchestrator: WorkflowOrchestrator):
    """Interactive blog post creation"""
    print("\n" + "="*70)
    print("📝 BLOG POST CREATION")
    print("="*70)
    
    topic = get_user_input("\nEnter blog post topic")
    if not topic:
        print("❌ Topic is required!")
        return
    
    word_count = get_user_input("Target word count", "1000")
    try:
        word_count = int(word_count)
    except:
        word_count = 1000
    
    tone = get_user_input("Writing tone (professional/casual/technical)", "professional")
    depth = get_user_input("Research depth (quick/standard/deep)", "standard")
    
    print(f"\n🚀 Starting content creation workflow...")
    print(f"   Topic: {topic}")
    print(f"   Words: {word_count}")
    print(f"   Tone: {tone}")
    print(f"   Research: {depth}")
    
    # Execute workflow
    result = orchestrator.create_content_workflow(
        topic=topic,
        content_type="blog_post",
        word_count=word_count,
        tone=tone,
        research_depth=depth
    )
    
    # Export results
    if result.get("status") != "failed":
        files = orchestrator.export_workflow(result)
        
        print("\n✅ Blog post created successfully!")
        print(f"\n📄 Preview (first 500 characters):")
        print("-"*70)
        print(result["content"][:500] + "...")
        print("-"*70)
        
        print(f"\n💾 Files saved:")
        for file_type, path in files.items():
            print(f"   - {file_type}: {path}")
    else:
        print(f"\n❌ Content creation failed: {result.get('error', 'Unknown error')}")


def create_article(orchestrator: WorkflowOrchestrator):
    """Interactive article creation"""
    print("\n" + "="*70)
    print("📰 ARTICLE CREATION")
    print("="*70)
    
    topic = get_user_input("\nEnter article topic")
    if not topic:
        print("❌ Topic is required!")
        return
    
    word_count = get_user_input("Target word count", "1500")
    try:
        word_count = int(word_count)
    except:
        word_count = 1500
    
    print(f"\n🚀 Starting article creation...")
    
    result = orchestrator.create_content_workflow(
        topic=topic,
        content_type="article",
        word_count=word_count,
        tone="professional",
        research_depth="standard"
    )
    
    if result.get("status") != "failed":
        files = orchestrator.export_workflow(result)
        print("\n✅ Article created successfully!")
        print(f"💾 Files saved to: {os.path.dirname(list(files.values())[0])}")
    else:
        print(f"\n❌ Article creation failed: {result.get('error', 'Unknown error')}")


def create_research_report(orchestrator: WorkflowOrchestrator):
    """Interactive research report creation"""
    print("\n" + "="*70)
    print("🔬 RESEARCH REPORT CREATION")
    print("="*70)
    
    topic = get_user_input("\nEnter research topic")
    if not topic:
        print("❌ Topic is required!")
        return
    
    print(f"\n🚀 Starting deep research and report creation...")
    
    result = orchestrator.create_content_workflow(
        topic=topic,
        content_type="research_report",
        word_count=2000,
        tone="academic",
        research_depth="deep"
    )
    
    if result.get("status") != "failed":
        files = orchestrator.export_workflow(result)
        print("\n✅ Research report created successfully!")
        print(f"\n📊 Research Statistics:")
        print(f"   - Sources found: {result['metadata']['sources_count']}")
        print(f"   - Word count: {result['metadata']['word_count']}")
        print(f"   - Quality score: {result['metadata']['final_score']*100:.1f}%")
    else:
        print(f"\n❌ Report creation failed: {result.get('error', 'Unknown error')}")


def custom_content(orchestrator: WorkflowOrchestrator):
    """Custom content creation with all options"""
    print("\n" + "="*70)
    print("⚙️ CUSTOM CONTENT CREATION")
    print("="*70)
    
    topic = get_user_input("\nEnter topic")
    if not topic:
        print("❌ Topic is required!")
        return
    
    content_type = get_user_input("Content type (blog_post/article/report/guide)", "blog_post")
    word_count = int(get_user_input("Word count", "1500"))
    tone = get_user_input("Tone (professional/casual/technical/academic)", "professional")
    depth = get_user_input("Research depth (quick/standard/deep)", "standard")
    
    print(f"\n🚀 Starting custom content creation...")
    
    result = orchestrator.create_content_workflow(
        topic=topic,
        content_type=content_type,
        word_count=word_count,
        tone=tone,
        research_depth=depth
    )
    
    if result.get("status") != "failed":
        files = orchestrator.export_workflow(result)
        print("\n✅ Custom content created successfully!")
    else:
        print(f"\n❌ Content creation failed: {result.get('error', 'Unknown error')}")


def view_statistics(orchestrator: WorkflowOrchestrator):
    """View system statistics"""
    print("\n" + "="*70)
    print("📊 SYSTEM STATISTICS")
    print("="*70)
    
    if not orchestrator.workflow_history:
        print("\n❌ No workflows executed yet!")
        return
    
    total_workflows = len(orchestrator.workflow_history)
    successful = sum(1 for w in orchestrator.workflow_history if w.get("status") == "completed")
    
    total_time = sum(w["timing"]["total_seconds"] for w in orchestrator.workflow_history)
    avg_time = total_time / total_workflows if total_workflows > 0 else 0
    
    total_words = sum(w["metadata"]["word_count"] for w in orchestrator.workflow_history)
    avg_words = total_words / total_workflows if total_workflows > 0 else 0
    
    print(f"\nWorkflow Summary:")
    print(f"  - Total workflows: {total_workflows}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {total_workflows - successful}")
    
    print(f"\nPerformance:")
    print(f"  - Average time: {avg_time/60:.2f} minutes")
    print(f"  - Total words generated: {total_words}")
    print(f"  - Average word count: {avg_words:.0f}")
    
    print(f"\nMessage Bus Stats:")
    stats = orchestrator.message_bus.get_statistics()
    print(f"  - Messages sent: {stats['total_messages_sent']}")
    print(f"  - Messages delivered: {stats['total_messages_delivered']}")
    
    print(f"\nRecent Workflows:")
    for i, workflow in enumerate(orchestrator.workflow_history[-5:], 1):
        print(f"\n  {i}. {workflow['topic'][:50]}")
        print(f"     Status: {workflow['status']}")
        print(f"     Time: {workflow['timing']['total_minutes']:.2f} min")
        print(f"     Score: {workflow['metadata']['final_score']*100:.0f}%")


def main():
    """Main program loop"""
    print_banner()
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found!")
        print("\n📝 Setup Instructions:")
        print("1. Copy .env.example to .env")
        print("2. Add your Groq API key:")
        print("   GROQ_API_KEY=your_key_here")
        print("\n💡 Get a free API key at: https://console.groq.com")
        return
    
    print("✅ API Key found")
    print("🚀 Initializing Multi-Agent System...")
    
    # Create outputs directory
    os.makedirs("./outputs", exist_ok=True)
    
    try:
        # Initialize orchestrator
        orchestrator = WorkflowOrchestrator(api_key)
        
        print("\n✅ System ready!")
        print("\n💡 This system uses:")
        print("   - Research Agent (Google search + web scraping)")
        print("   - Writer Agent (Content creation)")
        print("   - Reviewer Agent (Quality control)")
        print("   - Orchestrator (Workflow management)")
        
        # Main menu loop
        while True:
            print_menu()
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                create_blog_post(orchestrator)
            elif choice == "2":
                create_article(orchestrator)
            elif choice == "3":
                create_research_report(orchestrator)
            elif choice == "4":
                custom_content(orchestrator)
            elif choice == "5":
                view_statistics(orchestrator)
            elif choice == "6":
                print("\n👋 Thank you for using the Multi-Agent System!")
                print("✅ All workflows saved to ./outputs/\n")
                break
            else:
                print("\n❌ Invalid choice! Please enter 1-6.")
            
            input("\nPress Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ System interrupted by user")
        print("✅ All workflows saved to ./outputs/\n")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()