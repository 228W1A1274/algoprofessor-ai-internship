"""
Test System - Day 9 Multi-Agent System
Quick test to verify all components are working
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def test_imports():
    """Test that all imports work"""
    print("\n" + "="*60)
    print("TEST 1: Import Tests")
    print("="*60)
    
    try:
        from agent_definitions import BaseAgent, RESEARCHER_CONFIG, WRITER_CONFIG, REVIEWER_CONFIG
        print("✅ agent_definitions.py")
        
        from communication_protocol import MessageBus, Message, MessageType
        print("✅ communication_protocol.py")
        
        from custom_tools import GoogleSearchTool, WebScraperTool
        print("✅ custom_tools.py")
        
        from research_agent import ResearchAgent
        print("✅ research_agent.py")
        
        from writer_agent import WriterAgent
        print("✅ writer_agent.py")
        
        from reviewer_agent import ReviewerAgent
        print("✅ reviewer_agent.py")
        
        from workflow_orchestrator import WorkflowOrchestrator
        print("✅ workflow_orchestrator.py")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_key():
    """Test API key is set"""
    print("\n" + "="*60)
    print("TEST 2: API Key Test")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        print("\n💡 Setup instructions:")
        print("1. Copy .env.example to .env")
        print("2. Add your Groq API key:")
        print("   GROQ_API_KEY=your_key_here")
        print("\n Get a free key at: https://console.groq.com")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    return True


def test_tools():
    """Test search tools work"""
    print("\n" + "="*60)
    print("TEST 3: Search Tools Test")
    print("="*60)
    
    try:
        from custom_tools import GoogleSearchTool
        
        print("\n🔍 Testing Google search...")
        search_tool = GoogleSearchTool(max_results=3)
        results = search_tool.search("AI news", max_results=3)
        
        if results:
            print(f"✅ Search successful! Found {len(results)} results")
            print(f"\nFirst result:")
            print(f"   Title: {results[0]['title']}")
            print(f"   URL: {results[0]['url'][:50]}...")
            return True
        else:
            print("⚠️ Search returned no results (may be network issue)")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def test_agent_creation():
    """Test agent creation"""
    print("\n" + "="*60)
    print("TEST 4: Agent Creation Test")
    print("="*60)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Skipping (no API key)")
        return False
    
    try:
        from research_agent import ResearchAgent
        from writer_agent import WriterAgent
        from reviewer_agent import ReviewerAgent
        
        print("\n🤖 Creating agents...")
        
        researcher = ResearchAgent(api_key)
        print(f"✅ {researcher.config.name} created")
        
        writer = WriterAgent(api_key)
        print(f"✅ {writer.config.name} created")
        
        reviewer = ReviewerAgent(api_key)
        print(f"✅ {reviewer.config.name} created")
        
        print("\n✅ All agents created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_message_bus():
    """Test message bus"""
    print("\n" + "="*60)
    print("TEST 5: Message Bus Test")
    print("="*60)
    
    try:
        from communication_protocol import MessageBus, Message, MessageType, MessagePriority
        
        print("\n📡 Creating message bus...")
        bus = MessageBus()
        
        print("📤 Sending test message...")
        message = Message(
            from_agent="TestAgent1",
            to_agent="TestAgent2",
            message_type=MessageType.DATA_TRANSFER,
            content={"test": "data"},
            priority=MessagePriority.NORMAL
        )
        
        bus.send(message)
        
        print("📨 Receiving message...")
        received = bus.receive("TestAgent2", timeout=0.1)
        
        if received:
            print(f"✅ Message received: {received}")
            print(f"   From: {received.from_agent}")
            print(f"   To: {received.to_agent}")
            print(f"   Type: {received.message_type.value}")
            return True
        else:
            print("❌ Message not received")
            return False
            
    except Exception as e:
        print(f"❌ Message bus test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "🧪 DAY 9 MULTI-AGENT SYSTEM - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("API Key", test_api_key),
        ("Search Tools", test_tools),
        ("Agent Creation", test_agent_creation),
        ("Message Bus", test_message_bus)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use!")
        print("\n💡 Next step: Run the main system:")
        print("   python multi_agent_system.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Setup API key in .env file")
        print("   - Check internet connection for search tests")
    
    print()


if __name__ == "__main__":
    run_all_tests()
