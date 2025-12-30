"""Visualize LangGraph workflow using different methods"""

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from root .env file
# Get the root directory (parent of decision-agent)
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: try loading from current directory or default location
    load_dotenv()

from graph import app  # noqa: E402

# Method 1: Get graph structure and print it
print("=" * 80)
print("Method 1: Graph Structure")
print("=" * 80)
try:
    graph = app.get_graph()
    print(f"Nodes: {list(graph.nodes.keys())}")
    print(f"Number of nodes: {len(graph.nodes)}")
    if hasattr(graph, "edges"):
        print(f"Number of edges: {len(graph.edges)}")
except Exception as e:
    print(f"Error getting graph: {e}")

# Method 2: Print graph in text format (ASCII art)
print("\n" + "=" * 80)
print("Method 2: Graph Text Representation (ASCII)")
print("=" * 80)
try:
    ascii_diagram = app.get_graph().draw_ascii()
    print(ascii_diagram)
except Exception as e:
    print(f"ASCII diagram not available: {e}")
    print("Trying alternative method...")
    try:
        # Alternative: use print method if available
        print(app.get_graph())
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")

# Method 3: Generate Mermaid diagram
print("\n" + "=" * 80)
print("Method 3: Mermaid Diagram")
print("=" * 80)
try:
    mermaid_diagram = app.get_graph().draw_mermaid()
    print(mermaid_diagram)
    print("\n💡 You can copy this Mermaid diagram and paste it into:")
    print("   - https://mermaid.live/ (online editor)")
    print("   - GitHub/GitLab markdown files")
    print("   - Notion, Obsidian, etc.")

    # Save to file
    with open("decision_agent_graph.mmd", "w") as f:
        f.write(mermaid_diagram)
    print("\n✅ Mermaid diagram saved to decision_agent_graph.mmd")
except Exception as e:
    print(f"Mermaid diagram not available: {e}")
    print("💡 Try: pip install 'langgraph[visualization]'")

# Method 4: Save graph visualization to PNG (if graphviz available)
print("\n" + "=" * 80)
print("Method 4: Save Graph Visualization (PNG)")
print("=" * 80)
try:
    app.get_graph().draw_mermaid_png(output_file_path="decision_agent_graph.png")
    print("✅ Graph saved as decision_agent_graph.png")
except Exception as e:
    print(f"PNG export not available: {e}")
    print("💡 To enable PNG export:")
    print("   1. Install graphviz: sudo apt-get install graphviz (Linux)")
    print("      or: brew install graphviz (macOS)")
    print("   2. Install Python package: pip install pygraphviz")
    print("   3. Or use the Mermaid diagram with online tools")

# Method 5: Print detailed node information
print("\n" + "=" * 80)
print("Method 5: Detailed Node Information")
print("=" * 80)
try:
    graph = app.get_graph()
    for node_id in graph.nodes.keys():
        print(f"  - {node_id}")
except Exception as e:
    print(f"Error: {e}")

# Method 6: Use LangGraph Studio command
print("\n" + "=" * 80)
print("Method 6: LangGraph Studio")
print("=" * 80)
print("To use LangGraph Studio for interactive visualization:")
print("  1. Install: pip install langgraph-cli")
print("  2. Run: langgraph dev")
print("  3. Open browser to the provided URL")
print("  4. You'll see an interactive graph visualization")

print("\n" + "=" * 80)
print("Visualization Complete!")
print("=" * 80)
print("\n📚 Summary of visualization methods:")
print("   1. ✅ ASCII diagram (text-based)")
print("   2. ✅ Mermaid diagram (saved to .mmd file)")
print("   3. ⚠️  PNG export (requires graphviz)")
print("   4. 💡 LangGraph Studio (interactive, requires langgraph-cli)")
print("   5. 💡 LangSmith (runtime visualization and debugging)")
