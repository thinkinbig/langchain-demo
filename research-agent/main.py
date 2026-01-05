"""Main entry point for research agent"""

import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from graph.error_handler import (  # noqa: E402
    is_recoverable_error,
)
from langgraph.store.memory import InMemoryStore  # noqa: E402
from memory.langmem_integration import LongTermMemoryLangMemBridge  # noqa: E402
from nodes.human_approach_selector import format_approaches_for_display  # noqa: E402
from schemas import ResearchState  # noqa: E402


async def main():
    """Run research agent on a query"""
    import sys

    # Parse command line arguments
    # Usage: python main.py [query] [--resume thread_id] [--thread-id thread_id]
    query = None
    thread_id = None
    resume_mode = False

    # Check for resume/thread-id flags
    if "--resume" in sys.argv:
        resume_idx = sys.argv.index("--resume")
        if resume_idx + 1 < len(sys.argv):
            thread_id = sys.argv[resume_idx + 1]
            resume_mode = True
            # Remove processed args
            sys.argv.pop(resume_idx)
            sys.argv.pop(resume_idx)
    elif "--thread-id" in sys.argv:
        thread_id_idx = sys.argv.index("--thread-id")
        if thread_id_idx + 1 < len(sys.argv):
            thread_id = sys.argv[thread_id_idx + 1]
            # Remove processed args
            sys.argv.pop(thread_id_idx)
            sys.argv.pop(thread_id_idx)

    # Check for saved checkpoint file if no thread_id provided
    if not thread_id:
        try:
            checkpoint_file = ".last_checkpoint.txt"
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, "r") as f:
                    saved_thread_id = f.read().strip()
                    if saved_thread_id:
                        print("=" * 80)
                        print("MULTI-AGENT RESEARCH SYSTEM")
                        print("=" * 80)
                        print(f"\n💾 Found saved checkpoint: {saved_thread_id}")
                        user_choice = input("   Resume from checkpoint? (y/n): ").strip().lower()
                        if user_choice == 'y':
                            thread_id = saved_thread_id
                            resume_mode = True
        except Exception:
            pass  # Ignore errors reading checkpoint file

    # Get query from remaining args or use default
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = (
            "What are the key differences between Python and Rust "
            "programming languages?"
        )

    if not resume_mode:
        print("=" * 80)
        print("MULTI-AGENT RESEARCH SYSTEM")
        print("=" * 80)

    # Handle resume mode
    if resume_mode and thread_id:
        print("\n🔄 RESUME MODE: Continuing from checkpoint")
        print(f"   Thread ID: {thread_id}\n")

        # Try to get existing state from checkpointer
        try:
            from config import settings
            from graph import app

            # Check if checkpointer backend supports persistence
            if settings.CHECKPOINTER_BACKEND == "memory":
                print("   ⚠️  Checkpointer is using 'memory' backend.")
                print("   ⚠️  Memory checkpoints don't persist across restarts.")
                print("   ⚠️  Set CHECKPOINTER_BACKEND=sqlite for persistent checkpoints.\n")
                thread_id = None
                resume_mode = False
            else:
                config = {"configurable": {"thread_id": thread_id}}

                # Debug: print checkpointer info
                print(f"   Checking checkpoint with backend: {settings.CHECKPOINTER_BACKEND}")
                if settings.CHECKPOINTER_BACKEND == "sqlite":
                    from config import settings as config_settings
                    db_path = config_settings.CHECKPOINTER_DB_PATH
                    print(f"   SQLite DB path: {db_path}")
                    if os.path.exists(db_path):
                        import sqlite3
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        try:
                            cursor.execute(
                                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                                (thread_id,)
                            )
                            count = cursor.fetchone()[0]
                            print(f"   Found {count} checkpoint(s) for this thread_id")
                        except Exception as e:
                            print(f"   Could not query checkpoints: {e}")
                        finally:
                            conn.close()

                existing_state = await app.aget_state(config)

                # Check if checkpoint exists (has values or next nodes)
                checkpoint_exists = False
                if existing_state:
                    # Debug: print state info
                    print(f"   State object type: {type(existing_state)}")
                    print(f"   Has values attr: {hasattr(existing_state, 'values')}")
                    print(f"   Has next attr: {hasattr(existing_state, 'next')}")

                    if hasattr(existing_state, "values"):
                        print(f"   Values is not None: {existing_state.values is not None}")
                        if existing_state.values:
                            print(f"   Values dict keys: {list(existing_state.values.keys())[:5]}")
                            checkpoint_exists = True

                    if hasattr(existing_state, "next"):
                        print(f"   Next nodes: {existing_state.next}")
                        if existing_state.next:
                            checkpoint_exists = True

                if checkpoint_exists:
                    state_values = (
                        existing_state.values
                        if hasattr(existing_state, "values")
                        else {}
                    )
                    existing_query = state_values.get("query", "")

                    if existing_query:
                        print("   ✅ Checkpoint found!")
                        print(f"   Original Query: {existing_query}")
                        iter_count = state_values.get('iteration_count', 0)
                        print(f"   Current Iteration: {iter_count}")
                        findings = state_values.get('subagent_findings', [])
                        print(f"   Findings Count: {len(findings)}")

                        # Check next nodes to see where execution will resume
                        if hasattr(existing_state, "next") and existing_state.next:
                            next_nodes = existing_state.next
                            if isinstance(next_nodes, list):
                                next_nodes = (
                                    next_nodes[0] if next_nodes else "END"
                                )
                            print(f"   Will resume at: {next_nodes}\n")
                        else:
                            print("   Will continue from last checkpoint\n")

                        # Ask user if they want to continue with existing query
                        # or use the provided query (if different)
                        if query and query != existing_query:
                            print("   ⚠️  Query mismatch!")
                            print(f"      Existing: {existing_query}")
                            print(f"      New: {query}")
                            user_choice = input(
                                "   Continue with existing query? (y/n): "
                            ).strip().lower()
                            if user_choice == 'y':
                                query = existing_query
                            else:
                                print("   ⚠️  Using new query (new session)")
                                thread_id = None
                                resume_mode = False
                    else:
                        print("   ⚠️  Checkpoint found but no query in state")
                        print("   Starting fresh\n")
                        thread_id = None
                        resume_mode = False
                else:
                    print("   ⚠️  No existing checkpoint found for this thread_id")
                    print("   Starting fresh session\n")
                    thread_id = None
                    resume_mode = False
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            import traceback
            print(f"   Error details: {traceback.format_exc()}")
            print("   Starting fresh session\n")
            thread_id = None
            resume_mode = False
    else:
        print(f"\n🔍 Research Query: {query}\n")

    # Initialize Cost Controller
    from cost_control import (
        CostController,
        CostLimitExceeded,
        CostTrackingCallback,
        QueryBudget,
    )

    cost_controller = CostController()
    query_budget = QueryBudget()

    # 1. Check Daily Limit (Pre-flight check)
    # Estimate typical query cost (e.g., 50k tokens approx $0.10)
    can_accept, message = cost_controller.check_daily_limit(
        estimated_tokens=50_000, estimated_cost=0.10
    )
    if not can_accept:
        print(f"⛔ QUERY REJECTED: {message}")
        print("Daily budget exhausted. Please increase budget or wait for reset.")
        return

    print("✅ Daily budget check passed.")

    # Generate or use existing thread ID for checkpointer
    # If resuming, use provided thread_id; otherwise generate new one
    if not thread_id:
        thread_id = str(uuid.uuid4())
    recoverable_error_occurred = False
    last_error = None

    # Initialize state - if resuming, state will be loaded from checkpoint
    # Otherwise, create new initial state
    initial_state = None
    if not resume_mode:
        # Initialize state with Pydantic validation for new session
        initial_state = ResearchState(
            query=query,
            research_plan="",
            subagent_tasks=[],
            subagent_findings=[],
            iteration_count=0,
            needs_more_research=False,
            synthesized_results="",
            citations=[],
            final_report="",
        )
    # If resuming, initial_state will be None and app will continue from checkpoint

    # Initialize memory bridge for LangMem integration
    # Use InMemoryStore for LangMem (can be replaced with persistent store)
    try:
        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
        memory_bridge = LongTermMemoryLangMemBridge(store=store)
    except Exception as e:
        print(f"  ⚠️  LangMem not available: {e}")
        memory_bridge = None

    # Collect conversation messages for LangMem processing
    conversation_messages = []

    try:
        # Run the graph with Cost Tracking Callback
        # This will auto-update query_budget on every LLM call
        cost_callback = CostTrackingCallback(query_budget)

        # Set recursion limit based on MAX_ITERATIONS
        # Each iteration involves multiple nodes, so we set a safe limit
        # Typical path per iteration: complexity_analyzer -> lead_researcher ->
        # subagent(s) -> filter_findings -> synthesizer -> decision
        # This is ~6-10 nodes per iteration, so we use a multiplier of 15 for safety
        from config import settings
        recursion_limit = (settings.MAX_ITERATIONS + 1) * 15

        config = {
            "callbacks": [cost_callback],
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }

        # Use astream to handle interrupts
        # When interrupt_before is configured, execution will pause before
        # the specified node, allowing us to collect user input
        final_state = None
        interrupt_handled = False

        # Start streaming - execution will pause at interrupt points
        # If resuming, pass None to continue from checkpoint
        # Otherwise, pass initial_state to start new session
        stream_input = None if resume_mode else initial_state
        async for event in app.astream(
            stream_input, config=config, stream_mode="updates"
        ):
            # Check current state to see if we're at an interrupt
            current_state = await app.aget_state(config)

            # Check if we're paused before human_approach_selector
            # When interrupt_before is set, the state will indicate the next node
            if (
                not interrupt_handled
                and hasattr(current_state, "next")
                and current_state.next
            ):
                next_nodes = current_state.next
                if isinstance(next_nodes, str):
                    next_nodes = [next_nodes]

                # Check if human_approach_selector is in the next nodes
                if any(
                    "human_approach_selector" in str(node) for node in next_nodes
                ):
                    # We're at the interrupt - get approach_evaluation from state
                    state_dict = (
                        current_state.values
                        if hasattr(current_state, "values")
                        else current_state
                    )
                    approach_evaluation = state_dict.get("approach_evaluation")

                    if approach_evaluation:
                        # Display approaches to user
                        display_text = format_approaches_for_display(
                            approach_evaluation
                        )
                        print(display_text)

                        # Wait for user input (synchronous, blocking)
                        while True:
                            try:
                                user_input = input("Select approach (0-2): ").strip()
                                selected_index = int(user_input)

                                # Validate input
                                if 0 <= selected_index <= 2:
                                    # Update state with user selection
                                    app.update_state(
                                        config,
                                        {"selected_approach_index": selected_index}
                                    )
                                    print(
                                        f"\n✅ Selected approach {selected_index + 1}"
                                    )
                                    interrupt_handled = True

                                    # Resume execution after user input
                                    # Break out to handle resume separately
                                    break
                                else:
                                    print("⚠️  Invalid input. Please enter 0, 1, or 2.")
                            except ValueError:
                                print("⚠️  Invalid input. Please enter a number (0-2).")
                            except KeyboardInterrupt:
                                print("\n\n❌ Interrupted by user. Exiting...")
                                return

                    # After handling interrupt, resume execution
                    # Continue streaming from where we left off
                    # Using astream(None, ...) will continue from the current state
                    # This will execute human_approach_selector and then continue
                    async for resume_event in app.astream(
                        None, config=config, stream_mode="updates"
                    ):
                        # Process events from resumed execution
                        # Check if we're done
                        if "__end__" in resume_event:
                            final_state = resume_event["__end__"]
                            break
                    # Break out of the outer loop since we've handled the interrupt
                    break

            # Check if we're done
            if "__end__" in event:
                final_state = event["__end__"]
                break

        # If we didn't get final_state from stream, get it from state
        if final_state is None:
            current_state = await app.aget_state(config)
            final_state = (
                current_state.values
                if hasattr(current_state, "values")
                else current_state
            )

        # Display results
        print("\n" + "=" * 80)
        print("FINAL REPORT")
        print("=" * 80)
        print(final_state['final_report'] or "No report generated")
        print("=" * 80)

        print("\n📊 Summary:")
        print(f"   Iterations: {final_state['iteration_count']}")
        print(f"   Findings: {len(final_state['subagent_findings'])}")
        print(f"   Citations: {len(final_state['citations'])}")

        # Process conversation through LangMem for automatic memory extraction
        if memory_bridge:
            try:
                # Build conversation messages from state
                from langchain_core.messages import AIMessage, HumanMessage
                conversation_messages = [
                    HumanMessage(content=query),
                    AIMessage(content=final_state.get('final_report', '')),
                ]

                # Process through LangMem's background manager
                result = memory_bridge.process_conversation(conversation_messages)
                if result:
                    msg = (
                        "  🧠 LangMem: Automatically extracted and "
                        "consolidated memories"
                    )
                    print(msg)
            except Exception as e:
                print(f"  ⚠️  LangMem processing failed: {e}")

    except CostLimitExceeded as e:
        last_error = e
        recoverable_error_occurred = False
        print(f"\n⛔ EXCEPTION: Research stopped due to cost limit: {e}")
        print(
            "   History will be discarded on next run (unrecoverable error)"
        )
    except asyncio.TimeoutError as e:
        last_error = e
        recoverable_error_occurred = False
        from config import settings
        print(
            f"\n⛔ EXCEPTION: Research stopped due to timeout "
            f"({settings.TIMEOUT_MAIN}s exceeded)"
        )
        print(
            "   History will be discarded on next run (unrecoverable error)"
        )
    except Exception as e:
        last_error = e
        recoverable_error_occurred = is_recoverable_error(e)

        if recoverable_error_occurred:
            print(f"\n⚠️  Recoverable error during execution: {e}")
            print(f"   History preserved (thread_id: {thread_id})")
            print("   You can resume by reusing this thread_id")
        else:
            print(f"\n❌ Unrecoverable error during execution: {e}")
            print("   History will be discarded on next run")
    finally:
        # 2. Record actual usage to Daily Budget
        # Only what was consumed
        print("\n💰 Cost Report:")
        print(f"   Tokens Used: {query_budget.current_tokens}")
        print(f"   Est. Cost: ${query_budget.current_cost:.4f}")

        cost_controller.record_daily_usage(
            tokens=query_budget.current_tokens,
            cost=query_budget.current_cost
        )
        print("   (Recorded to daily budget)")

        # Provide guidance for error recovery
        if recoverable_error_occurred and last_error:
            print("\n💡 Recovery Tip:")
            print("   This error is recoverable. To resume from where you left off:")
            print(f"   - Reuse thread_id: {thread_id}")
            print("   - The checkpointer will restore the previous state")
            print("   - Execution will continue from the last checkpoint")
            print("\n   To resume, run:")
            print(f"   python main.py --resume {thread_id}")

            # Save thread_id to file for easy recovery
            try:
                checkpoint_file = ".last_checkpoint.txt"
                with open(checkpoint_file, "w") as f:
                    f.write(f"{thread_id}\n")
                print(f"   (Thread ID saved to {checkpoint_file})")
            except Exception:
                pass  # Don't fail if file write fails
        elif last_error and not recoverable_error_occurred:
            print("\n💡 Recovery Tip:")
            print("   This error is unrecoverable. For next run:")
            print("   - A new thread_id will be generated automatically")
            print("   - Previous history will be discarded")
            print("   - Execution will start fresh")

        # Cleanup expired memories (non-blocking)
        try:
            from memory.cleanup import cleanup_expired_memories  # noqa: E402

            asyncio.create_task(cleanup_expired_memories())
        except Exception:
            pass  # Don't fail if cleanup doesn't work


if __name__ == "__main__":
    asyncio.run(main())
