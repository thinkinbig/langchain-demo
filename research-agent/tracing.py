"""Local tracing for LLM calls - logs prompts and responses to file"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class LocalFileTracer(BaseCallbackHandler):
    """Callback handler that logs LLM interactions to a local file"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_run_id = None
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_file = self.log_dir / f"session_{timestamp}.jsonl"

    def _log(self, event_type: str, data: Dict[str, Any]):
        """Append a log entry to the session file"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **data
        }
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Log when LLM starts"""
        self._log("llm_start", {
            "model": serialized.get("kwargs", {}).get("model", "unknown"),
            "prompts": prompts,
            "run_id": str(kwargs.get("run_id", "")),
        })

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Log when chat model starts"""
        # Convert messages to readable format
        formatted_messages = []
        for msg_list in messages:
            for msg in msg_list:
                formatted_messages.append({
                    "role": msg.__class__.__name__.replace("Message", ""),
                    "content": (
                        msg.content[:500]
                        if len(msg.content) > 500
                        else msg.content
                    )
                })

        self._log("chat_model_start", {
            "model": serialized.get("kwargs", {}).get("model", "unknown"),
            "messages": formatted_messages,
            "run_id": str(kwargs.get("run_id", "")),
        })

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Log LLM response"""
        try:
            output = (
                response.generations[0][0].text
                if response.generations
                else str(response)
            )
        except Exception:
            output = str(response)

        self._log("llm_end", {
            "output": output[:1000] if len(output) > 1000 else output,
            "run_id": str(kwargs.get("run_id", "")),
        })

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log LLM errors"""
        self._log("llm_error", {
            "error": str(error),
            "run_id": str(kwargs.get("run_id", "")),
        })


# Global tracer instance
_tracer: Optional[LocalFileTracer] = None


def get_tracer() -> LocalFileTracer:
    """Get or create the global tracer instance"""
    global _tracer
    if _tracer is None:
        log_dir = os.getenv("TRACE_LOG_DIR", "logs")
        _tracer = LocalFileTracer(log_dir=log_dir)
        print(f"📝 Tracing enabled. Logs will be written to: {_tracer.session_file}")
    return _tracer


def get_callbacks() -> List[BaseCallbackHandler]:
    """Get callbacks list for LLM calls"""
    if os.getenv("ENABLE_TRACING", "false").lower() == "true":
        return [get_tracer()]
    return []
