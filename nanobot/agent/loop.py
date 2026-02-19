"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager
from nanobot.session.token_store import TokenStore


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        # Store current message context for error notifications
        self._current_message_context = {}
        
        # Tool output callback for real-time updates
        self._tool_output_callback: Callable | None = None
        
        # Set error callback for provider if it supports it
        if hasattr(self.provider, 'error_callback'):
            async def error_callback(error_message):
                # Get current message context
                context = self._current_message_context
                if context:
                    # Send error notification
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=context['channel'],
                            chat_id=context['chat_id'],
                            content=error_message
                        )
                    )
            
            self.provider.error_callback = error_callback
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        
        # Load max_window_context from config
        self.max_window_context = self._load_max_window_context()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)
    
    def set_tool_output_callback(self, callback: Callable | None) -> None:
        """Set callback for real-time tool output.
        
        Args:
            callback: Async function that accepts (channel, chat_id, tool_name, output) or None to disable
        """
        self._tool_output_callback = callback

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove </think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"</think>[\s\S]*?<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str]]:
        """Run the agent loop with LLM interactions.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback to push intermediate content to the user.

        Returns:
            Tuple of (final_content, tools_used_list).
        """

    def _load_max_window_context(self) -> int:
        """Load max_window_context from config, default to 100000."""
        try:
            from nanobot.config.loader import load_config
            config = load_config()
            return config.agents.defaults.max_window_context
        except Exception as e:
            logger.debug(f"Could not load max_window_context from config: {e}")
            return 100000  # Default value

    async def _summarize_context(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to summarize older messages for context compression.

        Args:
            messages: List of older messages to summarize.

        Returns:
            Summary string with prefix note.
        """
        # Build an optimized prompt for better summarization
        summary_prompt = """Please summarize the following conversation comprehensively.

IMPORTANT: Pay special attention to and PRESERVE:
- User's explicit requirements, requests, or instructions (marked with emphasis like **, CAPS, or explicit statements)
- Technical details, configuration values, file paths, or code snippets
- Decisions made or conclusions reached
- Any unresolved questions or pending tasks

Conversation to summarize:
"""

        # Format messages for the prompt (no truncation - keep full content)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                summary_prompt += f"\n{role}: {content}"

        summary_prompt += "\n\nProvide a comprehensive summary that captures all important details, especially user requirements and technical information:"

        try:
            # Call LLM to generate summary
            response = await self.provider.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                model=self.model,
                max_tokens=1500,  # Allow longer summary for comprehensive coverage
                temperature=0.3   # Lower temperature for more accurate summarization
            )

            # Handle response safely - content might be dict in some error cases
            content = response.content
            if isinstance(content, dict):
                content = str(content)
            
            if content:
                summary = content.strip() if hasattr(content, 'strip') else str(content)
                # Add prefix note to the summary
                prefix = "This is a summary of previous conversation. Some details may have been optimized. If anything is unclear, please ask directly.\n\n"
                return prefix + summary
            else:
                # Fallback: return count-based summary
                user_msgs = sum(1 for m in messages if m.get("role") == "user")
                assistant_msgs = sum(1 for m in messages if m.get("role") == "assistant")
                return f"This is a summary of previous conversation. Some details may have been optimized. If anything is unclear, please ask directly.\n\n{user_msgs} user messages and {assistant_msgs} assistant responses (details compressed)"

        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            # Fallback: return simple count with prefix
            return f"This is a summary of previous conversation. Some details may have been optimized. If anything is unclear, please ask directly.\n\n{len(messages)} messages (compression failed, using basic summary)"

    async def _compress_single_message(self, content: str) -> str:
        """
        Compress a single long message using LLM.
        First truncate if too long, then compress.
        
        Args:
            content: The long message content to compress.
            
        Returns:
            Compressed content within token limits.
        """
        from nanobot.session.manager import estimate_tokens
        
        # First check if content is too long for compression LLM
        # Assume compression LLM has similar maxWindowContext limit
        content_tokens = estimate_tokens(content)
        max_compress_input = int(self.max_window_context * 0.8)  # Use 80% of limit for safety
        
        if content_tokens > max_compress_input:
            # Truncate before compression
            logger.debug(f"Message too long for compression ({content_tokens} tokens), truncating first...")
            # Reserve 500 tokens for prompt overhead, truncate the rest
            keep_chars = int((max_compress_input - 500) * 2.12)
            truncated_notice = "\n\n[Content truncated before compression - showing first part only]"
            content = content[:keep_chars] + truncated_notice
        
        compress_prompt = """Please compress the following text while preserving key information:
- Keep important keywords and technical details
- Maintain the core meaning and context
- Be as detailed as possible within the limit
- Preserve any code snippets, file paths, or configuration values

Text to compress:
"""
        compress_prompt += f"\n{content}\n\nCompressed version (within 1500 characters):"
        
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": compress_prompt}],
                model=self.model,
                max_tokens=2000,  # Allow up to 2000 tokens for detailed compression
                temperature=0.3
            )
            
            # Handle response safely - content might be dict in some error cases
            content = response.content
            if isinstance(content, dict):
                content = str(content)
            
            if content:
                compressed = content.strip() if hasattr(content, 'strip') else str(content)
                # Add prefix note
                prefix = "[This message was compressed due to length. Original was too long for context window. Key points preserved:]\n\n"
                return prefix + compressed
            else:
                # Fallback: truncate with notice
                return "[Message too long and compression failed. Key content may be missing.]"
                
        except Exception as e:
            logger.warning(f"Failed to compress single message: {e}")
            # Fallback: return error notice
            return "[Message too long and compression failed. Key content may be missing.]"

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:

                    logger.error(f"Error processing message: {e}")

                    # Send error response

                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate output (defaults to bus publish).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        key = session_key or msg.session_key
        
        # Set current message context for error notifications and token tracking
        self._current_message_context = {
            'channel': msg.channel,
            'chat_id': msg.chat_id,
            'session_key': key
        }
        
        # Set session_key in provider for token tracking
        if hasattr(self.provider, 'session_key'):
            self.provider.session_key = key
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            # Summarize conversation and save to daily notes
            summary = await self._summarize_for_daily_notes(session)
            if summary:
                memory = MemoryStore(self.workspace)
                memory.append_today(summary)
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            
            # Build greeting message with memory context
            greeting_response = await self._generate_new_session_greeting()
            
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=greeting_response,
                                  metadata=msg.metadata or {})
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        


        self._set_tool_context(msg.channel, msg.chat_id)
        if cmd == "/reset":
            # Reset session without memory consolidation (like Feishu)
            msg_count = len(session.messages)
            session.clear()
            self.sessions.save(session)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"🐈 Session reset. Cleared {msg_count} messages.")
        if cmd == "/token":
            # Show token usage statistics
            token_store = TokenStore()
            usage = token_store.get_summary()
            
            def fmt(n: int) -> str:
                return f"{n:,}"
            
            lines = ["🐈 **Token Usage Statistics**\n"]
            
            periods = [
                ("Today", usage["today"]),
                ("This Week", usage["this_week"]),
                ("This Month", usage["this_month"]),
                ("All Time", usage["all_time"]),
            ]
            
            for period_name, data in periods:
                if data["request_count"] > 0:
                    lines.append(f"**{period_name}:**")
                    lines.append(f"  Requests: {fmt(data['request_count'])}")
                    lines.append(f"  Input: {fmt(data['prompt_tokens'])} tokens")
                    lines.append(f"  Output: {fmt(data['completion_tokens'])} tokens")
                    lines.append(f"  Total: {fmt(data['total_tokens'])} tokens")
                    lines.append("")
            
            # Model breakdown for All Time
            model_usage = usage["all_time"].get("model_usage", {})
            if model_usage:
                lines.append("**By Model (All Time):**")
                sorted_models = sorted(
                    model_usage.items(),
                    key=lambda x: x[1]["total_tokens"],
                    reverse=True
                )
                for model_name, data in sorted_models:
                    lines.append(f"  {model_name}: {fmt(data['request_count'])} req, {fmt(data['total_tokens'])} tokens")
            
            if len(lines) == 1:
                lines.append("No token usage data available yet. Start a conversation to begin tracking.")
            
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))
        
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/reset — Reset session without saving memory\n/token — Show token usage statistics\n/help — Show available commands")

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        # Get history and check if compression needed
        from nanobot.session.manager import estimate_messages_tokens, estimate_tokens
        raw_history = session.get_history(max_messages=50)
        history_tokens = estimate_messages_tokens(raw_history)

        # Log current token usage
        logger.info(f"Current context: ~{history_tokens} tokens (maxWindowContext: {self.max_window_context})")

        # Check for single message exceeding limit
        max_single_msg_tokens = 0
        longest_msg_idx = -1
        for i, hist_msg in enumerate(raw_history):
            msg_tokens = estimate_tokens(hist_msg.get("content", ""))
            if msg_tokens > max_single_msg_tokens:
                max_single_msg_tokens = msg_tokens
                longest_msg_idx = i
        
        # Initialize history with raw_history
        history = raw_history
        
        # Step 1: Handle single message exceeding limit
        if max_single_msg_tokens > self.max_window_context:
            # Single message exceeds limit - use LLM to compress it
            logger.warning(f"Single message exceeds maxWindowContext: ~{max_single_msg_tokens} tokens > {self.max_window_context}, compressing with LLM...")
            
            # Compress long messages using LLM
            new_history = []
            for i, hist_msg in enumerate(history):
                msg_tokens = estimate_tokens(hist_msg.get("content", ""))
                if msg_tokens > self.max_window_context:
                    # Use LLM to compress this single message
                    content = hist_msg.get("content", "")
                    compressed_content = await self._compress_single_message(content)
                    new_history.append({
                        "role": hist_msg.get("role", "assistant"),
                        "content": compressed_content
                    })
                    logger.info(f"Compressed message {i} from {len(content)} chars to {len(compressed_content)} chars")
                else:
                    new_history.append(hist_msg)
            
            # Update history with compressed version
            history = new_history
            
            # Sync compressed content back to session.messages to persist the compression
            # This ensures next conversation uses compressed version
            session.messages = []
            for hist_msg in history:
                session.messages.append({
                    "role": hist_msg.get("role", "assistant"),
                    "content": hist_msg.get("content", ""),
                    "timestamp": hist_msg.get("timestamp", 0)
                })
            
            # Immediately save to disk to persist compression
            self.sessions.save(session)
            logger.info(f"Synced compressed messages back to session and saved to disk")
            
            # Recalculate tokens after single message compression
            history_tokens = estimate_messages_tokens(history)
            logger.info(f"After single message compression: ~{history_tokens} tokens")
        
        # Step 2: Check if total still exceeds limit after single message handling
        if history_tokens > self.max_window_context:
            # Need to compress using LLM
            logger.info(f"Context window exceeded: ~{history_tokens} tokens > {self.max_window_context}, compressing with LLM...")
            # Use custom compression: keep last 50 non-system messages for better context retention
            all_messages = session.messages
            system_msgs = [m for m in all_messages if m.get("role") == "system"]
            non_system_msgs = [m for m in all_messages if m.get("role") != "system"]
            recent_msgs = non_system_msgs[-50:] if len(non_system_msgs) > 50 else non_system_msgs
            older_msgs = non_system_msgs[:-50] if len(non_system_msgs) > 50 else []

            if older_msgs:
                # Use LLM to summarize older messages
                summary = await self._summarize_context(older_msgs)
                # Build compressed history: keep all system messages + recent messages + summary as assistant message
                history = []
                
                # Add all system messages (unchanged)
                history.extend(system_msgs)
                
                # Add summary as a user message (to indicate it's compressed context)
                history.append({
                    "role": "user",
                    "content": "[Earlier conversation context has been compressed. Key points below:]"
                })
                
                # Add the summary as assistant message
                history.append({
                    "role": "assistant",
                    "content": summary
                })
                
                # Add recent messages (unchanged)
                history.extend(recent_msgs)
                
                logger.info(f"Context compressed: {len(older_msgs)} older messages summarized into assistant message")
                logger.debug(f"Summary preview: {summary[:150]}...")
            else:
                history = raw_history
        else:
            history = raw_history

        # Build initial messages
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str) -> None:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
                metadata=msg.metadata or {},
            ))

        final_content, tools_used = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )


        # Use the existing _run_agent_loop method
        final_content, tools_used, model_name = await self._run_agent_loop(initial_messages)
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Clean up leading/trailing whitespace and newlines
        final_content = final_content.strip()
        
        # Save to session without model prefix (only keep original content)
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        # Add model prefix to final output (only for user display)
        if model_name:
            final_content = f"[{model_name}]:\n{final_content}"
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
            message_id=msg.metadata.get("message_id") if msg.metadata else None,
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        


        # Set current message context for error notifications
        self._current_message_context = {
            'channel': origin_channel,
            'chat_id': origin_chat_id
        }
        
        # Use the origin session for context

        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _, model_name = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session without model prefix
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        # Add model prefix to final output (only for user display)
        if model_name:
            final_content = f"[{model_name}]:\n{final_content}"
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _summarize_for_daily_notes(self, session) -> str:
        """Summarize conversation for daily notes using LLM."""
        if not session.messages:
            return ""

        # Format messages for LLM
        lines = []
        for m in session.messages:
            if not m.get("content"):
                continue
            role = m.get('role', 'unknown')
            content = m.get('content', '')
            lines.append(f"[{role.upper()}]: {content}")
        conversation = "\n".join(lines)

        prompt = f""" You are a memory consolidation agent.  Summarize the following conversation concisely. Focus on:
1. Main topics discussed，A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.
2. Key decisions or actions taken
3. Important information to remember

Keep it. Write in Chinese (中文), using a neutral, informative tone.

Conversation:
{conversation}

Summary (in Chinese):"""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            summary = (response.content or "").strip()
            if summary:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                return f"## Session Summary [{timestamp}]\n\n{summary}\n"
            return ""
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return ""

    async def _generate_new_session_greeting(self) -> str:
        """Generate a greeting for new session with memory context."""
        try:
            # Build system prompt
            system_prompt = self.context.build_system_prompt()
            
            # Get memory context (long-term memory + today's notes)
            memory = MemoryStore(self.workspace)
            memory_context = memory.get_memory_context()
            
            # Get today's notes
            today_notes = memory.read_today()
            if today_notes:
                memory_context = f"{memory_context}\n\n## Today's Notes\n\n{today_notes}" if memory_context else f"## Today's Notes\n\n{today_notes}"
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            if memory_context:
                messages.append({
                    "role": "user", 
                    "content": f"这是我们之前的对话记忆：\n\n{memory_context}"
                })
            
            messages.append({
                "role": "user",
                "content": "现在开始了新的对话，请打招呼"
            })
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                model=self.model,
            )
            
            greeting = (response.content or "新会话已开始，有什么可以帮助你的吗？").strip()
            return greeting
            
        except Exception as e:
            logger.error(f"Failed to generate new session greeting: {e}")
            return "新会话已开始，有什么可以帮助你的吗？"



    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used, model_name).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        model_name = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Capture model name from response
            if response.model:
                model_name = response.model

            if response.has_tool_calls:
                # Send LLM's intermediate response (combine reasoning and content)
                if self._tool_output_callback and self._current_message_context:
                    try:
                        parts = []
                        if response.reasoning_content:
                            parts.append(response.reasoning_content.strip())
                        if response.content:
                            parts.append(response.content.strip())
                        if parts:
                            combined = "    ".join(parts)
                            await self._tool_output_callback(
                                self._current_message_context.get('channel'),
                                self._current_message_context.get('chat_id'),
                                "🤔",
                                f"🤔 Thinking...\n{combined}".strip()
                            )
                    except Exception as e:
                        logger.warning(f"LLM intermediate output callback failed: {e}")
                
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    
                    # Send tool call notification
                    if self._tool_output_callback and self._current_message_context:
                        try:
                            await self._tool_output_callback(
                                self._current_message_context.get('channel'),
                                self._current_message_context.get('chat_id'),
                                tool_call.name,
                                f"🔧 Calling {tool_call.name}..."
                            )
                        except Exception as e:
                            logger.warning(f"Tool output callback failed: {e}")
                    
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    
                    # Send tool result notification
                    if self._tool_output_callback and self._current_message_context:
                        try:
                            result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                            await self._tool_output_callback(
                                self._current_message_context.get('channel'),
                                self._current_message_context.get('chat_id'),
                                tool_call.name,
                                f"✅ {tool_call.name} completed:\n{result_preview}"
                            )
                        except Exception as e:
                            logger.warning(f"Tool output callback failed: {e}")
                    
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Based on the tool results above, decide your next action. If the task is complete, provide your final response to the user. If you need more information or actions, continue with the next step. Consider: 1) Did the tools return what you expected? 2) Do you need to make additional tool calls? 3) Is there any error that needs to be addressed?"})
            else:
                final_content = response.content
                break

        return final_content, tools_used, model_name

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate output.
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
