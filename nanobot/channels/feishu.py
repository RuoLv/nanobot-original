"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import re
import threading
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

if False:  # TYPE_CHECKING
    from nanobot.session.manager import SessionManager

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        DeleteMessageReactionRequest,
        Emoji,
        GetMessageResourceRequest,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Constants
MAX_PROCESSED_MESSAGES = 1000
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# Log level standards:
# - ERROR: Critical failures that prevent operation (SDK missing, config errors, message processing failures)
# - WARNING: Non-critical issues that don't stop operation (WebSocket errors, reaction failures, client not initialized)
# - INFO: Important operational events (startup, shutdown, connection status)
# - DEBUG: Detailed operation info (message sent/received, reactions added)

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message content.
    
    Supports two formats:
    1. Direct format: {"title": "...", "content": [...]}
    2. Localized format: {"zh_cn": {"title": "...", "content": [...]}}
    """
    def extract_from_lang(lang_content: dict) -> str | None:
        if not isinstance(lang_content, dict):
            return None
        title = lang_content.get("title", "")
        content_blocks = lang_content.get("content", [])
        if not isinstance(content_blocks, list):
            return None
        text_parts = []
        if title:
            text_parts.append(title)
        for block in content_blocks:
            if not isinstance(block, list):
                continue
            for element in block:
                if isinstance(element, dict):
                    tag = element.get("tag")
                    if tag == "text":
                        text_parts.append(element.get("text", ""))
                    elif tag == "a":
                        text_parts.append(element.get("text", ""))
                    elif tag == "at":
                        text_parts.append(f"@{element.get('user_name', 'user')}")
        return " ".join(text_parts).strip() if text_parts else None
    
    # Try direct format first
    if "content" in content_json:
        result = extract_from_lang(content_json)
        if result:
            return result
    
    # Try localized format
    for lang_key in ("zh_cn", "en_us", "ja_jp"):
        lang_content = content_json.get(lang_key)
        result = extract_from_lang(lang_content)
        if result:
            return result
    
    return ""


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(
        self,
        config: FeishuConfig,
        bus: MessageBus,
    ):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tool_output_callback: Any = None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread with reconnect loop
        def run_ws() -> None:
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning(f"Feishu WebSocket error: {e}")
                if self._running:
                    import time; time.sleep(5)
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> str | None:
        """Sync helper for adding reaction (runs in thread pool). Returns reaction_id on success."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
                return None
            
            reaction_id = response.data.reaction_id if response.data else None
            logger.debug(f"Added {emoji_type} reaction to message {message_id}, reaction_id: {reaction_id}")
            return reaction_id
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")
            return None

    async def _add_reaction(self, message_id: str, emoji_type: str = "THUMBSUP") -> str | None:
        """
        Add a reaction emoji to a message (non-blocking). Returns reaction_id on success.
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART
        """
        if not self._client or not Emoji:
            return None
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    def _delete_reaction_sync(self, message_id: str, reaction_id: str) -> None:
        """Sync helper for deleting reaction (runs in thread pool)."""
        try:
            request = DeleteMessageReactionRequest.builder() \
                .message_id(message_id) \
                .reaction_id(reaction_id) \
                .build()
            
            response = self._client.im.v1.message_reaction.delete(request)
            
            if not response.success():
                logger.warning(f"Failed to delete reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Deleted reaction from message {message_id}")
        except Exception as e:
            logger.warning(f"Error deleting reaction: {e}")
    
    async def _delete_reaction(self, message_id: str, reaction_id: str) -> None:
        """Delete a reaction from a message (non-blocking)."""
        if not self._client:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._delete_reaction_sync, message_id, reaction_id)
    
    def _create_message(self, receive_id: str, receive_id_type: str, msg_type: str, content: str) -> Any:
        """Create a message through Feishu API.
        
        Args:
            receive_id: Receiver ID (chat_id or open_id)
            receive_id_type: ID type ("chat_id" or "open_id")
            msg_type: Message type ("text", "image", "file", "interactive")
            content: Message content (JSON string for image/file, plain text for text)
            
        Returns:
            Response object from Feishu API
        """
        request = CreateMessageRequest.builder() \
            .receive_id_type(receive_id_type) \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            ).build()
        
        return self._client.im.v1.message.create(request)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    _CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into div/markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()]
            if before.strip():
                elements.extend(self._split_headings(before))
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:]
        if remaining.strip():
            elements.extend(self._split_headings(remaining))
        return elements or [{"tag": "markdown", "content": content}]

    def _split_headings(self, content: str) -> list[dict]:
        """Split content by headings, converting headings to div elements."""
        protected = content
        code_blocks = []
        for m in self._CODE_BLOCK_RE.finditer(content):
            code_blocks.append(m.group(1))
            protected = protected.replace(m.group(1), f"\x00CODE{len(code_blocks)-1}\x00", 1)

        elements = []
        last_end = 0
        for m in self._HEADING_RE.finditer(protected):
            before = protected[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            level = len(m.group(1))
            text = m.group(2).strip()
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**{text}**",
                },
            })
            last_end = m.end()
        remaining = protected[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})

        for i, cb in enumerate(code_blocks):
            for el in elements:
                if el.get("tag") == "markdown":
                    el["content"] = el["content"].replace(f"\x00CODE{i}\x00", cb)

        return elements or [{"tag": "markdown", "content": content}]

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to Feishu."""
        # Check if this is a tool output message
        if msg.metadata.get("is_tool_output"):
            await self._send_tool_output(msg)
            return
        
        # Determine receive_id_type based on chat_id format
        # open_id starts with "ou_", chat_id starts with "oc_"
        if msg.chat_id.startswith("oc_"):
            receive_id_type = "chat_id"
        else:
            receive_id_type = "open_id"
        
        # Check message type and route accordingly
        msg_type = msg.metadata.get("type", "text")
        
        if msg_type == "interactive":
            # Interactive card message
            sent_message_id = await self._send_interactive(msg, receive_id_type)
        elif msg.media:
            # Media (image/file) message
            sent_message_id = await self._send_media(msg, receive_id_type)
        else:
            # Text message
            sent_message_id = await self._send_text(msg, receive_id_type, msg.content)
        
        # Store message_id for reaction updates
        if sent_message_id:
            msg.message_id = sent_message_id
        
        # Update reactions on the original user message: remove Typing, add DONE
        typing_reaction_id = msg.metadata.get("typing_reaction_id")
        original_message_id = msg.metadata.get("message_id")
        if typing_reaction_id and original_message_id:
            await self._delete_reaction(original_message_id, typing_reaction_id)
        if original_message_id:
            await self._add_reaction(original_message_id, "DONE")
    
    async def _send_tool_output(self, msg: OutboundMessage) -> None:
        """Send real-time tool output to user."""
        try:
            tool_name = msg.metadata.get("tool_name", "tool")
            output = msg.content
            
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            # Build content with tool output
            content = json.dumps({
                "text": f"{output}"
            }, ensure_ascii=False)
            
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("text")
                    .content(content)
                    .build()
                ).build()
            
            response = self._client.im.v1.message.create(request)
            
            if not response.success():
                logger.warning(f"Failed to send tool output: {response.msg}")
            else:
                logger.debug(f"Tool output sent to {msg.chat_id}")
                
        except Exception as e:
            logger.error(f"Error sending tool output: {e}")
    
    def set_tool_output_callback(self, callback) -> None:
        """Set callback for real-time tool output."""
        self._tool_output_callback = callback
    
    async def _send_interactive(self, msg: OutboundMessage, receive_id_type: str) -> str | None:
        """Send interactive card message. Returns message_id on success."""
        elements = self._build_card_elements(msg.content)
        card = {
            "config": {"wide_screen_mode": True},
            "elements": elements,
        }
        content = json.dumps(card, ensure_ascii=False)
        
        response = self._create_message(msg.chat_id, receive_id_type, "interactive", content)
        
        if not response.success():
            logger.error(f"Failed to send interactive message: {response.msg}")
            return None
        
        message_id = response.data.message_id if response.data else None
        logger.debug(f"Interactive message sent to {msg.chat_id}, message_id: {message_id}")
        return message_id
    
    async def _send_media(self, msg: OutboundMessage, receive_id_type: str) -> str | None:
        """Send media message (image or file) through Feishu. Returns message_id on success."""
        if not msg.media:
            return None
        
        media_type = msg.media.get("type")
        media_url = msg.media.get("url")
        media_path = msg.media.get("path")
        media_content = msg.media.get("content")  # bytes
        
        try:
            if media_type in ["image", "file"]:
                return await self._upload_and_send_media(msg, receive_id_type, media_type, media_path, media_content)
            else:
                # Fallback to text message with media URL
                content = f"{msg.content}\n\n[Media: {media_url}]" if msg.content else f"[Media: {media_url}]"
                return await self._send_text(msg, receive_id_type, content)
        except Exception as e:
            logger.error(f"Error sending media: {e}")
            # Fallback to text message
            return await self._send_text(msg, receive_id_type, msg.content or "[Failed to send media]")
    
    async def _upload_and_send_media(
        self,
        msg: OutboundMessage,
        receive_id_type: str,
        media_type: str,
        media_path: str | None,
        media_content: bytes | None
    ) -> str | None:
        """Upload and send media (image or file) through Feishu. Returns message_id on success."""
        # Get file as file-like object
        if media_path:
            file_obj = open(media_path, "rb")
            file_name = media_path.split("/")[-1]
        elif media_content:
            file_obj = BytesIO(media_content)
            file_name = msg.media.get("name", "file") if media_type == "file" else "image"
        else:
            raise ValueError("No media content provided")
        
        try:
            # Upload media
            if media_type == "image":
                resource_key = await self._upload_image(file_obj)
                msg_type = "image"
                content = json.dumps({"image_key": resource_key})
            else:  # file
                resource_key = await self._upload_file(file_obj, file_name)
                msg_type = "file"
                content = json.dumps({"file_key": resource_key})
            
            # Send message
            response = self._create_message(msg.chat_id, receive_id_type, msg_type, content)
            
            if not response.success():
                raise Exception(f"Failed to send {media_type}: {response.msg}")
            
            message_id = response.data.message_id if response.data else None
            logger.debug(f"{media_type.capitalize()} sent to {msg.chat_id}, message_id: {message_id}")
            return message_id
            
        finally:
            # Close file if we opened it
            if media_path:
                file_obj.close()
    
    async def _upload_image(self, image_file: Any) -> str:
        """Upload image and return image_key.
        
        Args:
            image_file: File-like object for image
            
        Returns:
            Image key
        """
        request = CreateImageRequest.builder() \
            .request_body(
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(image_file)
                .build()
            ).build()
        
        response = self._client.im.v1.image.create(request)
        
        if not response.success():
            raise Exception(f"Failed to upload image: {response.msg}")
        
        return response.data.image_key
    
    async def _upload_file(self, file_obj: Any, file_name: str) -> str:
        """Upload file and return file_key.
        
        Args:
            file_obj: File-like object
            file_name: File name
            
        Returns:
            File key
        """
        request = CreateFileRequest.builder() \
            .request_body(
                CreateFileRequestBody.builder()
                .file_type("stream")
                .file_name(file_name)
                .file(file_obj)
                .build()
            ).build()
        
        response = self._client.im.v1.file.create(request)
        
        if not response.success():
            raise Exception(f"Failed to upload file: {response.msg}")
        
        return response.data.file_key
    
    async def _send_text(self, msg: OutboundMessage, receive_id_type: str, content: str) -> str | None:
        """Send a simple text message. Returns message_id on success."""
        elements = self._build_card_elements(content)
        card = {
            "config": {"wide_screen_mode": True},
            "elements": elements,
        }
        card_content = json.dumps(card, ensure_ascii=False)
        
        response = self._create_message(msg.chat_id, receive_id_type, "interactive", card_content)
        
        if not response.success():
            logger.error(f"Failed to send text: {response.msg}")
            return None
        
        message_id = response.data.message_id if response.data else None
        logger.debug(f"Text sent to {msg.chat_id}, message_id: {message_id}")
        return message_id
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    def _download_media(self, message_id: str, content: str, msg_type: str) -> tuple[str, list[str]]:
        """Download media (image or file) from Feishu message with retry mechanism.
        
        Args:
            message_id: Message ID
            content: Message content JSON string
            msg_type: Message type ("image" or "file")
            
        Returns:
            Tuple of (content string, media paths list)
        """
        media_paths = []
        
        try:
            content_data = json.loads(content)
            resource_key = content_data.get("image_key" if msg_type == "image" else "file_key", "")
            file_name = content_data.get("file_name", f"{msg_type}_{resource_key}") if msg_type == "file" else f"image_{resource_key}.jpg"
            
            logger.info(f"Downloading {msg_type}: {file_name} (key: {resource_key})")
            
            # Download with retry mechanism
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    logger.debug(f"Download attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} for {file_name}")
                    
                    # Create message resource request
                    req = GetMessageResourceRequest.builder() \
                        .message_id(message_id) \
                        .file_key(resource_key) \
                        .type(msg_type) \
                        .build()
                    
                    # Execute request
                    resp = self._client.im.v1.message_resource.get(req)
                    
                    if not resp.success():
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            logger.warning(f"Download failed for {file_name}: {resp.msg}, retrying...")
                            import time
                            time.sleep(RETRY_DELAY)
                            continue
                        logger.error(f"Failed to download {file_name} after {MAX_RETRY_ATTEMPTS} attempts: {resp.msg}")
                        return f"[User sent a {msg_type}: {file_name} (key: {resource_key})]\nFailed to get download URL: {resp.msg}", media_paths
                    
                    # Check file size
                    file_data = resp.file.read()
                    file_size_mb = len(file_data) / (1024 * 1024)
                    logger.debug(f"Downloaded {file_name}: {file_size_mb:.2f}MB")
                    
                    if len(file_data) > MAX_FILE_SIZE:
                        logger.warning(f"File {file_name} too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE // (1024*1024)}MB")
                        return f"[User sent a {msg_type}: {file_name} (key: {resource_key})]\nFile too large (>{MAX_FILE_SIZE // (1024*1024)}MB)", media_paths
                    
                    # Create media directory if not exists
                    media_dir = Path.home() / ".nanobot" / "media"
                    media_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save file
                    file_path = media_dir / file_name
                    with open(file_path, 'wb') as f:
                        f.write(file_data)
                    
                    logger.info(f"Successfully saved {msg_type}: {file_path}")
                    
                    content = f"[User sent a {msg_type}: {file_name} (key: {resource_key})]\nSaved to: {file_path}"
                    media_paths.append(str(file_path))
                    
                    return content, media_paths
                    
                except Exception as e:
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        logger.warning(f"Download error for {file_name}: {str(e)}, retrying...")
                        import time
                        time.sleep(RETRY_DELAY)
                        continue
                    logger.error(f"Failed to download {file_name} after {MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                    return f"[User sent a {msg_type}: {file_name}]\nError downloading {msg_type}: {str(e)}", media_paths
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {msg_type} message content")
            return f"[{msg_type}]", media_paths
        except Exception as e:
            logger.error(f"Unexpected error downloading {msg_type}: {str(e)}")
            return f"[User sent a {msg_type}]\nError downloading {msg_type}: {str(e)}", media_paths
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds limit
            while len(self._processed_message_ids) > MAX_PROCESSED_MESSAGES:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen"
            typing_reaction_id = await self._add_reaction(message_id, "Typing")
            
            # Parse message content
            media_paths = []
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            elif msg_type == "post":
                try:
                    content_json = json.loads(message.content)
                    content = _extract_post_text(content_json)
                except (json.JSONDecodeError, TypeError):
                    content = message.content or ""
            elif msg_type in ["image", "file"]:
                # Image or file message - parse key and download
                content, media_paths = self._download_media(message_id, message.content, msg_type)
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

            if not content:
                return

            # Forward to message bus (including /reset command, handled by AgentLoop)
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "typing_reaction_id": typing_reaction_id,
                }
            )

        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")


