"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import os
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
        GetFileRequest,
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


def _extract_share_card_content(content_json: dict, msg_type: str) -> str:
    """Extract text representation from share cards and interactive messages."""
    parts = []

    if msg_type == "share_chat":
        parts.append(f"[shared chat: {content_json.get('chat_id', '')}]")
    elif msg_type == "share_user":
        parts.append(f"[shared user: {content_json.get('user_id', '')}]")
    elif msg_type == "interactive":
        parts.extend(_extract_interactive_content(content_json))
    elif msg_type == "share_calendar_event":
        parts.append(f"[shared calendar event: {content_json.get('event_key', '')}]")
    elif msg_type == "system":
        parts.append("[system message]")
    elif msg_type == "merge_forward":
        parts.append("[merged forward messages]")

    return "\n".join(parts) if parts else f"[{msg_type}]"


def _extract_interactive_content(content: dict) -> list[str]:
    """Recursively extract text and links from interactive card content."""
    parts = []
    
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []

    if not isinstance(content, dict):
        return parts

    if "title" in content:
        title = content["title"]
        if isinstance(title, dict):
            title_content = title.get("content", "") or title.get("text", "")
            if title_content:
                parts.append(f"title: {title_content}")
        elif isinstance(title, str):
            parts.append(f"title: {title}")

    for element in content.get("elements", []) if isinstance(content.get("elements"), list) else []:
        parts.extend(_extract_element_content(element))

    card = content.get("card", {})
    if card:
        parts.extend(_extract_interactive_content(card))

    header = content.get("header", {})
    if header:
        header_title = header.get("title", {})
        if isinstance(header_title, dict):
            header_text = header_title.get("content", "") or header_title.get("text", "")
            if header_text:
                parts.append(f"title: {header_text}")
    
    return parts


def _extract_element_content(element: dict) -> list[str]:
    """Extract content from a single card element."""
    parts = []
    
    if not isinstance(element, dict):
        return parts
    
    tag = element.get("tag", "")
    
    if tag in ("markdown", "lark_md"):
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "div":
        text = element.get("text", {})
        if isinstance(text, dict):
            text_content = text.get("content", "") or text.get("text", "")
            if text_content:
                parts.append(text_content)
        elif isinstance(text, str):
            parts.append(text)
        for field in element.get("fields", []):
            if isinstance(field, dict):
                field_text = field.get("text", {})
                if isinstance(field_text, dict):
                    c = field_text.get("content", "")
                    if c:
                        parts.append(c)

    elif tag == "a":
        href = element.get("href", "")
        text = element.get("text", "")
        if href:
            parts.append(f"link: {href}")
        if text:
            parts.append(text)

    elif tag == "button":
        text = element.get("text", {})
        if isinstance(text, dict):
            c = text.get("content", "")
            if c:
                parts.append(c)
        url = element.get("url", "") or element.get("multi_url", {}).get("url", "")
        if url:
            parts.append(f"link: {url}")

    elif tag == "img":
        alt = element.get("alt", {})
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")

    elif tag == "note":
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))

    elif tag == "column_set":
        for col in element.get("columns", []):
            for ce in col.get("elements", []):
                parts.extend(_extract_element_content(ce))

    elif tag == "plain_text":
        content = element.get("content", "")
        if content:
            parts.append(content)

    else:
        for ne in element.get("elements", []):
            parts.extend(_extract_element_content(ne))
    
    return parts


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
                    logger.warning("Feishu WebSocket error: {}", e)
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
                logger.warning("Error stopping WebSocket client: {}", e)
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
                logger.warning("Failed to add reaction: code={}, msg={}", response.code, response.msg)
                return None
            
            reaction_id = response.data.reaction_id if response.data else None
            logger.debug("Added {} reaction to message {}", emoji_type, message_id)
            return reaction_id
        except Exception as e:
            logger.warning("Error adding reaction: {}", e)
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

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
            return "image"
        if ext in ("mp3", "m4a", "wav", "aac", "ogg", "flac"):
            return "audio"
        if ext in ("mp4", "avi", "mov", "mkv", "webm"):
            return "video"
        return "file"

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
        
        logger.debug(f"[Feishu send] Starting send, chat_id={msg.chat_id}, msg_type={msg_type}, has_media={msg.media is not None}, content={msg.content[:50] if msg.content else 'None'}...")
        
        if msg_type == "interactive":
            # Interactive card message
            logger.debug(f"[Feishu send] Sending interactive message")
            sent_message_id = await self._send_interactive(msg, receive_id_type)
        elif msg.media:
            # Media (image/file) message - support multiple files
            logger.debug(f"[Feishu send] Sending media message")
            sent_message_id = await self._send_media(msg, receive_id_type)
        else:
            # Text message
            logger.debug(f"[Feishu send] Sending text message")
            sent_message_id = await self._send_text(msg, receive_id_type, msg.content)
        
        logger.debug(f"[Feishu send] Message sent, sent_message_id={sent_message_id}")
        
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
        """Send real-time tool output to user with styling."""
        try:
            tool_name = msg.metadata.get("tool_name", "tool")
            output = msg.content or ""
            
            if not output:
                return
            
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            # Use Feishu post message with code_block for small gray text effect
            content = json.dumps({
                "zh_cn": {
                    "title": f"🔧 {tool_name}",
                    "content": [
                        [
                            {
                                "tag": "code_block",
                                "text": output
                            }
                        ]
                    ]
                }
            }, ensure_ascii=False)
            
            response = self._create_message(msg.chat_id, receive_id_type, "interactive", content)
            
            if not response.success():
                logger.warning(f"Failed to send interactive message: {response.msg}")
                return None
            
            message_id = response.data.message_id if response.data else None
            logger.debug(f"Interactive message sent to {msg.chat_id}, message_id: {message_id}")
            return message_id
        
        except Exception as e:
            logger.error(f"Error sending interactive message: {e}")
            return None

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}
    _AUDIO_EXTS = {".opus"}
    _FILE_TYPE_MAP = {
        ".opus": "opus", ".mp4": "mp4", ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
        ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
    }

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        try:
            with open(file_path, "rb") as f:
                request = CreateImageRequest.builder() \
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    ).build()
                response = self._client.im.v1.image.create(request)
                if response.success():
                    image_key = response.data.image_key
                    logger.debug("Uploaded image {}: {}", os.path.basename(file_path), image_key)
                    return image_key
                else:
                    logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading image {}: {}", file_path, e)
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                request = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(f)
                        .build()
                    ).build()
                response = self._client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    logger.debug("Uploaded file {}: {}", file_name, file_key)
                    return file_key
                else:
                    logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading file {}: {}", file_path, e)
            return None
        ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
    }

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload an image to Feishu and return the image_key."""
        try:
            with open(file_path, "rb") as f:
                request = CreateImageRequest.builder() \
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    ).build()
                response = self._client.im.v1.image.create(request)
                if response.success():
                    image_key = response.data.image_key
                    logger.debug("Uploaded image {}: {}", os.path.basename(file_path), image_key)
                    return image_key
                else:
                    logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading image {}: {}", file_path, e)
            return None

    def _upload_file_sync(self, file_path: str) -> str | None:
        """Upload a file to Feishu and return the file_key."""
        ext = os.path.splitext(file_path)[1].lower()
        file_type = self._FILE_TYPE_MAP.get(ext, "stream")
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                request = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(file_name)
                        .file(f)
                        .build()
                    ).build()
                response = self._client.im.v1.file.create(request)
                if response.success():
                    file_key = response.data.file_key
                    logger.debug("Uploaded file {}: {}", file_name, file_key)
                    return file_key
                else:
                    logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
                    return None
        except Exception as e:
            logger.error("Error uploading file {}: {}", file_path, e)
            return None

    def _download_image_sync(self, message_id: str, image_key: str) -> tuple[bytes | None, str | None]:
        """Download an image from Feishu message by message_id and image_key."""
        try:
            request = GetMessageResourceRequest.builder() \
                .message_id(message_id) \
                .file_key(image_key) \
                .type("image") \
                .build()
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                file_data = response.file
                # GetMessageResourceRequest returns BytesIO, need to read bytes
                if hasattr(file_data, 'read'):
                    file_data = file_data.read()
                return file_data, response.file_name
            else:
                logger.error("Failed to download image: code={}, msg={}", response.code, response.msg)
                return None, None
        except Exception as e:
            logger.error("Error downloading image {}: {}", image_key, e)
            return None, None

    def _download_file_sync(self, file_key: str) -> tuple[bytes | None, str | None]:
        """Download a file from Feishu by file_key."""
        try:
            request = GetFileRequest.builder().file_key(file_key).build()
            response = self._client.im.v1.file.get(request)
            if response.success():
                return response.file, response.file_name
            else:
                logger.error("Failed to download file: code={}, msg={}", response.code, response.msg)
                return None, None
        except Exception as e:
            logger.error("Error downloading file {}: {}", file_key, e)
            return None, None

    async def _download_and_save_media(
        self,
        msg_type: str,
        content_json: dict,
        message_id: str | None = None
    ) -> tuple[str | None, str]:
        """
        Download media from Feishu and save to local disk.

        Returns:
            (file_path, content_text) - file_path is None if download failed
        """
        loop = asyncio.get_running_loop()
        media_dir = Path.home() / ".nanobot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        data, filename = None, None

        if msg_type == "image":
            image_key = content_json.get("image_key")
            if image_key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_image_sync, message_id, image_key
                )
                if not filename:
                    filename = f"{image_key[:16]}.jpg"

        elif msg_type in ("audio", "file"):
            file_key = content_json.get("file_key")
            if file_key:
                data, filename = await loop.run_in_executor(
                    None, self._download_file_sync, file_key
                )
                if not filename:
                    ext = ".opus" if msg_type == "audio" else ""
                    filename = f"{file_key[:16]}{ext}"

        if data and filename:
            file_path = media_dir / filename
            file_path.write_bytes(data)
            logger.debug("Downloaded {} to {}", msg_type, file_path)
            return str(file_path), f"[{msg_type}: {filename}]"

        return None, f"[{msg_type}: download failed]"

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content: str) -> bool:
        """Send a single message (text/image/file/interactive) synchronously."""
        try:
            request = CreateMessageRequest.builder() \
                .receive_id_type(receive_id_type) \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ).build()
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu {} message: code={}, msg={}, log_id={}",
                    msg_type, response.code, response.msg, response.get_log_id()
                )
                return False
            logger.debug("Feishu {} message sent to {}", msg_type, receive_id)
            return True
        except Exception as e:
            logger.error("Error sending Feishu {} message: {}", msg_type, e)
            return False
    
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
        """Send media message (image/file) through Feishu. Returns message_id on success."""
        logger.debug(f"[Feishu _send_media] Starting media send, chat_id={msg.chat_id}, receive_id_type={receive_id_type}")
        logger.debug(f"[Feishu _send_media] msg.media type: {type(msg.media)}, value: {msg.media}")
        
        if not msg.media:
            logger.warning(f"[Feishu _send_media] No media found in message")
            return None
        
        # Support both old dict format and new list format
        media_list = msg.media if isinstance(msg.media, list) else [msg.media]
        logger.debug(f"[Feishu _send_media] Media list length: {len(media_list)}")
        
        last_message_id = None
        
        # Send each media file
        for idx, media_item in enumerate(media_list):
            logger.debug(f"[Feishu _send_media] Processing media item {idx+1}/{len(media_list)}: {media_item}")
            try:
                # Handle both dict format and string path format
                if isinstance(media_item, dict):
                    media_path = media_item.get("path")
                    media_content = media_item.get("content")
                    media_type = media_item.get("type")
                    logger.debug(f"[Feishu _send_media] Dict format - path={media_path}, type={media_type}, has_content={media_content is not None}")
                else:
                    # String path - auto-detect type
                    media_path = media_item
                    media_content = None
                    media_type = self._get_media_type(media_path) if media_path else "file"
                    logger.debug(f"[Feishu _send_media] String format - path={media_path}, auto_type={media_type}")
                
                if not media_path and not media_content:
                    logger.warning(f"[Feishu _send_media] Skipping invalid media item: {media_item}")
                    continue
                
                # Upload and send media
                logger.info(f"[Feishu _send_media] Uploading {media_type}: {media_path or 'content'}")
                message_id = await self._upload_and_send_media(
                    msg, receive_id_type, media_type, media_path, media_content
                )
                
                if message_id:
                    logger.info(f"[Feishu _send_media] ✓ Media sent successfully, message_id={message_id}")
                    last_message_id = message_id
                else:
                    logger.warning(f"[Feishu _send_media] ✗ Media send returned None")
                    
            except Exception as e:
                logger.error(f"[Feishu _send_media] Error sending media item {idx+1}: {e}", exc_info=True)
                # Continue with next media item
        
        # Send text content if present
        if msg.content and msg.content != "[empty message]":
            logger.debug(f"[Feishu _send_media] Sending text content: {msg.content[:100]}...")
            text_message_id = await self._send_text(msg, receive_id_type, msg.content)
            if text_message_id:
                logger.debug(f"[Feishu _send_media] Text sent, message_id={text_message_id}")
                last_message_id = text_message_id
        
        logger.debug(f"[Feishu _send_media] Completed, last_message_id={last_message_id}")
        return last_message_id
    
    async def _upload_and_send_media(
        self,
        msg: OutboundMessage,
        receive_id_type: str,
        media_type: str,
        media_path: str | None,
        media_content: bytes | None
    ) -> str | None:
        """Upload and send media (image or file) through Feishu. Returns message_id on success."""
        logger.debug(f"[Feishu _upload_and_send_media] Starting, media_type={media_type}, media_path={media_path}, has_content={media_content is not None}")
        
        # Get file as file-like object
        if media_path:
            logger.debug(f"[Feishu _upload_and_send_media] Opening file: {media_path}")
            file_obj = open(media_path, "rb")
            file_name = media_path.split("/")[-1]
        elif media_content:
            logger.debug(f"[Feishu _upload_and_send_media] Using content bytes, size={len(media_content)}")
            file_obj = BytesIO(media_content)
            file_name = msg.media.get("name", "file") if media_type == "file" else "image"
        else:
            logger.error(f"[Feishu _upload_and_send_media] No media content provided")
            raise ValueError("No media content provided")
        
        try:
            # Upload media
            if media_type == "image":
                logger.debug(f"[Feishu _upload_and_send_media] Uploading image...")
                resource_key = await self._upload_image(file_obj)
                logger.debug(f"[Feishu _upload_and_send_media] Image uploaded, image_key={resource_key}")
                msg_type = "image"
                content = json.dumps({"image_key": resource_key})
            else:  # file
                logger.debug(f"[Feishu _upload_and_send_media] Uploading file: {file_name}")
                resource_key = await self._upload_file(file_obj, file_name)
                logger.debug(f"[Feishu _upload_and_send_media] File uploaded, file_key={resource_key}")
                msg_type = "file"
                content = json.dumps({"file_key": resource_key})
            
            # Send message
            logger.debug(f"[Feishu _upload_and_send_media] Creating message, msg_type={msg_type}, chat_id={msg.chat_id}")
            response = self._create_message(msg.chat_id, receive_id_type, msg_type, content)
            
            if not response.success():
                logger.error(f"[Feishu _upload_and_send_media] Failed to send {media_type}: {response.msg}")
                raise Exception(f"Failed to send {media_type}: {response.msg}")
            
            message_id = response.data.message_id if response.data else None
            logger.debug(f"[Feishu _upload_and_send_media] {media_type.capitalize()} sent to {msg.chat_id}, message_id: {message_id}")
            return message_id
            
        finally:
            # Close file if we opened it
            if media_path:
                file_obj.close()
                logger.debug(f"[Feishu _upload_and_send_media] File closed")
    
    async def _upload_image(self, image_file: Any) -> str:
        """Upload image and return image_key.
        
        Args:
            image_file: File-like object for image
            
        Returns:
            Image key
        """
        logger.debug(f"[Feishu _upload_image] Creating upload request...")
        request = CreateImageRequest.builder() \
            .request_body(
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(image_file)
                .build()
            ).build()
        
        logger.debug(f"[Feishu _upload_image] Sending upload request to Feishu API...")
        response = self._client.im.v1.image.create(request)
        
        logger.debug(f"[Feishu _upload_image] Response success={response.success()}, msg={response.msg}")
        
        if not response.success():
            logger.error(f"[Feishu _upload_image] Failed to upload image: {response.msg}")
            raise Exception(f"Failed to upload image: {response.msg}")
        
        image_key = response.data.image_key
        logger.debug(f"[Feishu _upload_image] Image uploaded successfully, image_key={image_key}")
        return image_key
    
    async def _upload_file(self, file_obj: Any, file_name: str) -> str:
        """Upload file and return file_key.
        
        Args:
            file_obj: File-like object
            file_name: File name
            
        Returns:
            File key
        """
        logger.debug(f"[Feishu _upload_file] Creating upload request, file_name={file_name}...")
        request = CreateFileRequest.builder() \
            .request_body(
                CreateFileRequestBody.builder()
                .file_type("stream")
                .file_name(file_name)
                .file(file_obj)
                .build()
            ).build()
        
        logger.debug(f"[Feishu _upload_file] Sending upload request to Feishu API...")
        response = self._client.im.v1.file.create(request)
        
        logger.debug(f"[Feishu _upload_file] Response success={response.success()}, msg={response.msg}")
        
        if not response.success():
            logger.error(f"[Feishu _upload_file] Failed to upload file: {response.msg}")
            raise Exception(f"Failed to upload file: {response.msg}")
        
        file_key = response.data.file_key
        logger.debug(f"[Feishu _upload_file] File uploaded successfully, file_key={file_key}")
        return file_key
    
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
                    msg_type, response.code, response.msg, response.get_log_id()
                )
                return False
            logger.debug("Feishu {} message sent to {}", msg_type, receive_id)
            return True
        except Exception as e:
            logger.error("Error sending Feishu {} message: {}", msg_type, e)
            return False

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu, including media (images/files) if present."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
            loop = asyncio.get_running_loop()

            for file_path in msg.media:
                if not os.path.isfile(file_path):
                    logger.warning("Media file not found: {}", file_path)
                    continue
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    key = await loop.run_in_executor(None, self._upload_image_sync, file_path)
                    if key:
                        await loop.run_in_executor(
                            None, self._send_message_sync,
                            receive_id_type, msg.chat_id, "image", json.dumps({"image_key": key}, ensure_ascii=False),
                        )
                else:
                    key = await loop.run_in_executor(None, self._upload_file_sync, file_path)
                    if key:
                        media_type = "audio" if ext in self._AUDIO_EXTS else "file"
                        await loop.run_in_executor(
                            None, self._send_message_sync,
                            receive_id_type, msg.chat_id, media_type, json.dumps({"file_key": key}, ensure_ascii=False),
                        )

            if msg.content and msg.content.strip():
                card = {"config": {"wide_screen_mode": True}, "elements": self._build_card_elements(msg.content)}
                await loop.run_in_executor(
                    None, self._send_message_sync,
                    receive_id_type, msg.chat_id, "interactive", json.dumps(card, ensure_ascii=False),
                )

        except Exception as e:
            logger.error("Error sending Feishu message: {}", e)
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
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
            typing_reaction_id = await self._add_reaction(message_id, "Typing")
            
            # Parse message content
            media_paths = []
            # Add reaction
            await self._add_reaction(message_id, "THUMBSUP")
            # Parse content
            content_parts = []
            media_paths = []

            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}

                if text:
                    content_parts.append(text)

            elif msg_type == "post":
                chat_id=reply_to,
                media=media_paths,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "typing_reaction_id": typing_reaction_id,
                }
            )

        except Exception as e:
