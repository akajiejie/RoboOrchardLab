# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from dataclasses import dataclass
from typing import IO, Any, Iterator, Literal, Optional, Sequence

from mcap.exceptions import McapError
from mcap.reader import (
    McapReader as _McapReader,
    make_reader as mcap_make_reader,
)
from mcap.records import (
    Channel as McapChannel,
    Message as McapMessage,
    Schema as McapSchema,
)
from robo_orchard_core.utils.config import Config

from robo_orchard_lab.dataset.mcap.decoder import McapDecoderContext

__all__ = [
    "McapMessageTuple",
    "McapDecodedMessageTuple",
    "McapMessagesTuple",
    "McapMessageBatch",
    "MakeIterMsgArgs",
    "McapReader",
]


@dataclass
class McapMessageTuple:
    schema: Optional[McapSchema]
    channel: McapChannel
    message: McapMessage

    def decode(self, decoder_ctx: McapDecoderContext) -> Any:
        """Decode the message using the provided decoder context."""
        return decoder_ctx.decode_message(
            message_encoding=self.channel.message_encoding,
            message=self.message,
            schema=self.schema,
        )


@dataclass
class McapDecodedMessageTuple(McapMessageTuple):
    decoded_message: Any

    def decode(self, decoder_ctx: McapDecoderContext) -> Any:
        """Return the already decoded message.

        If not decoded, decode it.
        """
        if self.decoded_message is None:
            self.decoded_message = decoder_ctx.decode_message(
                message_encoding=self.channel.message_encoding,
                message=self.message,
                schema=self.schema,
            )
        return self.decoded_message


@dataclass
class McapMessagesTuple:
    schema: Optional[McapSchema]
    channel: McapChannel
    messages: list[McapMessage]

    def sort(
        self,
        key: Literal["log_time", "publish_time"] = "log_time",
        reverse: bool = False,
    ) -> None:
        """Sort the messages by log time."""
        if key == "log_time":
            # Sort by log_time, which is an attribute of McapMessage
            self.messages.sort(key=lambda msg: msg.log_time, reverse=reverse)
        elif key == "publish_time":
            # Sort by pub_time, which is an attribute of McapMessage
            self.messages.sort(
                key=lambda msg: msg.publish_time, reverse=reverse
            )
        else:
            raise ValueError(
                f"Invalid sort key: {key}. Use 'log_time' or 'pub_time'."
            )

    def __getitem__(self, index: int) -> McapMessage:
        return self.messages[index]

    def __len__(self) -> int:
        """Return the number of messages in the tuple."""
        return len(self.messages)

    def append(self, msg: McapMessage | McapMessageTuple) -> None:
        """Append a new message to the messages tuple."""

        if isinstance(msg, McapMessage):
            self.messages.append(msg)
        elif isinstance(msg, McapMessageTuple):
            # check if the channel matches
            if self.channel.topic != msg.channel.topic:
                raise ValueError(
                    f"Channel mismatch: {self.channel.topic} != {msg.channel.topic}"  # noqa: E501
                )
            self.messages.append(msg.message)
        else:
            raise TypeError(
                "msg must be an instance of McapMessage or McapMessageTuple"
            )

    def decode(self, decoder_ctx: McapDecoderContext) -> list[Any]:
        """Decode all messages in the tuple using the provided decoder context."""  # noqa: E501
        decoded_messages = []
        for msg in self.messages:
            decoded_message = decoder_ctx.decode_message(
                message_encoding=self.channel.message_encoding,
                message=msg,
                schema=self.schema,
            )
            decoded_messages.append(decoded_message)
        return decoded_messages


@dataclass
class McapMessageBatch:
    message_dict: dict[str, McapMessagesTuple]
    is_last_batch: bool = False

    def __contains__(self, topic: str) -> bool:
        """Check if the topic is in the batch."""
        return topic in self.message_dict

    def __len__(self) -> int:
        """Return the total number of messages in the batch."""
        return sum(
            len(messages_tuple)
            for messages_tuple in self.message_dict.values()
        )

    def append(self, msg: McapMessagesTuple | McapMessageTuple) -> None:
        """Append a new messages tuple to the batch."""
        c_id = msg.channel.topic

        if isinstance(msg, McapMessagesTuple):
            if c_id not in self.message_dict:
                self.message_dict[c_id] = msg
            else:
                self.message_dict[c_id].messages.extend(msg.messages)

        else:
            if c_id not in self.message_dict:
                self.message_dict[c_id] = McapMessagesTuple(
                    schema=msg.schema,
                    channel=msg.channel,
                    messages=[msg.message],
                )
            else:
                self.message_dict[c_id].append(msg)

    def decode(self, decoder_ctx: McapDecoderContext) -> dict[str, list[Any]]:
        """Decode all messages using the provided decoder context."""
        decoded_batch = {}
        for c_id, messages_tuple in self.message_dict.items():
            decoded_messages = messages_tuple.decode(decoder_ctx)
            decoded_batch[c_id] = decoded_messages
        return decoded_batch

    def sort(
        self,
        key: Literal["log_time", "publish_time"] = "log_time",
        reverse: bool = False,
    ) -> None:
        """Sort all messages in the batch by the specified key."""
        for messages_tuple in self.message_dict.values():
            messages_tuple.sort(key=key, reverse=reverse)

    def extend(
        self, other: McapMessageBatch | Sequence[McapMessagesTuple]
    ) -> None:
        if isinstance(other, McapMessageBatch):
            for c_id, messages_tuple in other.message_dict.items():
                if c_id not in self.message_dict:
                    self.message_dict[c_id] = messages_tuple
                else:
                    self.message_dict[c_id].messages.extend(
                        messages_tuple.messages
                    )
        else:
            for messages_tuple in other:
                self.append(messages_tuple)


class MakeIterMsgArgs(Config):
    """Configuration for iterating messages in McapReader."""

    topics: Optional[Sequence[str]] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    log_time_order: bool = True
    reverse: bool = False
    start_offset: Optional[int] = None
    duration: Optional[int] = None

    def __post_init__(self):
        """Post-initialization to ensure valid time range configuration."""

        if self.duration is not None and self.end_time is not None:
            raise ValueError("Cannot specify both duration and end_time.")

    def update_time_range(self, message_start_time: int):
        if self.duration is not None and self.end_time is not None:
            raise ValueError("Cannot specify both duration and end_time.")

        if self.start_offset is not None:
            if self.start_time is None:
                self.start_time = message_start_time
            self.start_time += self.start_offset
            # reset start_offset to None to maintain consistency
            self.start_offset = None

        if self.duration is not None:
            if self.start_time is None:
                self.start_time = message_start_time
            self.end_time = self.start_time + self.duration
            # reset duration to None to maintain consistency
            self.duration = None


class McapReader:
    """A wrapper around the mcap reader to provide more convenient access.

    New features compared to the original mcap reader:
    - Separate the decoding logic from the reader for better flexibility.
    - Provide batch reading of messages with configurable splitting.

    """  # noqa: E501

    def __init__(self, reader: _McapReader):
        self.reader = reader

        # Expose the reader's methods for compatibility
        self.get_header = reader.get_header
        self.get_summary = reader.get_summary
        self.iter_attachments = reader.iter_attachments
        self.iter_metadata = reader.iter_metadata

    @staticmethod
    def make_reader(
        stream: IO[bytes],
        validate_crcs: bool = False,
    ) -> McapReader:
        return McapReader(
            mcap_make_reader(stream=stream, validate_crcs=validate_crcs)
        )

    def _update_time_range(self, iter_config: MakeIterMsgArgs):
        """Update the start and end time based on the provided parameters.

        If start_offset or duration is provided, it will adjust the start_time
        and end_time based on the message start time from the summary
        statistics. Both time and offset/duration should be in
        nanoseconds(10^9 nanoseconds = 1 second).

        """

        if (
            iter_config.start_offset is not None
            or iter_config.duration is not None
        ):
            summary = self.get_summary()
            if summary is None:
                raise McapError("Summary is not available in the reader.")
            statistics = summary.statistics
            if statistics is None:
                raise McapError("Statistics are not available in the reader.")
            iter_config.update_time_range(
                message_start_time=statistics.message_start_time
            )

    def iter_messages(
        self,
        iter_config: Optional[MakeIterMsgArgs] = None,
    ) -> Iterator[McapMessageTuple]:
        if iter_config is None:
            iter_config = MakeIterMsgArgs()

        self._update_time_range(iter_config)

        topics = iter_config.topics
        for schema, channel, msg in self.reader.iter_messages(
            topics=topics,
            start_time=iter_config.start_time,
            end_time=iter_config.end_time,
            log_time_order=iter_config.log_time_order,
            reverse=iter_config.reverse,
        ):
            yield McapMessageTuple(
                schema=schema,
                channel=channel,
                message=msg,
            )

    def iter_decoded_messages(
        self,
        decoder_ctx: McapDecoderContext,
        iter_config: Optional[MakeIterMsgArgs] = None,
    ) -> Iterator[McapDecodedMessageTuple]:
        for msg in self.iter_messages(iter_config=iter_config):
            decoded_message = decoder_ctx.decode_message(
                message_encoding=msg.channel.message_encoding,
                message=msg.message,
                schema=msg.schema,
            )
            yield McapDecodedMessageTuple(
                schema=msg.schema,
                channel=msg.channel,
                message=msg.message,
                decoded_message=decoded_message,
            )
