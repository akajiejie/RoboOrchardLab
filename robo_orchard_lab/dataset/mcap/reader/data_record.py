# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

from dataclasses import dataclass

from robo_orchard_lab.dataset.mcap.reader.reader import (
    MakeIterMsgArgs,
    McapMessageBatch,
    McapReader,
)

__all__ = ["McapDataRecordChunk", "McapDataRecordChunks"]


@dataclass
class McapDataRecordChunk:
    """Data record chunk for reading messages from an MCAP file.

    The data record chunk is defined by messages in  specific topics and
    a time range. One data record can contain multiple messages.

    For training purposes, a data record is usually treated as one
    training sample, which contains all messages in the specified
    topics within the specified time range.
    """

    topics: list[str]
    log_time_start: int
    log_time_end: int

    def read(self, mcap_reader: McapReader) -> McapMessageBatch:
        """Read the data record from the mcap reader."""
        ret = McapMessageBatch({}, is_last_batch=True)
        for msg in mcap_reader.iter_messages(
            MakeIterMsgArgs(
                topics=self.topics,
                start_time=self.log_time_start,
                end_time=self.log_time_end,
                log_time_order=True,
                reverse=False,
            )
        ):
            ret.append(msg)
        return ret


@dataclass
class McapDataRecordChunks:
    """Data record chunks for reading from an MCAP file.

    In some cases, for example, we need to read multiple messages from
    different time ranges and groups them into a single data record,
    we can use this class to represent multiple data record chunks.
    """

    chunks: list[McapDataRecordChunk]

    def read(self, mcap_reader: McapReader) -> McapMessageBatch:
        """Read the data chunk record from the mcap reader."""
        ret = McapMessageBatch({}, is_last_batch=True)
        for record in self.chunks:
            batch = record.read(mcap_reader)
            ret.extend(batch)
        ret.sort()
        return ret
