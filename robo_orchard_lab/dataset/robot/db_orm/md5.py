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
from abc import abstractmethod
from typing import Generic, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BLOB

T = TypeVar("T", bound="MD5FieldMixin")


class MD5FieldMixin(Generic[T]):
    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @abstractmethod
    def update_md5(self) -> bytes:
        """Generate a unique MD5 hash for the class.

        The MD5 hash is generated from the JSON content and name.
        """
        raise NotImplementedError(
            "Subclasses must implement the update_md5 method."
        )

    @staticmethod
    def query_by_content_with_md5(
        session: Session, cls: Type[T], **kwargs
    ) -> T | None:
        """Query an instance of the class by its content and MD5 hash.

        This method use the MD5 hash to filter results, and then checks
        if all provided keyword arguments match the attributes of the
        instance.

        """

        md5 = cls(**kwargs).update_md5()

        stmt = select(cls).where(cls.md5 == md5)
        for result in session.execute(stmt).scalars():
            if all(getattr(result, k) == kwargs[k] for k in kwargs):
                return result
        return None
