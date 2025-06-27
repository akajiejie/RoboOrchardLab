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

import autoapi._mapper
from autoapi._mapper import LOGGER, operator


def _patched_create_class(self, data, options=None):
    """Create a class from the passed in data.

    Args:
        data: dictionary data of parser output
    """
    try:
        cls = self._OBJ_MAP[data["type"]]
    except KeyError:
        # this warning intentionally has no (sub-)type
        LOGGER.warning(f"Unknown type: {data['type']}")
    else:
        obj = cls(
            data,
            class_content=self.app.config.autoapi_python_class_content,
            options=self.app.config.autoapi_options,
            jinja_env=self.jinja_env,
            app=self.app,
            url_root=self.url_root,
        )

        for child_data in data.get("children", []):
            for child_obj in self.create_class(child_data, options=options):
                obj.children.append(child_obj)

        # Some objects require children to establish their docstring
        # or type annotations (eg classes with inheritance),
        # so do this after all children have been created.
        lines = obj.docstring.splitlines()
        if lines:
            # Add back the trailing newline that .splitlines removes
            lines.append("")
            if "autodoc-process-docstring" in self.app.events.events:
                self.app.emit(
                    "autodoc-process-docstring",
                    cls.type,
                    obj.name,
                    obj,
                    None,
                    lines,
                )
        obj.docstring = "\n".join(lines)
        self._record_typehints(obj)

        # Parser gives children in source order already
        if self.app.config.autoapi_member_order == "alphabetical":
            obj.children.sort(key=operator.attrgetter("name"))
        elif self.app.config.autoapi_member_order == "groupwise":
            obj.children.sort(key=lambda x: (x.member_order, x.name))

        yield obj


def patch_autoapi():
    autoapi._mapper.Mapper.create_class = _patched_create_class
