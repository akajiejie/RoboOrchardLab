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
import importlib

IGNORE_PARENT_PACKAGES = ["transformers."]


def autodoc_process_docstring_event(app, what, name, obj, options, lines):
    if what != "class":
        return

    for base_class in obj.bases:
        if any((base_class.startswith(pkg) for pkg in IGNORE_PARENT_PACKAGES)):
            module_path, class_name = obj.obj["full_name"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            live_class = getattr(module, class_name)
            if live_class.__doc__ is None:
                lines.clear()
                break
