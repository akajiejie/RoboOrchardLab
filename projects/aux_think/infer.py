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


import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value


def process_file(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
            result = {
                "success": check_inf_nan(int(data.get("success", 0))),
                "spl": check_inf_nan(data.get("spl", 0)),
                "distance_to_goal": check_inf_nan(
                    data.get("distance_to_goal", 0)
                ),
                "oracle_success": check_inf_nan(
                    int(data.get("oracle_success", 0))
                ),
                "path_length": data.get("path_length", 0),
                "valid": True,
            }
            return result
    except Exception:
        return {"valid": False, "file": file_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    log_dir = os.path.join(args.path, "log")
    json_files = [os.path.join(log_dir, j) for j in os.listdir(log_dir)]

    succ = 0
    spl = 0
    distance_to_goal = 0
    path_length = 0
    oracle_succ = 0
    total = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, path) for path in json_files]
        for future in as_completed(futures):
            result = future.result()
            if not result["valid"]:
                print(f"Failed to parse: {result['file']}")
                continue
            succ += result["success"]
            spl += result["spl"]
            distance_to_goal += result["distance_to_goal"]
            oracle_succ += result["oracle_success"]
            path_length += result["path_length"]
            total += 1

    if total == 0:
        print("No valid JSON files found.")
    else:
        print(f"Success rate: {succ}/{total} ({succ / total:.3f})")
        print(f"OSR: {oracle_succ}/{total} ({oracle_succ / total:.3f})")
        print(f"SPL: {spl:.3f}/{total} ({spl / total:.3f})")
        print(f"Distance to goal: {distance_to_goal / total:.3f}")
        print(f"Path length: {path_length / total:.3f}")
