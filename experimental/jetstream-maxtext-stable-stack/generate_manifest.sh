#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script generates a manifest of currently installed Python packages, along with their versions.
# The manifest is named with a timestamp for easy versioning and tracking.

export PREFIX='default'

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

# Set the Manifest file name with the date for versioning
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
MANIFEST_FILE="${PREFIX}_manifest_${TIMESTAMP}.txt"

# Freeze packages installed and their version to the Manifest file, with sorted and commented Manifest
pip freeze | sort > "$MANIFEST_FILE"

# Maxtext depend on main branch of jetstream we don't want.
# Remove google-jetstream from the Manifest file
grep -vE '^google-jetstream(==|>=|<=|>|<| |@|$)' "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"

# Write commit details to the Manifest file
if [[ -n "$MAXTEXT_COMMIT_HASH" ]]; then
    echo "# maxtext commit hash: $MAXTEXT_COMMIT_HASH" | cat - "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"
fi
if [[ -n "$JETSTREAM_COMMIT_HASH" ]]; then
    echo "# JetStream commit hash: $JETSTREAM_COMMIT_HASH" | cat - "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"
fi

# Add a header comment to the Manifest file
echo "# Python Packages Frozen at: ${TIMESTAMP}" | cat - "$MANIFEST_FILE" > temp && mv temp "$MANIFEST_FILE"
