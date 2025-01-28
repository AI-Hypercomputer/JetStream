/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Uses Github's API to create the release and wait for result.
// We use a JS script since github CLI doesn't provide a way to wait for the release's creation and returns immediately.

module.exports = async (github, context, core) => {
	try {
		const response = await github.rest.repos.createRelease({
			draft: false,
			generate_release_notes: true,
			name: process.env.RELEASE_TAG,
			owner: context.repo.owner,
			prerelease: false,
			repo: context.repo.repo,
			tag_name: process.env.RELEASE_TAG,
		});

		core.setOutput('upload_url', response.data.upload_url);
	} catch (error) {
		core.setFailed(error.message);
	}
}