import { readFileSync } from "fs";
import * as core from "@actions/core";
import OpenAI from "openai";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import minimatch from "minimatch";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const OPENAI_API_KEY: string = core.getInput("OPENAI_API_KEY");
const OPENAI_API_MODEL: string = core.getInput("OPENAI_API_MODEL");

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

/**
 * key is the file name, value is the set of line numbers that are valid for comments
 */
type PRDiffInfo = Record<string, Set<number>>;

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function getPRDiffInfo(
  owner: string,
  repo: string,
  pull_number: number
): Promise<PRDiffInfo> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });

  // @ts-expect-error - response.data is a string
  const prDiff = parseDiff(response.data);

  const prDiffInfo: PRDiffInfo = {};

  let i = 0;
  for (const file of prDiff) {
    if (!file.to)
      throw new Error(
        i + " File name is undefined:\n" + JSON.stringify(prDiff)
      );

    const changedLines = new Set<number>();

    for (const chunk of file.chunks) {
      for (const change of chunk.changes) {
        switch (change.type) {
          case "normal":
            changedLines.add(change.ln2);
            break;
          case "add":
            changedLines.add(change.ln);
            break;
          case "del":
            // del is not a line number, so we don't need to add it to the set
            break;
          default:
            console.log(
              "file:::::::::\n",
              JSON.stringify(file, null, 2),
              "\n------,,,,,,,,,"
            );
            throw new Error("Unknown change type: " + (change as any).type);
        }
      }
    }

    prDiffInfo[file.to || ""] = changedLines;
    i++;
  }

  return prDiffInfo;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails,
  prDiffInfo: PRDiffInfo
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue;
    if (!file.to)
      throw new Error("File name is undefined::\n" + JSON.stringify(file));

    const fileDiffInfo = prDiffInfo[file.to];

    if (!fileDiffInfo) {
      console.log(`No PR diff info found for file: ${file.to}`);
      continue;
    }

    const fileContent = await getFileContent(
      prDetails.owner,
      prDetails.repo,
      file.to!,
      prDetails.pull_number
    );

    for (const chunk of file.chunks) {
      const prompt = createPrompt(
        file,
        chunk,
        prDetails,
        fileContent,
        fileDiffInfo
      );
      console.log("Prompt:::\n", prompt, "\n----------------===");
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, fileDiffInfo, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

async function getFileContent(
  owner: string,
  repo: string,
  path: string,
  pull_number: number
): Promise<string> {
  try {
    const prResponse = await octokit.pulls.get({
      owner,
      repo,
      pull_number,
    });

    const contentResponse = await octokit.rest.repos.getContent({
      owner,
      repo,
      path,
      ref: prResponse.data.head.sha,
    });

    if (
      !Array.isArray(contentResponse.data) &&
      "content" in contentResponse.data
    ) {
      const content = Buffer.from(
        contentResponse.data.content,
        "base64"
      ).toString();
      return content;
    }

    throw new Error(`File ${path} not found in PR`);
  } catch (error) {
    console.error(`Error fetching file content: ${error}`);
    throw error;
  }
}

function createPrompt(
  file: File,
  chunk: Chunk,
  prDetails: PRDetails,
  fileContent: string,
  validLines: Set<number>
): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}]}
- Do not give positive comments or compliments.
- For languages with type systems like TypeScript, import statements or importers almost never need to be changed, so please do not suggest changes to them.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.
- ONLY review the code changes in the provided diff, not the entire file content.
- CRITICAL: You can ONLY comment on the following line numbers: ${Array.from(
    validLines
  ).join(", ")}
- If you want to reference code outside the diff, include it in your comment but set the lineNumber to a valid diff line.

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.

Pull request title: ${prDetails.title}

${
  prDetails.description
    ? `Pull request description:\n\n---\n${prDetails.description}\n---`
    : ""
}

Please provide a review based on the diff and file content.

Before showing the diff, here's an explanation of the diff format:
\`\`\`
In a Git diff, @@ is part of a "hunk header," which indicates the starting position and range of the changes in both the old and new files.

@@ -<start line>,<number of lines> +<start line>,<number of lines> @@

	•	-<start line>,<number of lines>: Refers to the starting line and number of lines affected in the original file (before the changes).
	•	+<start line>,<number of lines>: Refers to the starting line and number of lines affected in the modified file (after the changes).

For the content lines:
- Lines starting with " -" indicate deleted lines
- Lines starting with " +" indicate added lines
- Lines without " -" or " +" indicate unchanged lines in this commit
\`\`\`

Here's the diff for this change:
\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`

Below is the complete content of the file after this change. Please provide a review based on the above diff and the following file content:
\`\`\`
${fileContent}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: OPENAI_API_MODEL,
    temperature: 0.2,
    max_tokens: 700,
    top_p: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
  };

  try {
    const response = await openai.chat.completions.create({
      ...queryConfig,
      // the model should support JSON output
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: prompt,
        },
      ],
    });

    const res = response.choices[0].message?.content?.trim() || "{}";

    return JSON.parse(res).reviews;
  } catch (error) {
    console.error("getAIResponse Error:", error);
    return null;
  }
}

function createComment(
  file: File,
  validLines: Set<number>,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }

    const lineNumber = Number(aiResponse.lineNumber);

    if (!validLines.has(lineNumber)) {
      console.log(
        `Skipping comment for line ${lineNumber} as it's not in the PR diff`
      );
      return [];
    }

    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: lineNumber,
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
    side: "RIGHT",
  });
}

function isValidPath(path: string): boolean {
  const pathParts = path.split("/");
  return !pathParts.some((part) => part.startsWith("."));
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const validFiles = filteredDiff.filter((file) => isValidPath(file.to ?? ""));

  const prDiffInfo = await getPRDiffInfo(
    prDetails.owner,
    prDetails.repo,
    prDetails.pull_number
  );

  const comments = await analyzeCode(validFiles, prDetails, prDiffInfo);
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
