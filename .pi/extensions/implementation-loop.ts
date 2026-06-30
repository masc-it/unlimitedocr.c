import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const COMMAND = "implementation-loop";
const STOP_COMMAND = "implementation-loop-stop";
const STATUS_KEY = "implementation-loop";
const CONTEXT_THRESHOLD_PERCENT = 60;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 10_000;
const QA_CHECKPOINT_INTERVAL = 5;

const IMPLEMENTATION_PROMPT = `read the context in @docs/implementation_plan.md @docs/architecture.md
@docs/foundations.md and proceed with the next implementation item.

when you pick up the item, first explain what it is about and what is needed for.

when you're done and the code works as intended, check of the item checkbox in the plan doc and commit with a short message.

Notes:
- we're focusing on Metal backend (this pc has 8GB free, m1 pro), so skip CUDA-related items.
- focus on implementation. Don't write unit tests, we do periodic extensive QA testing with temporary probes.
- if there is not further item to work on, return "WORK DONE".

write good code.`;

const QA_PROMPT = `read the context in @docs/implementation_plan.md @docs/architecture.md
@docs/foundations.md.

QA checkpoint: stress test the code to ensure the last 5 implementation checkboxes are properly working as expected. Write temporary probes under data.tmp/probes and delete them at the end of turn.
If anything fails or behaves incorrectly, find the root cause and fix it.

write good code.`;

type PromptKind = "implementation" | "qa";
type CompactionContinuation = () => void | Promise<void>;

interface MessageLike {
	role?: string;
	stopReason?: string;
	errorMessage?: string;
	content?: unknown;
}

export default function (pi: ExtensionAPI) {
	let running = false;
	let compacting = false;
	let sending = false;
	let iteration = 0;
	let runId = 0;
	let retryCount = 0;
	let compactionRetryCount = 0;
	let retryTimeout: ReturnType<typeof setTimeout> | undefined;
	let compactionRetryTimeout: ReturnType<typeof setTimeout> | undefined;
	let qaDue = false;
	let lastPromptKind: PromptKind = "implementation";

	function notify(ctx: ExtensionContext, message: string, level: "info" | "warning" | "error" = "info") {
		if (ctx.hasUI) {
			ctx.ui.notify(message, level);
		}
	}

	function setStatus(ctx: ExtensionContext, text?: string) {
		if (ctx.hasUI) {
			ctx.ui.setStatus(STATUS_KEY, text);
		}
	}

	function clearRetryTimeout() {
		if (retryTimeout === undefined) return;
		clearTimeout(retryTimeout);
		retryTimeout = undefined;
	}

	function clearCompactionRetryTimeout() {
		if (compactionRetryTimeout === undefined) return;
		clearTimeout(compactionRetryTimeout);
		compactionRetryTimeout = undefined;
	}

	function clearRetryTimers() {
		clearRetryTimeout();
		clearCompactionRetryTimeout();
	}

	function stopLoop(ctx: ExtensionContext, message: string, level: "info" | "warning" | "error" = "info") {
		clearRetryTimers();
		running = false;
		compacting = false;
		sending = false;
		retryCount = 0;
		compactionRetryCount = 0;
		qaDue = false;
		lastPromptKind = "implementation";
		runId += 1;
		setStatus(ctx, undefined);
		notify(ctx, message, level);
	}

	function contextPercent(ctx: ExtensionContext): number | null {
		const usage = ctx.getContextUsage();
		if (!usage || usage.tokens === null) return null;
		return usage.percent ?? (usage.contextWindow > 0 ? (usage.tokens / usage.contextWindow) * 100 : null);
	}

	function promptFor(kind: PromptKind): string {
		return kind === "qa" ? QA_PROMPT : IMPLEMENTATION_PROMPT;
	}

	function sendPrompt(ctx: ExtensionContext, prompt: string, deliverAsFollowUp = false) {
		if (!deliverAsFollowUp && ctx.isIdle()) {
			pi.sendUserMessage(prompt);
			return;
		}

		pi.sendUserMessage(prompt, { deliverAs: "followUp" });
	}

	function queuePrompt(ctx: ExtensionContext, kind: PromptKind, deliverAsFollowUp = false): boolean {
		const prompt = promptFor(kind);
		try {
			sendPrompt(ctx, prompt, deliverAsFollowUp);
			lastPromptKind = kind;
			return true;
		} catch (error) {
			// Race-safe fallback if the agent became busy between isIdle() and sendUserMessage().
			try {
				pi.sendUserMessage(prompt, { deliverAs: "followUp" });
				lastPromptKind = kind;
				return true;
			} catch (fallbackError) {
				const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
				stopLoop(ctx, `/${COMMAND} stopped: failed to queue prompt: ${message}`, "error");
				return false;
			}
		}
	}

	function retryPrompt(ctx: ExtensionContext, reason: string) {
		if (!running) return;
		if (retryTimeout !== undefined) {
			setStatus(ctx, `implementation-loop retry ${retryCount}/${MAX_RETRIES} already scheduled`);
			return;
		}
		if (ctx.hasPendingMessages()) {
			setStatus(ctx, `implementation-loop retry ${retryCount}/${MAX_RETRIES} waiting for queued prompt`);
			return;
		}
		if (retryCount >= MAX_RETRIES) {
			stopLoop(ctx, `/${COMMAND} stopped: ${reason} after ${MAX_RETRIES} retries.`, "error");
			return;
		}

		retryCount += 1;
		const expectedRunId = runId;
		const retryKind = lastPromptKind;
		setStatus(ctx, `implementation-loop retry ${retryCount}/${MAX_RETRIES} in 10s`);
		notify(ctx, `/${COMMAND}: ${reason}; retrying ${retryCount}/${MAX_RETRIES} in 10 seconds.`, "warning");
		retryTimeout = setTimeout(() => {
			retryTimeout = undefined;
			if (!running || expectedRunId !== runId) return;
			if (ctx.hasPendingMessages()) {
				setStatus(ctx, `implementation-loop retry ${retryCount}/${MAX_RETRIES} waiting for queued prompt`);
				return;
			}
			queuePrompt(ctx, retryKind, true);
		}, RETRY_DELAY_MS);
	}

	function retryCompaction(
		ctx: ExtensionContext,
		expectedRunId: number,
		percent: number,
		reason: string,
		onCompacted?: CompactionContinuation,
	) {
		if (!running || expectedRunId !== runId) return;
		if (compactionRetryTimeout !== undefined) {
			setStatus(ctx, `implementation-loop compaction retry ${compactionRetryCount}/${MAX_RETRIES} already scheduled`);
			return;
		}
		if (compactionRetryCount >= MAX_RETRIES) {
			stopLoop(ctx, `/${COMMAND} stopped: compaction failed after ${MAX_RETRIES} retries (${reason}).`, "error");
			return;
		}

		compactionRetryCount += 1;
		setStatus(ctx, `implementation-loop compaction retry ${compactionRetryCount}/${MAX_RETRIES} in 10s`);
		notify(
			ctx,
			`/${COMMAND}: compaction failed (${reason}); retrying ${compactionRetryCount}/${MAX_RETRIES} in 10 seconds.`,
			"warning",
		);
		compactionRetryTimeout = setTimeout(() => {
			compactionRetryTimeout = undefined;
			if (!running || expectedRunId !== runId) return;
			startCompaction(ctx, expectedRunId, percent, onCompacted);
		}, RETRY_DELAY_MS);
	}

	function runCompactionContinuation(ctx: ExtensionContext, continuation: CompactionContinuation) {
		void Promise.resolve()
			.then(continuation)
			.catch((error) => {
				const message = error instanceof Error ? error.message : String(error);
				retryPrompt(ctx, `post-compaction continuation error (${message})`);
			});
	}

	function startCompaction(
		ctx: ExtensionContext,
		expectedRunId: number,
		percent: number,
		onCompacted?: CompactionContinuation,
	) {
		if (!running || compacting || expectedRunId !== runId) return;

		clearRetryTimeout();
		clearCompactionRetryTimeout();
		compacting = true;
		setStatus(ctx, `implementation-loop compacting (${percent.toFixed(1)}%)`);
		notify(ctx, `/${COMMAND}: context at ${percent.toFixed(1)}%; compacting before continuing.`, "info");

		try {
			ctx.compact({
				customInstructions:
					"This session is running the /implementation-loop extension. Preserve the implementation goal, completed changes, current work, important decisions, files modified/read, test results, QA checkpoint findings, and the next implementation item from docs/implementation_plan.md. Omit stale detail and large file contents.",
				onComplete: () => {
					compacting = false;
					if (!running || expectedRunId !== runId) {
						setStatus(ctx, undefined);
						return;
					}

					compactionRetryCount = 0;
					notify(ctx, `/${COMMAND}: compaction completed; continuing.`, "info");
					if (onCompacted) {
						runCompactionContinuation(ctx, onCompacted);
						return;
					}
					void continueLoop(ctx, expectedRunId, false);
				},
				onError: (error) => {
					compacting = false;
					retryCompaction(ctx, expectedRunId, percent, error.message, onCompacted);
				},
			});
		} catch (error) {
			compacting = false;
			const message = error instanceof Error ? error.message : String(error);
			retryCompaction(ctx, expectedRunId, percent, message, onCompacted);
		}
	}

	async function continueLoop(ctx: ExtensionContext, expectedRunId: number, allowCompaction = true) {
		if (!running || compacting || expectedRunId !== runId || sending) return;
		sending = true;

		try {
			const percent = contextPercent(ctx);
			if (allowCompaction && percent !== null && percent >= CONTEXT_THRESHOLD_PERCENT) {
				startCompaction(ctx, expectedRunId, percent);
				return;
			}

			if (qaDue) {
				qaDue = false;
				setStatus(ctx, `implementation-loop QA after #${iteration}`);
				queuePrompt(ctx, "qa", true);
				return;
			}

			iteration += 1;
			if (iteration % QA_CHECKPOINT_INTERVAL === 0) {
				qaDue = true;
			}
			setStatus(ctx, `implementation-loop #${iteration}`);
			queuePrompt(ctx, "implementation", true);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			retryPrompt(ctx, `extension error (${message})`);
		} finally {
			sending = false;
		}
	}

	function getTextContent(content: unknown): string {
		if (typeof content === "string") return content;
		if (!Array.isArray(content)) return "";

		return content
			.map((block) => {
				if (block && typeof block === "object" && "text" in block && typeof block.text === "string") {
					return block.text;
				}
				return "";
			})
			.join(" ");
	}

	function findAgentError(messages: readonly MessageLike[]): string | undefined {
		for (const message of messages) {
			if (message.role !== "assistant") continue;
			if (message.stopReason !== "error" && message.stopReason !== "aborted") continue;

			const details = [message.errorMessage, getTextContent(message.content)].filter(Boolean).join(" ").trim();
			return details || `assistant stopped with ${message.stopReason}`;
		}
		return undefined;
	}

	function hasWorkDoneMessage(messages: readonly MessageLike[]): boolean {
		return messages.some(
			(message) => message.role === "assistant" && getTextContent(message.content).toUpperCase().includes("WORK DONE"),
		);
	}

	pi.on("session_start", (_event, ctx) => {
		setStatus(ctx, running ? (qaDue ? `implementation-loop QA after #${iteration}` : `implementation-loop #${iteration}`) : undefined);
	});

	pi.on("session_shutdown", (_event, ctx) => {
		clearRetryTimers();
		setStatus(ctx, undefined);
	});

	pi.on("agent_end", async (event, ctx) => {
		if (!running || compacting) return;

		try {
			const expectedRunId = runId;
			const messages = event.messages as readonly MessageLike[];
			if (hasWorkDoneMessage(messages)) {
				stopLoop(ctx, `/${COMMAND} stopped: WORK DONE.`, "info");
				return;
			}

			const agentError = findAgentError(messages);
			if (!agentError) {
				retryCount = 0;
				clearRetryTimeout();
			}

			const percent = contextPercent(ctx);
			if (percent !== null && percent >= CONTEXT_THRESHOLD_PERCENT) {
				startCompaction(
					ctx,
					expectedRunId,
					percent,
					agentError ? () => retryPrompt(ctx, `agent error: ${agentError}`) : undefined,
				);
				return;
			}

			if (agentError) {
				retryPrompt(ctx, `agent error: ${agentError}`);
				return;
			}

			await continueLoop(ctx, expectedRunId, false);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			retryPrompt(ctx, `agent_end handler error (${message})`);
		}
	});

	pi.registerCommand(COMMAND, {
		description: "Start the automatic next implementation-item loop",
		handler: async (_args, ctx) => {
			if (running) {
				notify(ctx, `/${COMMAND} is already running. Use /${STOP_COMMAND} to stop it.`, "warning");
				return;
			}

			clearRetryTimers();
			running = true;
			compacting = false;
			sending = false;
			iteration = 0;
			retryCount = 0;
			compactionRetryCount = 0;
			qaDue = false;
			lastPromptKind = "implementation";
			runId += 1;
			setStatus(ctx, "implementation-loop starting");
			notify(ctx, `/${COMMAND} started.`, "info");
			void continueLoop(ctx, runId, false);
		},
	});

	pi.registerCommand(STOP_COMMAND, {
		description: "Stop the automatic next implementation-item loop",
		handler: async (_args, ctx) => {
			if (!running && !compacting) {
				notify(ctx, `/${COMMAND} is not running.`, "info");
				return;
			}

			stopLoop(ctx, `/${COMMAND} stopped. The current agent turn is not aborted.`, "info");
		},
	});
}
