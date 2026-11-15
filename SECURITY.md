# Security policy

Recording an agent can preserve sensitive application information. Use this
guidance when choosing capture settings, storing recordings or reporting a
vulnerability. This library is not a sandbox: run only tools and plugins you trust.

## Reporting a vulnerability

Keep sensitive evidence out of public issues. Report vulnerabilities through
[GitHub's private reporting channel](https://github.com/mylonasc/langchain-ollie/security/advisories/new).

Do not post credentials, prompts, tool data, databases or private traces in public
issues. If private reporting is unavailable, open an issue requesting a private
contact channel, without exploit details or trace attachments.
No response-time SLA is promised.

## Protecting recordings

Collect only what you need, and review the resulting artifact before sharing it.
The [capture and recording limits](README.md#safety-and-recording-limits) explain
how to opt into more detail without removing safety budgets.

Metadata-only capture is the default, not a guarantee that every application
identifier is non-sensitive. Review names, IDs, resources and explicit links
before sharing. Bounded/full capture deliberately retains application data.
Keep trace/database files private; redaction is not encryption. SQLite WAL/SHM,
backups and deleted pages may retain data. Deletion is not secure erase. Use
filesystem permissions, encryption and retention appropriate to your workload.
Never load untrusted database/profile artifacts into a privileged process.

Evaluation artifacts can retain case inputs, agent answers and check evidence
independently of trace capture settings. Treat saved runs, reports and dataset
proposals with the same care as trace files. See
[saving evaluation results](docs/evals.md#save-and-reopen-results).

## Supported versions

Check the [supported runtime scope](README.md#supported-scope-and-migration) before
reporting a compatibility problem. The current 0.1.x local-first API is supported
on the combinations exercised in CI; other runtimes are not claimed as tested.
