# Security policy

The current 0.1.x local-first API is supported on the combinations in CI. This
library is not a sandbox: run only agent tools and plugins that you trust.

Do not post credentials, prompts, tool data, databases or private traces in public
issues. Report vulnerabilities privately through GitHub security advisories
("Report a vulnerability"). If private reporting is unavailable, open a body-free
issue requesting a private contact channel, not an exploit or trace attachment.
No response-time SLA is promised.

Metadata-only capture is the default, not a guarantee that every application
identifier is non-sensitive. Review names, IDs, resources and explicit links
before sharing. Bounded/full capture deliberately retains application data.
Keep trace/database files private; redaction is not encryption. SQLite WAL/SHM,
backups and deleted pages may retain data. Deletion is not secure erase. Use
filesystem permissions, encryption and retention appropriate to your workload.
Never load untrusted database/profile artifacts into a privileged process.
