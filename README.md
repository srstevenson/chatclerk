# chatclerk

chatclerk converts LLM chat exports to directories of Markdown files. It
supports [ChatGPT], [Claude], and [Grok] exports, and preserves conversation
structure, metadata, and associated assets such as generated images, artefacts,
and file attachments.

It allows you to archive conversations from LLM services to storage under your
control, to provide indefinite access independent of LLM service uptime or
account closure, like you might archive emails from webmail services with
[isync] or [OfflineIMAP].

Each conversation is saved as a Markdown file named after its unique identifier,
with metadata, timestamps, and messages formatted by role. Associated assets are
saved in subdirectories matching the conversation filename.

## Installation

chatclerk requires Python 3.14 or later. Install it from GitHub using [uv] with:

```sh
uv tool install git+https://github.com/srstevenson/chatclerk.git
```

Alternatively, you can clone the repository and install from source:

```sh
git clone https://github.com/srstevenson/chatclerk.git
cd chatclerk
uv tool install .
```

This will install the `chatclerk-chatgpt`, `chatclerk-claude`, and
`chatclerk-grok` executables into `~/.local/bin`. Ensure `~/.local/bin` is in
your shell's `PATH`.

## Usage

chatclerk provides three command-line tools, one for each supported LLM service.

### ChatGPT

To convert a ChatGPT export, download your data from
<https://chatgpt.com/#settings/DataControls> and extract the archive. Then run:

```sh
chatclerk-chatgpt -i /path/to/chatgpt-export -o chatgpt-logs
```

This converts all conversations to Markdown files in the `chatgpt-logs`
directory. Generated images and other assets are saved to subdirectories.

### Claude

To convert a Claude export, download your data from
<https://claude.ai/settings/data-privacy-controls> and extract the archive. Then
run:

```sh
chatclerk-claude -i /path/to/claude-export -o claude-logs
```

This converts all conversations to Markdown files in the `claude-logs`
directory. Artefacts and tool results are preserved in the output.

### Grok

To convert a Grok export, download your data from <https://grok.com/?_s=data>
and extract the archive. Then run:

```sh
chatclerk-grok -i /path/to/grok-export -o grok-logs
```

This converts all conversations to Markdown files in the `grok-logs` directory.
Artefacts and file attachments are saved to subdirectories.

## Development

chatclerk uses [uv] for dependency management and [Ruff] and [basedpyright] for
formatting, linting, and type checking.

To clone the repository and set up a development environment run:

```sh
git clone https://github.com/srstevenson/chatclerk.git
cd chatclerk
uv sync
```

Format code and run linters with:

```sh
make fmt
make lint
```

Check the code formatting and linting without modifying files with:

```sh
make check
```

[basedpyright]: https://docs.basedpyright.com/
[ChatGPT]: https://chatgpt.com/
[Claude]: https://claude.ai/
[Grok]: https://grok.com/
[isync]: https://isync.sourceforge.io/
[OfflineIMAP]: https://github.com/OfflineIMAP/offlineimap3
[Ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/
