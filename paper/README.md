# Conference paper

```bash
make          # compile the draft; TODO/NUM markers render in colour
make sync     # pull the latest generated tables and figures from ../results/
make check    # fail if any marker or unverified citation remains
make clean
```

## How this is wired

Tables and figures are **generated**, not written here. `make sync` copies them
from `../results/figures/`, so regenerating results with `src/figures.py` and
rebuilding is enough — the paper cannot silently drift from the numbers it
reports. Placeholder figures are provided so the skeleton compiles before any
results exist.

## The two markers

`\TODO{...}` marks prose you must write. `\NUM{...}` marks a number that must
come from a results file rather than from memory — a figure typed from
recollection is the easiest way to publish something you cannot reproduce.

`make check` refuses to pass while either remains, and also fails while any
`TODO-VERIFY` entry is left in `references.bib`. Run it before you submit.

## Venue

The skeleton uses the standard `article` class so it compiles anywhere. Switch
`\documentclass` to your venue's class — IEEEtran, acmart or llncs — once the
venue is chosen; the body carries over unchanged.

## Citations

`references.bib` contains 16 entries, of which several are flagged
`TODO-VERIFY`. Open every one and confirm the venue, year, volume and pages
against the publisher page. Do not cite anything you have not read: examiners
and reviewers check, and a wrong citation costs more credibility than a missing
one.
