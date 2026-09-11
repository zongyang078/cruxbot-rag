# Corpus collection

These five scripts build the 338,433-document corpus from its original sources,
and a sixth repairs forum URLs afterwards. Run them in order from the
repository root:

```bash
pip install -e ".[ingest]"

python scripts/ingest/01_openbeta_collect.py    # GraphQL API, 47 US states
python scripts/ingest/02_kaggle_8a_collect.py   # Mountain Project routes + forums
python scripts/ingest/02b_aac_collect.py        # American Alpine Club articles
python scripts/ingest/03_reddit_collect.py      # public JSON endpoints
python scripts/ingest/04_clean_and_unify.py     # dedup, clean, unify the schema
```

The output is `data/unified/cruxbot_unified.json`, which
`cruxbot.indexing.corpus.to_jsonl` converts to the streamable form the indexer
reads. See [`docs/DATA_COLLECTION_METHODS.md`](../../docs/DATA_COLLECTION_METHODS.md)
for what each source contributes and the cleaning applied to it.

`fix_forum_urls.py` is a one-off repair: the Kaggle mirror of the Mountain
Project forums did not preserve per-post URLs, so those entries share a
site-level link. The script recovers specific URLs where an id survives in the
data and flags the rest, which is why the retrieval pipeline distinguishes
citable URLs from site-level ones at all.

## Status

Carried over from the [predecessor repository](https://github.com/zongyang078/CruxBot)
with formatting brought in line with this one, but **otherwise unchanged and
untested**. They are network- and API-key-bound, run once, and produced the
corpus the rest of the project is built on; the rest of `src/` is covered by
tests and these are not. They are here so the corpus can be rebuilt from its
sources rather than copied from a release -- most of it is not redistributable.

Two of the six sources can be refreshed: Reddit (public JSON, worth polling
daily) and OpenBeta (GraphQL, changes slowly). The other four are frozen Kaggle
snapshots, so re-running those scripts reproduces the same data rather than
updating it.
