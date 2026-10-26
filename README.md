# Dataset included

- File: `msmarco/train.jsonl`
- Dataset: MS MARCO Passage Ranking training records in the query/positive-passages/negative-passages JSONL schema used by the experiment scripts.
- Use: the included Web scripts read this training file and construct their deterministic reservoir sample and train/validation/test partitions from it. The current Web-search evaluation is a query-local candidate ranking task using one supplied positive and 30 supplied negatives per eligible query; it is not a full-corpus MS MARCO benchmark run.
- The original workspace also contained `dev.jsonl`, `dl20.jsonl`, and compressed duplicates. They are omitted because none of the included core scripts reads them.
- Keep the dataset unchanged. Check and comply with the MS MARCO dataset terms when redistributing this package.

The original training file is approximately 5.24 GB. Confirm the copied file size before running experiments.
