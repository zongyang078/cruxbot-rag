"""Build the search indexes from the unified corpus.

`chunking` is pure and testable; the pipeline modules touch the embedding
model and vector store, so their heavy imports are deferred.
"""
