"""Global identity layer — cross-camera person/vehicle re-identification.

Assigns persistent `global_id`s to observations across cameras using ReID
embeddings. Consumes per-camera pipeline outputs (objects with `embedding_b64`
and `camera_id`) and produces a global identity report.

Flow:
    process_cameras()  →  IdentityManager.ingest(camera_objects)
                            →  matcher (cosine + topology boost)
                            →  IdentityStore (persistent)
                            →  global_identity_report.json
"""
