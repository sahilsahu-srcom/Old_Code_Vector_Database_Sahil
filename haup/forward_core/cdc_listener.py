"""
File Summary:
CDC event listener for HAUP v2.0. Reacts to Debezium change events from PostgreSQL WAL to keep vector DB in sync.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

CDCListener()  [Class → Object]
||
├── __init__()  [Method] ---------------------------------> Initialize Kafka config, vector_db, model, strategy
│
├── start_cdc_consumer()  [Method] -----------------------> Start Kafka consumer loop
│       │
│       ├── _create_consumer()  [Method] -----------------> Create KafkaConsumer instance
│       │
│       ├── Infinite poll loop ---------------------------> Wait for Kafka events
│       │
│       ├── _handle_message()  [Method] ------------------> Process each event message
│       │
│       └── consumer.close() -----------------------------> Clean shutdown on stop
│
├── _handle_message()  [Method] --------------------------> Handle one Debezium CDC event
│       │
│       ├── Extract op + rowid ---------------------------> Parse operation type and row ID
│       │
│       ├── [Conditional Branch] op == 'c' or 'u' --------> INSERT or UPDATE operation
│       │       │
│       │       └── _upsert_vector()  [Method] -----------> Re-embed and upsert to vector DB
│       │
│       ├── [Conditional Branch] op == 'd' ---------------> DELETE operation
│       │       │
│       │       └── _delete_vector()  [Method] -----------> Remove vector from DB
│       │
│       └── [Conditional Branch] op == 'r' ---------------> Snapshot event — skip silently
│
├── _upsert_vector()  [Method] ---------------------------> Re-embed row and upsert into vector DB
│       │
│       ├── _serialize_all_rows()  [Function] ------------> Convert row dict to text string
│       │
│       ├── model.encode() --------------------------------> Generate embedding vector
│       │
│       └── vector_db.upsert() ---------------------------> Write updated vector to ChromaDB
│
├── _delete_vector()  [Method] ---------------------------> Delete vector entry from ChromaDB
│       │
│       └── vector_db.delete() ---------------------------> Remove by rowid string
│
├── _create_consumer()  [Method] -------------------------> Instantiate KafkaConsumer
│       │
│       ├── KafkaConsumer()  [Class → Object] ------------> Connect to Kafka broker and topic
│       │
│       └── [Exception Block] ----------------------------> Raise RuntimeError if kafka-python missing
│
└── stop()  [Method] -------------------------------------> Set _running flag to False to exit poll loop

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging

logger = logging.getLogger("haup.cdc_listener")

OP_CREATE   = 'c'
OP_UPDATE   = 'u'
OP_DELETE   = 'd'
OP_SNAPSHOT = 'r'


"""================= Startup class CDCListener ================="""
class CDCListener:

    """================= Startup method __init__ ================="""
    def __init__(self, kafka_broker: str, topic: str,
                 vector_db, model, strategy,
                 poll_timeout_ms: int = 500):
        self.kafka_broker    = kafka_broker
        self.topic           = topic
        self.vector_db       = vector_db
        self.model           = model
        self.strategy        = strategy
        self.poll_timeout_ms = poll_timeout_ms
        self._running        = False
    """================= End method __init__ ================="""

    """================= Startup method start_cdc_consumer ================="""
    def start_cdc_consumer(self) -> None:
        consumer = self._create_consumer()
        logger.info(f"CDC listener started: {self.kafka_broker}  topic={self.topic}")
        self._running = True

        while self._running:
            event = consumer.poll(
                timeout_ms=self.poll_timeout_ms
            )
            if not event:
                continue

            for _, messages in event.items():
                for message in messages:
                    try:
                        self._handle_message(message)
                    except Exception as exc:
                        logger.error(f"CDC handle error: {exc}")

        consumer.close()
        logger.info("CDC listener stopped.")
    """================= End method start_cdc_consumer ================="""

    """================= Startup method _handle_message ================="""
    def _handle_message(self, message) -> None:
        value  = message.value
        op     = value.get('op')
        after  = value.get('after', {}) or {}
        before = value.get('before', {}) or {}

        rowid_col = self.strategy.rowid_col
        rowid     = after.get(rowid_col) or before.get(rowid_col)

        if rowid is None:
            logger.warning(f"No rowid in CDC message (col={rowid_col})")
            return

        if op in (OP_CREATE, OP_UPDATE):
            self._upsert_vector(rowid, after)
        elif op == OP_DELETE:
            self._delete_vector(rowid)
        elif op == OP_SNAPSHOT:
            pass
    """================= End method _handle_message ================="""

    """================= Startup method _upsert_vector ================="""
    def _upsert_vector(self, rowid, new_row: dict) -> None:
        from forward_core.worker_pool_manager import _serialize_all_rows
        text   = _serialize_all_rows([new_row], self.strategy)[0]
        vector = self.model.encode([text])[0]

        self.vector_db.upsert(
            ids        = [str(rowid)],
            embeddings = [vector.tolist()],
            metadatas  = [{'rowid': rowid}],
        )
        logger.info(f"CDC UPSERT rowid={rowid}")
    """================= End method _upsert_vector ================="""

    """================= Startup method _delete_vector ================="""
    def _delete_vector(self, rowid) -> None:
        self.vector_db.delete(ids=[str(rowid)])
        logger.info(f"CDC DELETE rowid={rowid}")
    """================= End method _delete_vector ================="""

    """================= Startup method _create_consumer ================="""
    def _create_consumer(self):
        try:
            from kafka import KafkaConsumer
            import json
            return KafkaConsumer(
                self.topic,
                bootstrap_servers  = [self.kafka_broker],
                value_deserializer = lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset  = 'earliest',
                enable_auto_commit = True,
                group_id           = 'haup-cdc-group',
            )
        except ImportError:
            raise RuntimeError(
                "kafka-python not installed.\n"
                "Run: pip install kafka-python --break-system-packages"
            )
    """================= End method _create_consumer ================="""

    """================= Startup method stop ================="""
    def stop(self) -> None:
        self._running = False
    """================= End method stop ================="""

"""================= End class CDCListener ================="""


if __name__ == "__main__":
    print("CDCListener: import OK")
    print("  Requires Kafka + Debezium at runtime.")