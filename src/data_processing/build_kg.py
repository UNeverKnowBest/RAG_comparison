from collections import defaultdict

import pandas as pd
from config import KG_DIR, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from neo4j import GraphDatabase
from tqdm import tqdm

batch_size = 2000


class KGBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print("Connected to Neo4j")

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("Clearing database")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def setup_schema(self):
        print("Setting up schema")
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """)
            session.run("""
                CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                FOR (e:Entity) ON EACH [e.name, e.mentions]
            """)
        print("Constraints are created")

    def import_entities(self, df, size=batch_size):
        print(f"Importing {len(df)} entities")
        records = df.to_dict("records")
        with self.driver.session() as session:
            for i in tqdm(range(0, len(records), size), desc="Entities"):
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (e:Entity {name: row.name})
                    SET e.mentions = row.mentions,
                        e.n_mentions = row.n_mentions
                """,
                    batch=records[i : i + size],
                )

    def import_relations(self, df, size=batch_size):
        print(f"Importing {len(df)} relations")
        records = df.to_dict("records")
        grouped = defaultdict(list)
        for r in records:
            grouped[r["type"]].append(r)

        with self.driver.session() as session:
            for rel_type, rel_records in tqdm(grouped.items(), desc="Relation types"):
                for i in range(0, len(rel_records), size):
                    query = f"""
                        UNWIND $batch AS row
                        MATCH (h:Entity {{name: row.head}})
                        MATCH (t:Entity {{name: row.tail}})
                        MERGE (h)-[r:`{rel_type}`]->(t)
                        SET r.original_relation = row.original_relation,
                            r.question_id       = row.question_id,
                            r.step              = row.step,
                            r.question_type     = row.question_type
                    """
                    session.run(query, batch=rel_records[i : i + size])


def main():
    nodes_df = pd.read_csv(KG_DIR / "nodes.csv")
    relations_df = pd.read_csv(KG_DIR / "relations.csv")
    builder = KGBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    builder.clear_database()
    builder.setup_schema()
    builder.import_entities(nodes_df)
    builder.import_relations(relations_df)
    print("Knowledge graph complete")
    builder.close()


if __name__ == "__main__":
    main()
