"""
Neo4j database connection and management module.
"""

from neo4j import GraphDatabase, Driver
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import NEO4J_CONFIG


class Neo4jConnection:
    """Manages Neo4j database connection."""
    
    _instance: Optional['Neo4jConnection'] = None
    _driver: Optional[Driver] = None
    
    def __new__(cls):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self) -> Driver:
        """Establish connection to Neo4j database."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_CONFIG.uri,
                auth=(NEO4J_CONFIG.username, NEO4J_CONFIG.password)
            )
            self._driver.verify_connectivity()
        return self._driver
    
    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver, connecting if necessary."""
        if self._driver is None:
            self.connect()
        return self._driver
    
    def close(self):
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def run_query(self, query: str, **params) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dictionaries."""
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]
    
    def run_write_query(self, query: str, **params) -> None:
        """Execute a write query."""
        with self.driver.session() as session:
            session.run(query, **params)
    
    def is_connected(self) -> bool:
        """Check if the database connection is active."""
        try:
            if self._driver:
                self._driver.verify_connectivity()
                return True
        except Exception:
            pass
        return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics."""
        stats = {}
        try:
            # Count nodes by label
            labels_query = "CALL db.labels() YIELD label RETURN label"
            labels = self.run_query(labels_query)
            
            for label_record in labels:
                label = label_record['label']
                count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                count = self.run_query(count_query)
                stats[label] = count[0]['count'] if count else 0
        except Exception as e:
            stats['error'] = str(e)
        
        return stats


# Global database connection instance
db = Neo4jConnection()


def get_db() -> Neo4jConnection:
    """Get the database connection instance."""
    return db
