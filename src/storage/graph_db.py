"""
NetworkX knowledge graph operations for relationship management
"""

import networkx as nx
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import Node, Relationship, RelationshipType


class GraphDB:
    """NetworkX knowledge graph operations"""
    
    def __init__(self, settings):
        self.settings = settings
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.graph_file = Path(settings.data_dir) / "knowledge_graph.pkl"
    
    async def initialize(self):
        """Initialize NetworkX graph"""
        # Load existing graph if it exists
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
            except Exception as e:
                # If loading fails, start with empty graph
                self.graph = nx.DiGraph()
        else:
            # Ensure data directory exists
            self.graph_file.parent.mkdir(parents=True, exist_ok=True)
    
    def add_node(self, node: Node):
        """Add context node to knowledge graph
        
        Args:
            node: Node object with context_id, description, category, tags, and other attributes
            
        Raises:
            ValueError: If context_id already exists
        """
        if node.context_id in self.graph:
            raise ValueError(f"Context ID '{node.context_id}' already exists in graph")
            
        # Convert Node to dict for NetworkX storage
        node_data = {
            "description": node.description,
            "category": node.category,
            "tags": node.tags,
            "updated_at": node.updated_at,
            **node.attributes
        }
        
        self.graph.add_node(node.context_id, **node_data)
        self._save_graph()
    
    def add_relationship(self, relationship: Relationship):
        """Add relationship edge to knowledge graph
        
        Args:
            relationship: Relationship object with source, target, and type info
            
        Raises:
            ValueError: If source or target nodes don't exist
        """
        if relationship.source_id not in self.graph:
            raise ValueError(f"Source node '{relationship.source_id}' not found in graph")
        if relationship.target_id not in self.graph:
            raise ValueError(f"Target node '{relationship.target_id}' not found in graph")
            
        edge_data = {
            "relationship": relationship.rel_type.value,
            "updated_at": relationship.updated_at,
            **relationship.attributes
        }
        
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            **edge_data
        )
        self._save_graph()
    
    def find_related(self, context_id: str, max_depth: int = 3) -> List[Node]:
        """Find related contexts via graph traversal
        
        Args:
            context_id: Starting context ID
            max_depth: Maximum traversal depth
            
        Returns:
            List of related Node objects
        """
        if context_id not in self.graph:
            return []
        
        related_nodes = []
        visited = set()
        
        # BFS traversal to find related nodes
        queue = [(context_id, 0)]
        visited.add(context_id)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            # Get neighbors (both incoming and outgoing)
            neighbors = list(self.graph.successors(current_id)) + list(self.graph.predecessors(current_id))
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    
                    # Get the node and add to results
                    node = self.get_node(neighbor_id)
                    if node:
                        related_nodes.append(node)
        
        return related_nodes
    
    def remove_node(self, context_id: str):
        """Remove context node and all its edges"""
        if context_id in self.graph:
            self.graph.remove_node(context_id)
            self._save_graph()
    
    def get_node(self, context_id: str) -> Optional[Node]:
        """Get node as Node object"""
        if context_id not in self.graph:
            return None
            
        node_data = dict(self.graph.nodes[context_id])
        
        # Extract known fields and put rest in attributes
        known_fields = {"description", "category", "tags", "updated_at"}
        attributes = {k: v for k, v in node_data.items() if k not in known_fields}
        
        return Node(
            context_id=context_id,
            description=node_data.get("description", ""),
            category=node_data.get("category"),
            tags=node_data.get("tags", []),
            updated_at=node_data.get("updated_at"),
            attributes=attributes
        )
    
    def update_node(self, node: Node):
        """Update existing node"""
        if node.context_id not in self.graph:
            raise ValueError(f"Context ID '{node.context_id}' not found in graph")
            
        # Update the node data
        node_data = {
            "description": node.description,
            "category": node.category,
            "tags": node.tags,
            "updated_at": node.updated_at,
            **node.attributes
        }
        
        self.graph.add_node(node.context_id, **node_data)
        self._save_graph()
    
    def node_exists(self, context_id: str) -> bool:
        """Check if a node exists in the graph"""
        return context_id in self.graph
    
    def get_all_context_ids(self) -> List[str]:
        """Get all context IDs in the graph"""
        return list(self.graph.nodes())
    
    def _save_graph(self):
        """Save graph to disk"""
        try:
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            # TODO: Add proper logging
            print(f"Failed to save graph: {e}") 