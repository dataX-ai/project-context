"""
Test script for Neo4j GraphBuilder with sample conversation strings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from storage.graph import GraphBuilder
from config import settings


def test_simple_conversation():
    """Test with a simple design conversation"""
    
    conversation = """
    The Frontend React application uses Axios for HTTP requests to the backend.
    The Backend API service decides to implement Express.js framework for routing.
    The Database layer component uses PostgreSQL for persistent data storage.
    The Backend API service depends on the Database layer for user data.
    The Authentication service decides to use JWT tokens for session management.
    The Frontend component depends on the Authentication service for user login.
    The Notification service uses Redis for message queuing.
    The Backend API service uses the Notification service for sending alerts.
    The Logging component decides to implement Winston for application logging.
    """
    
    print("Testing simple conversation:")
    print(conversation)
    print("-" * 50)
    
    return conversation


def test_complex_conversation():
    """Test with a more complex design discussion"""
    
    conversation = """
    The Frontend React application uses Axios for HTTP requests to the backend.
    The Backend API service decides to implement Express.js framework for routing.
    The Database layer component uses PostgreSQL for persistent data storage.
    The Backend API service depends on the Database layer for user data.
    The Authentication service decides to use JWT tokens for session management.
    The Frontend component depends on the Authentication service for user login.
    The Notification service uses Redis for message queuing.
    The Backend API service uses the Notification service for sending alerts.
    The Logging component decides to implement Winston for application logging.
    """
    
    print("Testing complex conversation:")
    print(conversation)
    print("-" * 50)
    
    return conversation


def test_architecture_discussion():
    """Test with architectural decision conversation"""
    
    conversation = """
    I'm thinking of building my backend using a microservices architecture. I've decided to use Docker containers for deployment since they make scaling much easier. For the load balancer, I'm planning to use NGINX for traffic distribution - it's reliable and I've used it before. The API Gateway will depend on multiple microservices for request routing, so I need to make sure that's well designed. For monitoring, I want to implement Prometheus for metrics collection because everyone says it's great for microservices. I'm also considering MongoDB for document storage in my database cluster since our data isn't really relational. For caching, I think Redis Cluster would work well for distributed caching across services. And for the message queue component, I've been reading about Apache Kafka for event streaming - seems like the right choice for handling high throughput events between services.
    """
    
    print("Testing architecture discussion:")
    print(conversation)
    print("-" * 50)
    
    return conversation


def test_basic_neo4j_write():
    """Test basic Neo4j write functionality"""
    
    try:
        with GraphBuilder() as graph_builder:
            print("🧪 Testing basic Neo4j write...")
            
            # Try to create a simple test node
            create_query = "CREATE (test:TestNode {name: 'test', created: timestamp()}) RETURN test"
            result = graph_builder.query_graph(create_query)
            print(f"✅ Test node created: {result}")
            
            # Check if it exists
            check_query = "MATCH (test:TestNode {name: 'test'}) RETURN count(test) as count"
            count_result = graph_builder.query_graph(check_query)
            print(f"📊 Test nodes found: {count_result[0]['count'] if count_result else 0}")
            
            # Clean up
            cleanup_query = "MATCH (test:TestNode {name: 'test'}) DELETE test"
            graph_builder.query_graph(cleanup_query)
            print("🧹 Test node cleaned up")
            
            return True
            
    except Exception as e:
        print(f"❌ Basic Neo4j write test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_graph_test(conversation_text: str, test_name: str):
    """Run graph building test with given conversation"""
    
    if not settings.azure_openai_endpoint:
        print("⚠️  Please set AZURE_OPENAI_ENDPOINT in .env file")
        return
    
    if not settings.azure_openai_api_key:
        print("⚠️  Please set AZURE_OPENAI_API_KEY in .env file")
        return
        
    if not settings.azure_openai_deployment_name:
        print("⚠️  Please set AZURE_OPENAI_DEPLOYMENT_NAME in .env file")
        return
    
    # Check Azure OpenAI configuration
    print(f"🔗 Using Azure OpenAI Endpoint: {settings.azure_openai_endpoint}")
    print(f"🤖 Using Deployment: {settings.azure_openai_deployment_name}")
    print(f"📋 Using API Version: {settings.azure_openai_api_version}")
    
    try:
        with GraphBuilder() as graph_builder:
            
            print(f"🚀 Building graph for: {test_name}")
            
            # Check database connectivity first
            try:
                test_query = "RETURN 1 as test"
                graph_builder.query_graph(test_query)
                print("✅ Neo4j connection successful")
            except Exception as e:
                print(f"❌ Neo4j connection failed: {e}")
                return
            
            # Check what's in the database before building
            existing_nodes = graph_builder.query_graph("MATCH (n) RETURN count(n) as node_count")
            print(f"📊 Existing nodes in database: {existing_nodes[0]['node_count'] if existing_nodes else 0}")
            
            # Build the graph
            try:
                print("🔨 Starting graph building...")
                graph_builder.build_graph_from_text_sync(conversation_text)
                print("✅ Graph building completed")
            except Exception as e:
                print(f"❌ Graph building failed: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Check what was created
            new_nodes = graph_builder.query_graph("MATCH (n) RETURN count(n) as node_count")
            print(f"📊 Total nodes after building: {new_nodes[0]['node_count'] if new_nodes else 0}")
            
            # Check what labels exist
            labels_query = "CALL db.labels()"
            labels = graph_builder.query_graph(labels_query)
            print(f"🏷️  Available labels: {[l['label'] for l in labels]}")
            
            # Check what relationship types exist
            rel_types_query = "CALL db.relationshipTypes()"
            rel_types = graph_builder.query_graph(rel_types_query)
            print(f"🔗 Available relationship types: {[r['relationshipType'] for r in rel_types]}")
            
            # Query results only if we have data
            if new_nodes and new_nodes[0]['node_count'] > 0:
                print("\n📊 Results:")
                
                components = graph_builder.get_components()
                print(f"Components ({len(components)}): {[c['name'] for c in components]}")
                
                technologies = graph_builder.get_technologies()
                print(f"Technologies ({len(technologies)}): {[t['name'] for t in technologies]}")
                
                decisions = graph_builder.get_decisions()
                print(f"Decisions ({len(decisions)}): {[d['summary'] for d in decisions]}")
                
                # Get all relationships
                all_relationships = graph_builder.get_all_relationships()
                print(f"\n🔗 All Relationships: {len(all_relationships)}")
                for rel in all_relationships[:5]:  # Show top 5
                    print(f"  {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
            else:
                print("❌ No data was created in the database")
            
            print("✅ Test completed!\n")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


def test_custom_query():
    """Test custom Cypher queries"""
    
    try:
        with GraphBuilder() as graph_builder:
            
            print("🔍 Testing custom queries:")
            
            # Find components that use multiple technologies
            query = """
            MATCH (c:Component)-[r:USES]->(t:Technology)
            WITH c, count(t) as tech_count
            WHERE tech_count > 1
            RETURN c.name as component, tech_count
            ORDER BY tech_count DESC
            """
            
            results = graph_builder.query_graph(query)
            print("Components using multiple technologies:")
            for result in results:
                print(f"  {result['component']}: {result['tech_count']} technologies")
            
            # Find dependency chains
            dependency_query = """
            MATCH path = (a:Component)-[:DEPENDS_ON*1..3]->(b:Component)
            RETURN [node in nodes(path) | node.name] as dependency_chain
            LIMIT 5
            """
            
            chains = graph_builder.query_graph(dependency_query)
            print("\nDependency chains:")
            for chain in chains:
                print(f"  {' -> '.join(chain['dependency_chain'])}")
            
    except Exception as e:
        print(f"❌ Error during custom query test: {e}")


def main():
    """Run all tests"""
    
    print("🧪 Neo4j GraphBuilder Test Suite")
    print("=" * 50)
    
    # Test basic Neo4j functionality first
    if not test_basic_neo4j_write():
        print("❌ Basic Neo4j write test failed - stopping here")
        return
    
    # Test different conversation types
    conversations = [
        (test_simple_conversation(), "Simple Conversation"),
        # (test_complex_conversation(), "Complex Conversation"),
        # (test_architecture_discussion(), "Architecture Discussion")
    ]
    
    for conversation, name in conversations:
        run_graph_test(conversation, name)
        print("-" * 50)
    
    # Test custom queries
    test_custom_query()
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    main() 