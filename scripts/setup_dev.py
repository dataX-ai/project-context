#!/usr/bin/env python3
"""
Development environment setup script for Project Context MCP Server
"""

import os
import shutil
from pathlib import Path


def setup_development_environment():
    """Set up development environment"""
    print("Setting up Project Context MCP Server development environment...")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # LanceDB directory (will be created automatically by LanceDB)
    lancedb_dir = data_dir / "lancedb"
    print(f"✅ LanceDB will be created at: {lancedb_dir}")
    
    print(f"✅ Created data directory: {data_dir}")
    
    # Copy environment template
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print(f"✅ Created {env_file} from template")
    
    print("\n🚀 Development environment ready!")
    print("\nNext steps:")
    print("1. Install dependencies: uv sync")
    print("2. Activate environment: source .venv/bin/activate (or uv run for commands)")
    print("3. Run the server: uv run python -m src.main")
    print("\n📊 Database Setup:")
    print("• LanceDB: High-performance vector database with Apache Arrow")
    print("• NetworkX: Knowledge graph for relationships")
    print("• Sentence Transformers: Local embeddings (all-MiniLM-L6-v2)")
    print("• Storage: All data stored locally in ./data/")


if __name__ == "__main__":
    setup_development_environment() 