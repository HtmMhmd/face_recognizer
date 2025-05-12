#!/usr/bin/env python3
"""
Test script for the face recognition database service using ZMQ communication.
This script tests the basic database operations like adding users, retrieving users,
and deleting users through ZMQ communication.
"""

import time
import numpy as np
import logging
from src.database.api.db_client import DatabaseClient
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DB_Service_Test")

def main():
    """Test the database service functionality."""
    logger.info("Starting database service test...")
    
    # Initialize database client
    client = DatabaseClient(deployment_mode="local")
    
    # Create a test user with a random embedding
    username = f"test_user_{int(time.time())}"
    embedding = np.random.rand(512).astype(np.float32)
    
    try:
        # Test adding a user
        logger.info(f"Adding user {username}...")
        result = client.add_user(username, embedding)
        logger.info(f"Add user result: {result}")
        
        # Test getting the user back
        logger.info(f"Getting user {username}...")
        user = client.get_user(username)
        if user:
            logger.info(f"User found: {user.get('date_added', '')}")
            
            # Verify embedding
            retrieved_embedding = user.get('embedding')
            if retrieved_embedding is not None:
                distance = np.linalg.norm(embedding - retrieved_embedding)
                logger.info(f"Embedding distance: {distance}")
                
                if distance < 0.001:
                    logger.info("Embedding verification: SUCCESS")
                else:
                    logger.error("Embedding verification: FAILED")
        else:
            logger.error(f"User {username} not found!")
        
        # Test getting all users
        logger.info("Getting all users...")
        all_users = client.get_all_embeddings()
        logger.info(f"Total users: {len(all_users)}")
        
        # Test updating last login
        logger.info(f"Updating last login for {username}...")
        result = client.update_last_login(username)
        logger.info(f"Update last login result: {result}")
        
        # Get user again to check the last login timestamp
        updated_user = client.get_user(username)
        if updated_user:
            logger.info(f"Last login: {updated_user.get('last_login', 'Not available')}")
        
        # Test deleting the user
        logger.info(f"Deleting user {username}...")
        result = client.delete_user(username)
        logger.info(f"Delete user result: {result}")
        
        # Verify user was deleted
        deleted_user = client.get_user(username)
        if deleted_user is None:
            logger.info("User deletion verification: SUCCESS")
        else:
            logger.error("User deletion verification: FAILED")
            
    except Exception as e:
        logger.error(f"Error during database service test: {e}")
    finally:
        # Close the client
        client.close()
        logger.info("Database service test completed")

if __name__ == "__main__":
    main()
