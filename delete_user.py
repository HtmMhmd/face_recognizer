#!/usr/bin/env python3

import argparse
import sys
from database import FaceDatabase

class UserNotFoundException(Exception):
    """Exception raised when a user is not found in the database."""
    pass

def delete_user(username: str, force: bool = False) -> bool:
    """
    Delete a user from the face recognition database by username.
    
    Args:
        username (str): Username to delete from the database
        force (bool): If True, don't raise an exception when user doesn't exist
        
    Returns:
        bool: True if deletion was successful
        
    Raises:
        UserNotFoundException: If the user doesn't exist and force=False
    """
    try:
        # Initialize the database connection
        database_handler = FaceDatabase()
        
        # Check if user exists
        users = database_handler.get_all_users()
        if username not in users:
            if force:
                print(f"User '{username}' not found in database. Nothing to delete.")
                return False
            else:
                raise UserNotFoundException(f"User '{username}' not found in database")
        
        # Delete the user
        database_handler.delete_user(username)
        print(f"✅ User '{username}' has been successfully deleted from the database.")
        return True
        
    except UserNotFoundException:
        # Re-raise the exception
        raise
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return False

def list_users() -> list:
    """
    List all users currently in the database.
    
    Returns:
        list: List of usernames in the database
    """
    try:
        database_handler = FaceDatabase()
        users = database_handler.get_all_users()
        return users
    except Exception as e:
        print(f"Error listing users: {str(e)}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a user from the face recognition system")
    
    # Create a mutually exclusive group for the actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--delete", "-d", type=str, help="Username to delete from the database")
    group.add_argument("--list", "-l", action="store_true", help="List all users in the database")
    
    # Add force option for delete
    parser.add_argument("--force", "-f", action="store_true", help="Don't raise an error if user doesn't exist")
    
    args = parser.parse_args()
    
    # List users
    if args.list:
        users = list_users()
        if users:
            print("Users in the database:")
            for user in users:
                print(f"  - {user}")
            print(f"Total users: {len(users)}")
        else:
            print("No users found in the database.")
    
    # Delete a user
    if args.delete:
        try:
            delete_user(args.delete, args.force)
        except UserNotFoundException as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
