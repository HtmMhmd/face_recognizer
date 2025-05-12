# Face Recognition System ZMQ Communication Architecture

This document describes the ZMQ-based communication architecture for the distributed face recognition system. The system consists of multiple services that communicate with each other using ZMQ to process images, detect faces, recognize users, and manage database operations.

## Services Overview

The system consists of the following services:

1. **Capture Service**: Captures video frames from a camera and sends them to the Image Processing Service.
2. **Image Processing Service**: Processes raw frames, detects faces, and sends cropped faces to the Recognition Service.
3. **Recognition Service**: Recognizes faces using embeddings and sends results to the Dashboard Service.
4. **Database Service**: Handles user data storage and retrieval operations via ZMQ messaging.
5. **Dashboard Service**: Provides a web interface for interaction and displays results.

## ZMQ Communication Patterns

The system uses the following ZMQ communication patterns:

1. **Publisher-Subscriber**: Used for one-to-many distribution of messages.
2. **Request-Reply**: Used for command-based operations with acknowledgment.

## Message Topics

| Topic             | From                  | To                        | Purpose                       |
| -----             | ----                  | --                        | -------                       |
| Capture           | Dashboard Service     | Capture Service           | Trigger frame capture         |
| Image             | Capture Service       | Image Processing Service  | Send captured frames          |
| CroppedFace       | Image Processing Service | Recognition Service    | Send detected face crops      |
| RecognitionResult | Recognition Service   | Dashboard Service         | Send recognition results      |
| AddUserRequest    | Dashboard Service     | Recognition Service       | Request to add a new user     |
| AddUserResponse   | Recognition Service   | Dashboard Service         | Result of add user operation  |
| DBRequest         | Any Service           | Database Service          | Database operation request    |
| DBResponse        | Database Service      | Any Service               | Database operation response   |
| StatusUpdate      | Any Service           | Dashboard Service         | Service status updates        |

## Port Configuration

The services use the following port configuration:

| Port | Purpose                                                |
| ---- | -------                                                |
| 5555 | Capture Service        -> Receive capture commands     |
| 5556 | Capture Service        -> Publish images               |
| 5557 | Image Processing Service -> Publish cropped faces      |
| 5558 | Recognition Service    -> Publish recognition results  |
| 5559 | Dashboard Service      -> Publish add user requests    |
| 5560 | Recognition Service    -> Publish add user responses   |
| 5561 | Any Service            -> Publish database requests    |
| 5562 | Database Service       -> Publish database responses   |
| 8080 | Dashboard Service      -> Web interface                |

## Database Service Communication

The Database Service uses a request/response pattern over ZMQ to handle database operations. The following operations are supported:

### Commands:

1. **add_user**: Add a new user with face embedding to the database
   - Parameters: `username`, `embedding` (as bytes), `password` (optional)
   - Response: Success status and message

2. **get_user**: Get user information by username
   - Parameters: `username`
   - Response: User data including embedding, date added, and last login

3. **delete_user**: Delete a user from the database
   - Parameters: `username`
   - Response: Success status and message

4. **get_all_users**: Get all users and their embeddings
   - Parameters: None
   - Response: Dictionary of all users with their data

5. **update_last_login**: Update the last login timestamp for a user
   - Parameters: `username`
   - Response: Success status and message

### Message Format:

**Request**:
```json
{
  "command": "<command_name>",
  "request_id": "<unique_id>",
  "<parameter_name>": "<parameter_value>"
}
```

**Response**:
```json
{
  "request_id": "<unique_id>",
  "success": true/false,
  "message": "<optional_message>",
  "<data_field>": "<data_value>"
}
```

## Deployment

The system can be deployed in two modes:

1. **Local Mode**: All services run on the same machine, using localhost for communication.
2. **Docker Mode**: Services run in separate Docker containers, using container names for networking.

## Configuration

All ZMQ settings are defined in `src/config/config.yaml`, including:

- Port numbers for each service
- Hostnames for different deployment modes
- Topic names for ZMQ communication
- Commands for service operations

## Example Flow

1. The Dashboard Service sends a "Capture" message to the Capture Service.
2. The Capture Service captures a frame and publishes it on the "Image" topic.
3. The Image Processing Service receives the image, detects faces, and publishes cropped faces on the "CroppedFace" topic.
4. The Recognition Service receives the cropped face, extracts embeddings, and queries the Database Service via the "DBRequest" topic.
5. The Database Service processes the request and responds on the "DBResponse" topic.
6. The Recognition Service compares embeddings, identifies the user, and publishes results on the "RecognitionResult" topic.
7. The Dashboard Service receives recognition results and updates the UI.

## Error Handling

Each service implements error handling and logging. If a service encounters an error:

1. The error is logged with appropriate severity.
2. If relevant, a status update is published.
3. For critical errors, the service may attempt to restart or exit gracefully.

## Service Orchestration

The Service Orchestrator manages the lifecycle of all services, including:

1. Starting services in the correct order
2. Monitoring service status
3. Graceful shutdown on termination signals
4. Configuration validation
