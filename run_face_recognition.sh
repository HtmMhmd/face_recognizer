#!/usr/bin/env bash

# Face Recognition System Runner Script

# Default configuration
MODE="local"  # local, docker, or simulator
SERVICES=true
SIMULATOR=false
START_DASHBOARD=true

# Display help
function show_help {
    echo "Face Recognition System Runner"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --mode MODE        Mode to run (local, docker, simulator)"
    echo "  -s, --services         Start all services (default: true)"
    echo "  --no-services          Don't start services"
    echo "  --simulator            Run with simulator instead of real camera"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --mode local        Run all services locally"
    echo "  $0 --mode docker       Run using Docker Compose"
    echo "  $0 --simulator         Run with simulator (fake camera)"
    echo "  $0 --no-services       Don't start services (if you want to start them manually)"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--mode)
            MODE="$2"
            shift
            shift
            ;;
        -s|--services)
            SERVICES=true
            shift
            ;;
        --no-services)
            SERVICES=false
            shift
            ;;
        --simulator)
            SIMULATOR=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Function to check if a command is available
function check_command {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed."
        exit 1
    fi
}

# Display configuration
echo "Face Recognition System Configuration:"
echo "  Mode: $MODE"
echo "  Start Services: $SERVICES"
echo "  Use Simulator: $SIMULATOR"
echo ""

# Check for required tools
check_command "python3"
if [ "$MODE" = "docker" ]; then
    check_command "docker"
    check_command "docker-compose"
fi

# Function to run services locally
function run_local {
    echo "Starting Face Recognition System locally..."
    
    if [ "$SERVICES" = true ]; then
        # Run the service orchestrator
        if [ "$SIMULATOR" = true ]; then
            echo "Starting simulator..."
            # Start simulator in background
            python3 face_recognition_simulator.py &
            SIMULATOR_PID=$!
            
            # Give the simulator time to initialize
            sleep 2
        fi
        
        # Start the service orchestrator
        echo "Starting service orchestrator..."
        python3 service_orchestrator.py
        
        # Clean up simulator if it was started
        if [ "$SIMULATOR" = true ] && [ -n "$SIMULATOR_PID" ]; then
            echo "Stopping simulator..."
            kill $SIMULATOR_PID 2>/dev/null || true
        fi
    else
        echo "Services not started as requested."
        echo "To start services manually, run:"
        echo "  python3 service_orchestrator.py"
    fi
}

# Function to run with Docker Compose
function run_docker {
    echo "Starting Face Recognition System with Docker Compose..."
    
    if [ "$SERVICES" = true ]; then
        docker-compose -f docker-compose-zmq.yaml build
        docker-compose -f docker-compose-zmq.yaml up
    else
        echo "Services not started as requested."
        echo "To start services manually with Docker, run:"
        echo "  docker-compose -f docker-compose-zmq.yaml up"
    fi
}

# Run the system in the specified mode
case $MODE in
    "local")
        run_local
        ;;
    "docker")
        run_docker
        ;;
    "simulator")
        SIMULATOR=true
        run_local
        ;;
    *)
        echo "Unknown mode: $MODE"
        show_help
        ;;
esac

echo "Face Recognition System stopped"
