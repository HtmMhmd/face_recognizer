#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import subprocess
import time
import signal
import atexit
from src.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Service_Orchestrator")

class ServiceOrchestrator:
    """
    Orchestrates the deployment and management of all face recognition system services.
    """
    def __init__(self, deployment_mode="local", use_docker=False):
        """
        Initialize the service orchestrator.
        
        Args:
            deployment_mode (str): Deployment mode ('local' or 'docker')
            use_docker (bool): Legacy parameter. If True, sets deployment_mode to "docker"
        """
        # Support legacy parameter
        if use_docker:
            deployment_mode = "docker"
        self.settings = Settings()
        self.deployment_mode = deployment_mode
        self.processes = {}
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down services...")
        self.cleanup()
        sys.exit(0)
        
    def start_service(self, service_name, command, env=None):
        """Start a service process."""
        if service_name in self.processes and self.processes[service_name].poll() is None:
            logger.warning(f"Service {service_name} is already running.")
            return
            
        logger.info(f"Starting {service_name}...")
        if env:
            env_vars = os.environ.copy()
            env_vars.update(env)
        else:
            env_vars = os.environ.copy()
            
        process = subprocess.Popen(
            command,
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes[service_name] = process
        logger.info(f"{service_name} started with PID {process.pid}")
        
        # Start logger threads
        self._start_log_threads(service_name, process)
        
    def _start_log_threads(self, service_name, process):
        """Start threads to capture and log stdout/stderr from services."""
        import threading
        
        def log_output(stream, log_prefix):
            for line in stream:
                logger.info(f"{log_prefix}: {line.strip()}")
                
        threading.Thread(
            target=log_output,
            args=(process.stdout, f"{service_name}-STDOUT"),
            daemon=True
        ).start()
        
        threading.Thread(
            target=log_output,
            args=(process.stderr, f"{service_name}-STDERR"),
            daemon=True
        ).start()
        
    def deploy_local(self):
        """Deploy services locally as separate processes."""
        logger.info("Deploying services locally...")
        
        # Start Database Service
        self.start_service(
            "db_service",
            ["python", "-m", "src.services.database_service", 
             "--db-request-port", "5561", 
             "--db-response-port", "5562"]
        )
        time.sleep(2)  # Wait for database service to initialize
        
        # Start Dashboard Service 
        self.start_service(
            "dashboard_service",
            ["python", "-m", "src.services.dashboard_service", 
             "--capture-port", "5555",
             "--cropped-face-port", "5557",
             "--recognition-port", "5558",
             "--add-user-port", "5559",
             "--add-user-response-port", "5560",
             "--host", "localhost",
             "--dashboard-host", "0.0.0.0",
             "--dashboard-port", "8080"]
        )
        time.sleep(1)
        
        # Start Capture Service
        self.start_service(
            "capture_service",
            ["python", "-m", "src.services.capture_service", 
             "--capture-port", "5555", 
             "--image-port", "5556", 
             "--host", "localhost"]
        )
        time.sleep(1)
        
        # Start Image Processing Service
        self.start_service(
            "image_processing_service",
            ["python", "-m", "src.services.image_processing_service",
             "--image-port", "5556",
             "--cropped-face-port", "5557",
             "--host", "localhost",
             "--detector", "mediapipe"]
        )
        time.sleep(1)
        
        # Start Recognition Service
        self.start_service(
            "recognition_service",
            ["python", "-m", "src.services.recognition_service",
             "--cropped-face-port", "5557",
             "--recognition-port", "5558",
             "--add-user-port", "5559",
             "--add-user-response-port", "5560",
             "--host", "localhost"]
        )
        
        logger.info("All services deployed locally!")
        logger.info("Dashboard available at: http://localhost:8080")
        
    def deploy_docker(self):
        """Deploy services using Docker Compose."""
        logger.info("Deploying services with Docker Compose...")
        
        # Run docker-compose
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker-compose-enhanced.yaml", "up", "-d"],
                check=True
            )
            logger.info("Docker Compose services started successfully!")
            logger.info("Dashboard available at: http://localhost:8080")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker Compose services: {e}")
            sys.exit(1)
            
    def check_service_status(self):
        """Check status of all running services."""
        if self.deployment_mode == "docker":
            # Check docker services
            try:
                result = subprocess.run(
                    ["docker-compose", "-f", "docker-compose-enhanced.yaml", "ps"],
                    check=True,
                    stdout=subprocess.PIPE,
                    universal_newlines=True
                )
                logger.info("Docker services status:")
                logger.info("\n" + result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to check Docker services status: {e}")
        else:
            # Check local processes
            for name, process in self.processes.items():
                status = "RUNNING" if process.poll() is None else f"STOPPED (exit code {process.poll()})"
                logger.info(f"{name}: {status}")
                
    def cleanup(self):
        """Clean up all running services."""
        logger.info("Cleaning up services...")
        
        if self.deployment_mode == "docker":
            # Stop docker services
            try:
                subprocess.run(
                    ["docker-compose", "-f", "docker-compose-enhanced.yaml", "down"],
                    check=True
                )
                logger.info("Docker Compose services stopped successfully!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop Docker Compose services: {e}")
        else:
            # Terminate local processes
            for name, process in self.processes.items():
                if process.poll() is None:  # Process is still running
                    logger.info(f"Terminating {name}...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        logger.info(f"{name} terminated")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{name} did not terminate gracefully, killing...")
                        process.kill()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System Service Orchestrator")
    parser.add_argument("--mode", choices=["local", "docker"], default="local",
                        help="Deployment mode: local (separate processes) or docker (containers)")
    parser.add_argument("--action", choices=["start", "stop", "status"], default="start",
                        help="Action to perform: start, stop, or check status")
    
    args = parser.parse_args()
    
    orchestrator = ServiceOrchestrator(deployment_mode=args.mode)
    
    if args.action == "start":
        if args.mode == "docker":
            orchestrator.deploy_docker()
        else:
            orchestrator.deploy_local()
    elif args.action == "stop":
        orchestrator.cleanup()
    elif args.action == "status":
        orchestrator.check_service_status()

if __name__ == "__main__":
    main()