#!/usr/bin/env python3
import argparse
import subprocess
import time
import signal
import sys
import os
import logging
from threading import Thread
from src.config.settings import Settings
from src.utils.zmq_utils import ZmqConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Service_Orchestrator")

class ServiceOrchestrator:
    """Orchestrates the face recognition system services."""
    
    def __init__(self, use_docker=False):
        """Initialize the service orchestrator.
        
        Args:
            use_docker (bool): Whether to use Docker for services
        """
        # Load application settings
        self.settings = Settings()
        
        self.use_docker = use_docker
        self.processes = {}
        self.running = True
        self.deployment_mode = "docker" if use_docker else "local"
        
        # Use configuration for ports
        self.ports = {
            'capture': ZmqConfig.get_port("capture", 5555),
            'image': ZmqConfig.get_port("image", 5556),
            'cropped_face': ZmqConfig.get_port("cropped_face", 5557),
            'recognition': ZmqConfig.get_port("recognition", 5558),
            'add_user': ZmqConfig.get_port("add_user", 5559),
            'add_user_response': ZmqConfig.get_port("add_user_response", 5560),
            'db_request': ZmqConfig.get_port("db_request", 5561),
            'db_response': ZmqConfig.get_port("db_response", 5562),
            'dashboard': self.settings.api.get("port", 8080)
        }
        
        # Host configuration
        self.dashboard_host = self.settings.api.get("host", "0.0.0.0")
        
        # Default host configuration based on deployment mode
        if self.use_docker:
            # Get Docker host configuration
            self.zmq_host = self.settings.zmq.get("hosts", {}).get("docker_host", "host.docker.internal")
        else:
            self.zmq_host = self.settings.zmq.get("hosts", {}).get("local", "localhost")
            
        logger.info(f"Service Orchestrator initialized with mode: {self.deployment_mode}")
        logger.info(f"ZMQ Host: {self.zmq_host}")
        logger.info(f"Dashboard Host: {self.dashboard_host}, Port: {self.ports['dashboard']}")
    
    def _run_command(self, name, command):
        """Run a command in a subprocess.
        
        Args:
            name (str): Name of the service
            command (list): Command to run as list of arguments
        """
        try:
            logger.info(f"Starting {name} service: {' '.join(command)}")
            
            # Start process and register for cleanup
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[name] = process
            
            # Start thread to read and log output
            def log_output():
                for line in iter(process.stdout.readline, ''):
                    if self.running:
                        logger.info(f"[{name}] {line.rstrip()}")
                    else:
                        break
            
            Thread(target=log_output, daemon=True).start()
            
            return process
            
        except Exception as e:
            logger.error(f"Error starting {name} service: {e}")
            return None
    
    def start_capture_service(self):
        """Start the Capture Service."""
        command = [
            "python", "-m", "src.services.capture_service",
            "--capture-port", str(self.ports['capture']),
            "--image-port", str(self.ports['image']),
            "--host", self.zmq_host,
            "--deployment-mode", self.deployment_mode
        ]
        
        return self._run_command("capture", command)
    
    def start_image_processing_service(self):
        """Start the Image Processing Service."""
        command = [
            "python", "-m", "src.services.image_processing_service",
            "--image-port", str(self.ports['image']),
            "--cropped-face-port", str(self.ports['cropped_face']),
            "--host", self.zmq_host,
            "--detector", self.settings.detection.get("default_model", "mediapipe"),
            "--deployment-mode", self.deployment_mode
        ]
        
        return self._run_command("image_processing", command)
    
    def start_recognition_service(self):
        """Start the Recognition Service."""
        command = [
            "python", "-m", "src.services.recognition_service",
            "--cropped-face-port", str(self.ports['cropped_face']),
            "--recognition-port", str(self.ports['recognition']),
            "--add-user-port", str(self.ports['add_user']),
            "--add-user-response-port", str(self.ports['add_user_response']),
            "--host", self.zmq_host,
            "--deployment-mode", self.deployment_mode
        ]
        
        return self._run_command("recognition", command)
    
    def start_dashboard_service(self):
        """Start the Dashboard Service."""
        command = [
            "python", "-m", "src.services.dashboard_service",
            "--capture-port", str(self.ports['capture']),
            "--cropped-face-port", str(self.ports['cropped_face']),
            "--recognition-port", str(self.ports['recognition']),
            "--add-user-port", str(self.ports['add_user']),
            "--add-user-response-port", str(self.ports['add_user_response']),
            "--host", self.zmq_host,
            "--dashboard-host", self.dashboard_host,
            "--dashboard-port", str(self.ports['dashboard']),
            "--deployment-mode", self.deployment_mode
        ]
        
        return self._run_command("dashboard", command)
    
    def start_database_service(self):
        """Start the Database Service."""
        command = [
            "python", "-m", "src.services.database_service",
            "--db-request-port", str(self.ports['db_request']),
            "--db-response-port", str(self.ports['db_response']),
            "--deployment-mode", self.deployment_mode
        ]
        
        return self._run_command("database", command)
    
    def start_all_services(self):
        """Start all services in the correct order."""
        logger.info("Starting all services...")
        
        # Start database service first since other services depend on it
        self.start_database_service()
        time.sleep(1)
        
        # Start services in sequence with small delay between them
        self.start_capture_service()
        time.sleep(1)
        
        self.start_image_processing_service()
        time.sleep(1)
        
        self.start_recognition_service()
        time.sleep(1)
        
        self.start_dashboard_service()
        
        logger.info("All services started")
    
    def stop_all_services(self):
        """Stop all running services."""
        logger.info("Stopping all services...")
        self.running = False
        
        for name, process in self.processes.items():
            if process.poll() is None:  # Process is still running
                logger.info(f"Stopping {name} service...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Killing {name} service that didn't terminate")
                    process.kill()
        
        self.processes = {}
        logger.info("All services stopped")
    
    def wait_for_completion(self):
        """Wait for all processes to complete or for user interrupt."""
        try:
            # Wait for any process to finish or for user interrupt
            while self.running and any(p.poll() is None for p in self.processes.values()):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop_all_services()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System Orchestrator")
    parser.add_argument("--docker", action="store_true", default=False,
                      help="Use Docker for services")
    
    args = parser.parse_args()
    
    # Register signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        orchestrator.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start orchestrator
    orchestrator = ServiceOrchestrator(use_docker=args.docker)
    
    try:
        orchestrator.start_all_services()
        logger.info(f"Dashboard UI available at http://localhost:{orchestrator.ports['dashboard']}")
        orchestrator.wait_for_completion()
    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
    finally:
        orchestrator.stop_all_services()

if __name__ == "__main__":
    main()
