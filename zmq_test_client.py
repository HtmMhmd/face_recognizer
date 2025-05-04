import zmq
import time
import sys
import cv2
import numpy as np

class ZMQClient:
    def __init__(self, server_address="tcp://localhost:5555"):
        """Initialize ZMQ client to connect to the automation server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        print(f"Connected to ZMQ server at {server_address}")
        
    def send_init(self):
        """Send 'init' command and wait for response."""
        print("Sending 'init' command...")
        self.socket.send_string("init")
        response = self.socket.recv_string()
        print(f"Received response: {response}")
        return response
        
    def send_capture(self):
        """Send 'capture' command and wait for image response."""
        print("Sending 'capture' command...")
        self.socket.send_string("capture")
        response = self.socket.recv()
        
        # Check if we got a binary response (image) or a string (error)
        try:
            # Try to convert response to string - if it works, it's a string message
            response_str = response.decode('utf-8')
            print(f"Received text response: {response_str}")
            return None
        except UnicodeDecodeError:
            # If decoding fails, it's likely an image
            print("Received image response")
            try:
                # Convert bytes to numpy array
                img_array = np.frombuffer(response, dtype=np.uint8)
                # Decode the image
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                # Save the image to a file
                cv2.imwrite("captured_image.jpg", img)
                print("Black image saved as 'captured_image.jpg'")
                return img
            except Exception as e:
                print(f"Error processing image response: {e}")
                return None
                
    def close(self):
        """Close ZMQ connection."""
        self.socket.close()
        self.context.term()
        print("Connection closed")

def main():
    # Default to localhost if no address provided
    server_address = sys.argv[1] if len(sys.argv) > 1 else "tcp://localhost:5555"
    
    client = ZMQClient(server_address)
    
    try:
        # Demo sequence - send init, wait, then send capture
        init_response = client.send_init()
        
        if init_response == "1":
            print("Init successful, container started")
            
            # Wait a bit before sending capture command
            print("Waiting 5 seconds...")
            time.sleep(5)
            
            # Now send capture command
            client.send_capture()
        else:
            print(f"Init failed with response: {init_response}")
            
    finally:
        client.close()

if __name__ == "__main__":
    main()