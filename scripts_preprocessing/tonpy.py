import rclpy
from rclpy.node import Node
from dvs_msgs.msg import EventArray
import numpy as np
import argparse
import time
import signal
import sys

class EventSaver(Node):
    def __init__(self, topic_name, output_name):
        super().__init__('event_saver')
        self.subscription = self.create_subscription(
            EventArray, 
            topic_name, 
            self.callback, 
            10  # Queue size
        )
        self.events = []
        self.output_name = output_name
        self.topic_name = topic_name
        self.message_count = 0
        self.event_count = 0
        self.last_message_time = time.time()
        
        # Create a timer to check status every 2 seconds
        self.timer = self.create_timer(2.0, self.check_status)
        
        print(f"EventSaver initialized")
        print(f"Subscribing to topic: {topic_name}")
        print(f"Output file: {output_name}")
        print("Waiting for messages... (Press Ctrl+C to stop and save)")
        
        # Set up signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def callback(self, msg):
        self.message_count += 1
        self.last_message_time = time.time()
        
        # Process events
        events_in_msg = len(msg.events)
        self.event_count += events_in_msg
        
        print(f"Message #{self.message_count}: {events_in_msg} events (Total: {self.event_count})")
        
        for e in msg.events:
            ts = e.ts.sec + 1e-9 * e.ts.nanosec
            self.events.append((e.x, e.y, ts, 1 if e.polarity else -1))

    def check_status(self):
        """Check if we're still receiving messages"""
        time_since_last = time.time() - self.last_message_time
        
        if self.message_count == 0:
            print("No messages received yet. Checking topic availability...")
            self.check_topic_info()
        elif time_since_last > 5.0:
            print(f"Warning: No messages for {time_since_last:.1f} seconds")
        else:
            print(f"Status: {self.message_count} messages, {self.event_count} events total")

    def check_topic_info(self):
        """Check if topic exists and get info"""
        topic_names_and_types = self.get_topic_names_and_types()
        
        print(f"Available topics:")
        for name, types in topic_names_and_types:
            if 'event' in name.lower() or 'davis' in name.lower():
                print(f"  {name} [{', '.join(types)}]")
        
        # Check if our topic exists
        topic_exists = any(name == self.topic_name for name, _ in topic_names_and_types)
        if not topic_exists:
            print(f"ERROR: Topic '{self.topic_name}' not found!")
            print("Available topics that might be relevant:")
            for name, types in topic_names_and_types:
                if any(word in name.lower() for word in ['event', 'davis', 'camera']):
                    print(f"  {name}")

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\nReceived interrupt signal. Saving {len(self.events)} events...")
        self.save()
        sys.exit(0)

    def save(self):
        if len(self.events) == 0:
            print("No events to save!")
            return
            
        print(f"Converting {len(self.events)} events to structured array...")
        arr = np.array(self.events, dtype=[('x', 'u2'), ('y', 'u2'), ('t', 'f8'), ('p', 'i1')])
        
        print(f"Saving to {self.output_name}...")
        np.savez_compressed(self.output_name, events=arr)
        
        print(f"Successfully saved {len(self.events)} events to {self.output_name}")
        
        # Print some statistics
        if len(self.events) > 0:
            x_vals = arr['x']
            y_vals = arr['y']
            t_vals = arr['t']
            p_vals = arr['p']
            
            print(f"Statistics:")
            print(f"  X range: {x_vals.min()} - {x_vals.max()}")
            print(f"  Y range: {y_vals.min()} - {y_vals.max()}")
            print(f"  Time range: {t_vals.min():.6f} - {t_vals.max():.6f} seconds")
            print(f"  Duration: {t_vals.max() - t_vals.min():.6f} seconds")
            print(f"  Polarities: {np.unique(p_vals)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, required=True, help='ROS 2 topic to listen to')
    parser.add_argument('--out', type=str, required=True, help='Output .npz filename')
    parser.add_argument('--max-events', type=int, default=None, help='Stop after this many events')
    parser.add_argument('--timeout', type=float, default=None, help='Stop after this many seconds')
    args = parser.parse_args()

    print("Initializing ROS2...")
    rclpy.init()
    
    try:
        node = EventSaver(topic_name=args.topic, output_name=args.out)
        
        start_time = time.time()
        
        while rclpy.ok():
            # Spin once with timeout
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # Check exit conditions
            if args.max_events and node.event_count >= args.max_events:
                print(f"Reached maximum events ({args.max_events}). Stopping...")
                break
                
            if args.timeout and (time.time() - start_time) >= args.timeout:
                print(f"Reached timeout ({args.timeout}s). Stopping...")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("Cleaning up...")
        if 'node' in locals():
            node.save()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()