import numpy as np
import torch

def sample_and_mask_events(events, N=1024):
    """
    Sample N events from the input events array.
    - If len(events) < N: Pad with -1 and mask False for padding
    - If len(events) >= N: Randomly sample N events
    
    Args:
        events: Structured numpy array with fields ['x', 'y', 't', 'p']
        N: Number of events to sample (default 1024)
    
    Returns:
        sampled_events: (N, 4) array of [x, y, t, p]
        mask: (N,) boolean array (True for valid events, False for padding)
    """
    
    num_events = len(events)
    
    # Initialize output arrays
    sampled_events = np.full((N, 4), -1.0, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    
    if num_events == 0:
        # No events at all - return empty (all -1, all False)
        return sampled_events, mask
    
    elif num_events < N:
        # Fewer events than N: use all events and pad
        sampled_events[:num_events, 0] = events['x']
        sampled_events[:num_events, 1] = events['y']
        sampled_events[:num_events, 2] = events['t']
        sampled_events[:num_events, 3] = events['p']
        mask[:num_events] = True
    
    else:
        # More events than N: randomly sample N events
        indices = np.random.choice(num_events, N, replace=False)
        sampled_events[:, 0] = events['x'][indices]
        sampled_events[:, 1] = events['y'][indices]
        sampled_events[:, 2] = events['t'][indices]
        sampled_events[:, 3] = events['p'][indices]
        mask[:] = True
    
    return sampled_events, mask


def sample_and_mask(events, N=1024):
    """
    PyTorch wrapper for sample_and_mask_events.
    Returns tensors instead of numpy arrays.
    
    Args:
        events: Structured numpy array with fields ['x', 'y', 't', 'p']
        N: Number of events to sample (default 1024)
    
    Returns:
        events_tensor: (N, 4) float tensor
        mask_tensor: (N,) bool tensor
    """
    
    sampled_events, mask = sample_and_mask_events(events, N)
    events_tensor = torch.from_numpy(sampled_events).float()
    mask_tensor = torch.from_numpy(mask).bool()
    
    return events_tensor, mask_tensor


if __name__ == "__main__":
    # Testing with different scenarios
    from loadevents import event_extractor
    
    dataset_root = r"/media/adarsh/One Touch/EventSLAM/dataset/train"
    dataset = event_extractor(dataset_root)
    
    print("=" * 60)
    print("Testing sample_and_mask function")
    print("=" * 60)
    
    packets = dataset[0]
    print(f"\nOriginal packets: {len(packets)}")
    
    # Test with last packet (115 events - less than N)
    last_packet = packets[-1]
    num_events = len(last_packet['left_events_strip'])
    print(f"\n--- Test 1: Packet with {num_events} events (< N=1024) ---")
    
    events_tensor, mask_tensor = sample_and_mask(last_packet['left_events_strip'], N=1024)
    print(f"Output shape: {events_tensor.shape}")
    print(f"Mask shape: {mask_tensor.shape}")
    print(f"Valid events (mask=True): {mask_tensor.sum().item()}")
    print(f"Padded events (mask=False): {(~mask_tensor).sum().item()}")
    
    # Check that padding is correct
    print(f"\nFirst valid event: {events_tensor[0]}")
    print(f"First padded event (should be all -1): {events_tensor[num_events]}")
    
    # Test with first packet (should have 1024 events - equal to N)
    first_packet = packets[0]
    num_events_first = len(first_packet['left_events_strip'])
    print(f"\n--- Test 2: Packet with {num_events_first} events ---")
    
    events_tensor2, mask_tensor2 = sample_and_mask(first_packet['left_events_strip'], N=1024)
    print(f"Output shape: {events_tensor2.shape}")
    print(f"Valid events (mask=True): {mask_tensor2.sum().item()}")
    
    # Test with empty events
    print(f"\n--- Test 3: Empty events array ---")
    empty_events = np.array([], dtype=[('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('p', 'f4')])
    events_tensor3, mask_tensor3 = sample_and_mask(empty_events, N=1024)
    print(f"Output shape: {events_tensor3.shape}")
    print(f"Valid events (mask=True): {mask_tensor3.sum().item()}")
    print(f"All values are -1: {torch.all(events_tensor3 == -1).item()}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)