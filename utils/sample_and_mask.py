import numpy as np
import torch

def sample_and_mask_events(events, N = 1024):
    """To sample events when len(events)<N,
    we put all 4 values as -1 in this case and add a mask with the Boolean value 0 (False) for such events
    """

    num_events = len(events)

    sampled_events = np.full((N, 4), -1.0, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)

    if num_events < N:
        sampled_events[:num_events, 0] = events['x']
        sampled_events[:num_events, 1] = events['y']
        sampled_events[:num_events, 2] = events['t']
        sampled_events[:num_events, 3] = events['p']
        mask[:num_events] = True
    else:
        sampled_events[:, 0] = events['x']
        sampled_events[:, 1] = events['y']
        sampled_events[:, 2] = events['t']
        sampled_events[:, 3] = events['p']
        mask[:] = True

    return sampled_events, mask

def sample_and_mask(events, N=1024):
    
    """
    This function exists to return a pytorch tensor for the sampled and masked events with a single call
    """

    sampled_events, mask = sample_and_mask_events(events, N)
    events_tensor = torch.from_numpy(sampled_events).float()
    mask_tensor = torch.from_numpy(mask).bool()

    return events_tensor, mask_tensor

if __name__ == "__main__":

    # only for testing, dosen't get used in the main loop
    from loadevents import event_extractor
    dataset_root = r"//media/adarsh/One Touch/EventSLAM/dataset/train"
    dataset = event_extractor(dataset_root)

    packets = dataset[0]
    print(f"Original packets: {len(packets)}")
    
    last_packet = packets[-1]
    print(f"Number of events in last packet: {len(last_packet['left_events_strip'])}")

    events_tensor, mask_tensor = sample_and_mask(last_packet['left_events_strip'], N=1024)
    print(f"Number of events in last packet after sampling and masking: {len(events_tensor)}")

    a, b = 110, 120 # just because the last packet of first sequence, has 115 events
    subset = events_tensor[a:b]
    print(subset)