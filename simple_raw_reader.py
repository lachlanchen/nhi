#!/usr/bin/env python3
"""
Simple script to read Metavision .raw file into numpy arrays
"""

import sys
sys.path.append('/usr/lib/python3/dist-packages')


import numpy as np
from metavision_core.event_io import EventsIterator


def read_raw_simple(raw_file_path):
    """
    Simple function to read all events from a .raw file
    
    Returns:
        tuple: (x, y, t, p, width, height) where each is a numpy array
    """
    
    # Create iterator
    mv_iterator = EventsIterator(input_path=raw_file_path, delta_t=1000000)
    height, width = mv_iterator.get_size()
    
    # Collect all events
    all_events = []
    
    print("Reading events...")
    for events in mv_iterator:
        if events.size > 0:
            all_events.append(events)
    
    if not all_events:
        print("No events found!")
        return None, None, None, None, width, height
    
    # Concatenate all events
    all_events = np.concatenate(all_events)
    
    # Extract coordinates
    x = all_events['x']
    y = all_events['y']
    t = all_events['t']
    p = all_events['p']
    
    print(f"Loaded {len(x):,} events")
    print(f"Time range: {t[0]} - {t[-1]} μs ({(t[-1]-t[0])/1e6:.2f} seconds)")
    print(f"Sensor size: {width} x {height}")
    
    return x, y, t, p, width, height


# Example usage
if __name__ == "__main__":
    # Read your specific file
    raw_file = "sync_imaging/plastics_3/sync_recording_event_20250707_113525.raw"
    
    x, y, t, p, width, height = read_raw_simple(raw_file)
    
    if x is not None:
        print(f"\nEvent data shapes:")
        print(f"x: {x.shape}, range: {x.min()}-{x.max()}")
        print(f"y: {y.shape}, range: {y.min()}-{y.max()}")  
        print(f"t: {t.shape}, range: {t.min()}-{t.max()}")
        print(f"p: {p.shape}, unique values: {np.unique(p)}")
        
        # Example: Count events per polarity
        pos_events = np.sum(p == 1)
        neg_events = np.sum(p == 0)
        print(f"\nPolarity distribution:")
        print(f"Positive events (ON): {pos_events:,} ({pos_events/len(p)*100:.1f}%)")
        print(f"Negative events (OFF): {neg_events:,} ({neg_events/len(p)*100:.1f}%)")
        
        # Example: Get events in first 100ms
        first_100ms = t[0] + 100000  # 100ms in microseconds
        mask = t < first_100ms
        print(f"\nEvents in first 100ms: {np.sum(mask):,}")