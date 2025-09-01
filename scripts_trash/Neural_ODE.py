import torch
import torch.nn as nn
from torchdiffeq import odeint
from loadeventdata import extract_events

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim = 64):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, t, z):
        batch_size = z.shape[0]
        t_expanded = t.expand(batch_size, 1)
        z_with_time = torch.cat([z, t_expanded], dim=1)
        dz_dt = self.net(z_with_time)
        return dz_dt
    
class ODEBlock(nn.Module):
    def __init__(self, odefunc, integration_time=[0.0, 1.0], method = 'dopri5'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(integration_time, dtype=torch.float32)
        self.method = method

    def forward(self, z0):
        self.integration_time = self.integration_time.to(z0.device)
        z_trajectory = odeint(
            self.odefunc,
            z0,
            self.integration_time,
            method=self.method,
            rtol=1e-5,
            atol=1e-6
        )
        z1 = z_trajectory[-1]
        return z1
    
class NeuralODEWarp(nn.Module):
    def __init__(self, hidden_dim=64, integration_time=[0.0, 1.0], method='dopri5'):
        super(NeuralODEWarp, self).__init__()
        self.odefunc = ODEFunc(hidden_dim=hidden_dim)
        self.odeblock = ODEBlock(
            self.odefunc,
            integration_time=integration_time,
            method=method
        )

    def forward(self, events):
        xyz = events[:, :3]
        polarity = events[:, 3:4]
        warped_xyz = self.odeblock(xyz)
        warped_events = torch.cat([warped_xyz, polarity], dim=1)
        return warped_events
    
def load_and_process_events(dataset_root, sequence_idx=0, max_events_per_camera=1000):
    dataset = extract_events(dataset_root, max_events=max_events_per_camera)
    if len(dataset) ==  0:
        raise ValueError("No sequences found in the datset")
    sample = dataset[sequence_idx]
    left_events = sample['left']
    right_events = sample['right']

    def structured_to_tensor(events):
        if len(events) == 0:
            return torch.zeros(0, 4, dtype=torch.float32)
        x = events['x']
        y = events['y']
        t = events['t']
        p = events['p']
        events_tensor = torch.stack([
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32)
        ], dim=1)
        
        return events_tensor
    
    left_tensor = structured_to_tensor(left_events)
    right_tensor = structured_to_tensor(right_events)
    all_events = torch.cat([left_tensor, right_tensor], dim=0)

    print(f"Loaded sequence: {sample['sequence_info']['scenario']}/{sample['sequence_info']['sequence']}")
    print(f"Left events: {len(left_events)}, Right events: {len(right_events)}")
    print(f"Total events: {len(all_events)}")
    print(f"Time Window: {sample['t_start']:.6f}s to {sample['t_end']:.6f}s")

    return all_events

if __name__ == "__main__":
    print("Neural ODE Event Warping Module")
    dataset_root = "/home/adarsh/Documents/SRM/dataset/train" #Update the path with actual location
    events = load_and_process_events(dataset_root, sequence_idx=0, max_events_per_camera=1000)
    neural_ode = NeuralODEWarp(hidden_dim=64, integratio_time=[0.0, 1.0])
    print(f"events loaded: {events.shape}")

    print("testing Neural ODE")
    with torch.no_grad():
        warped = neural_ode(events)
        print(f"events warped: {warped.shape}")
        print(f"warped events range - x': [{warped[:, 0].min():.3f}, {warped[:, 0].max():.3f}]")
        print(f"warped events range - y': [{warped[:, 1].min():.3f}, {warped[:, 1].max():.3f}]")
        print(f"warped events range - t': [{warped[:, 2].min():.3f}, {warped[:, 2].max():.3f}]")

    neural_ode.train()
    events.requires_grad_(True)
    warped_events = neural_ode(events)
    loss = warped_events.mean() #dummy loss, to be replaced with slam, stereo loss later
    loss.backward()
    print(f"gradient computed successfully. loss: {loss.item():.6f}")

    