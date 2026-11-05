"""Weighted rectified-flow loss."""


def weighted_flow_loss(v, noise, x0, m_lat, outside_weight=0.05):
    """v: predicted velocity, noise and x0: endpoints of x_t = (1 - t) x0 + t noise, m_lat: (B,1,64,64) latent mask."""
    w = m_lat + (1 - m_lat) * outside_weight
    err = (v - (noise.float() - x0.float())).square().mean(1, keepdim=True)
    return (err * w).sum() / w.sum()
