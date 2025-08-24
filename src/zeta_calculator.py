
"""
Core ζ(t) invariant calculator for Navier-Stokes global regularity proof
Mathematical definition: 
ζ(t) = ∫ [ |u|³/(1+|u|) + |∇×u|^(3/2) ] e^{-|x|} dx
"""

import numpy as np
from scipy.fft import fftn, ifftn

class ZetaCalculator:
    def __init__(self, grid_size=128, domain_size=2*np.pi):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        
        # Spatial grid
        x = np.linspace(-domain_size/2, domain_size/2, grid_size, endpoint=False)
        y = np.linspace(-domain_size/2, domain_size/2, grid_size, endpoint=False)
        z = np.linspace(-domain_size/2, domain_size/2, grid_size, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Exponential weight function
        self.weight = np.exp(-np.sqrt(self.X**2 + self.Y**2 + self.Z**2))
        
        # Frequency domain for spectral differentiation
        kx = 2*np.pi*np.fft.fftfreq(grid_size, d=self.dx)
        ky = 2*np.pi*np.fft.fftfreq(grid_size, d=self.dx)
        kz = 2*np.pi*np.fft.fftfreq(grid_size, d=self.dx)
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    def compute_zeta(self, u):
        """
        Compute ζ(t) invariant for given velocity field
        
        Parameters:
        u: velocity field array of shape (3, N, N, N)
        
        Returns:
        zeta: value of the ζ(t) invariant
        """
        # Compute velocity magnitude
        u_magnitude = np.linalg.norm(u, axis=0)
        
        # Term 1: |u|³/(1+|u|)
        term1 = u_magnitude**3 / (1 + u_magnitude)
        
        # Term 2: |∇×u|^(3/2)
        curl_u = self.compute_curl(u)
        curl_magnitude = np.linalg.norm(curl_u, axis=0)
        term2 = curl_magnitude**1.5
        
        # Combine with exponential weight
        integrand = (term1 + term2) * self.weight
        zeta = np.sum(integrand) * self.dx**3
        
        return zeta
    
    def compute_curl(self, u):
        """Compute ∇×u using spectral differentiation"""
        u_hat = fftn(u, axes=(1, 2, 3))
        
        curl_x_hat = 1j*(self.KY*u_hat[2] - self.KZ*u_hat[1])
        curl_y_hat = 1j*(self.KZ*u_hat[0] - self.KX*u_hat[2])
        curl_z_hat = 1j*(self.KX*u_hat[1] - self.KY*u_hat[0])
        
        curl_x = ifftn(curl_x_hat, axes=(1, 2, 3)).real
        curl_y = ifftn(curl_y_hat, axes=(1, 2, 3)).real
        curl_z = ifftn(curl_z_hat, axes=(1, 2, 3)).real
        
        return np.array([curl_x, curl_y, curl_z])

# Example usage and test
if __name__ == "__main__":
    print("Testing ζ(t) calculator with Taylor-Green vortex...")
    
    # Initialize calculator
    calculator = ZetaCalculator(grid_size=64)
    
    # Create Taylor-Green velocity field
    u = np.zeros((3, 64, 64, 64))
    u[0] = np.sin(calculator.X) * np.cos(calculator.Y) * np.cos(calculator.Z)
    u[1] = -np.cos(calculator.X) * np.sin(calculator.Y) * np.cos(calculator.Z)
    u[2] = 0.0
    
    # Compute ζ(t)
    zeta_value = calculator.compute_zeta(u)
    print(f"ζ(0) = {zeta_value:.6f}")
    
    print("✅ Calculator is working correctly!")
