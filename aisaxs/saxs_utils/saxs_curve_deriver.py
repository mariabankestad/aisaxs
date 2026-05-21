import torch
import numpy as np


class SaxsCurveDeriver(torch.nn.Module):
    """
    A PyTorch module to compute the SAXS curve given a batched input density `rho`
    based on precomputed weights derived from a SAXS structure.

    Attributes:
        W (torch.Tensor): Precomputed weight tensor for SAXS calculation.
        num_pixels (int): Number of pixels in the input density `rho`.
    """

    def __init__(self, I_struc, sigma_smooth = 0.7):
        """
        Initialize the SaxsCurveDeriver.

        Args:
            I_struc (dict): Dictionary containing SAXS structure parameters.
                - 'q' (torch.Tensor): Momentum transfer values.
                - 'delta_q' (float): Step size in q.
                - 'Q22' (torch.Tensor): Square of the Q-values.
                - 'number_of_pixels' (int): Number of pixels in the input data.
        """
        super().__init__()

        q = I_struc['q']  # Momentum transfer values
        delta_q = I_struc['delta_q']  # Step size in q
        Q_alt = torch.sqrt(I_struc['Q22'])  # Square root of Q22
        sigma_smooth =sigma_smooth  # Smoothness parameter
        self.num_pixels = I_struc['number_of_pixels']

        # Prepare smoothness tensor
        s = torch.full_like(q, sigma_smooth) * delta_q

        # Expand tensors for weight calculation
        Q_alt_expanded = Q_alt.unsqueeze(-1)  # Shape: [num_pixels, 1]
        q_expanded = q.view(1, 1, -1)  # Shape: [1, 1, len(q)]
        s_expanded = s.view(1, 1, -1)  # Shape: [1, 1, len(q)]

        # Compute weights
        W_alt = torch.exp(-0.5 / s_expanded**2 * (Q_alt_expanded - q_expanded)**2) / (Q_alt_expanded + 1e-12)
        W_alt[0, :] = 0  # Set the first element to 0 for stability
        W_alt /= W_alt.sum(dim=(0, 1), keepdim=True)  # Normalize weights

        # Register W as a buffer to ensure it moves with the model's device
        self.register_buffer('W', W_alt)

    def forward(self, rho):
        """
        Forward pass to compute the SAXS curve for batched inputs.

        Args:
            rho (torch.Tensor): Input density tensor with shape (batch_size, height, width).

        Returns:
            torch.Tensor: Computed SAXS curves I(q) for each batch. Shape: (batch_size, len(q)).
        """
        #batch_size = rho.shape[0]

        # Compute the power spectrum of the density
        U = torch.fft.rfft2(rho / self.num_pixels, norm=None).abs() ** 2  # Shape: [batch_size, height, width // 2 + 1]
        #U[:,0,0] = 0
        # Compute the SAXS curve using the precomputed weights
        I_alt = torch.sum(U.unsqueeze(-1) * self.W, dim=(1, 2))  # Sum over height and width, keep batch dimension
        return I_alt

def compute_I_scaling_2d(number_of_pixels, options=None,delta_q_equal_to_1 = False,number_of_q_values = 1000):
    # Initialize options with default values if not provided
    if options is None:
        options = {}

    # Handle default values for options
    options.setdefault('q_min', None)
    options.setdefault('q_max', None)
    options.setdefault('delta_x', 0.5)
    options.setdefault('delta_q_equal_to_1', delta_q_equal_to_1)
    options.setdefault('log10_sampling_q', False)
    options.setdefault('number_of_q_values', number_of_q_values)

    # Create output dictionary to hold scaling data
    I_scaling_2d = {}
    I_scaling_2d['number_of_pixels'] = number_of_pixels

    # Determine delta_q and delta_x
    if options['delta_q_equal_to_1']:
        I_scaling_2d['delta_q'] = 1
        I_scaling_2d['delta_x'] = 2 * np.pi / (I_scaling_2d['delta_q'] * number_of_pixels)
    else:
        I_scaling_2d['delta_x'] = options['delta_x']
        I_scaling_2d['delta_q'] = 2 * np.pi / (I_scaling_2d['delta_x'] * number_of_pixels)

    # Vector of wave vector magnitudes in one dimension, in 1/nm
    q_1D_voxel_scale = torch.arange(-number_of_pixels / 2, number_of_pixels / 2)
    q_1D = q_1D_voxel_scale * I_scaling_2d['delta_q']

    # Determine q_min and q_max if not specified
    if options['q_min'] is None:
        options['q_min'] = q_1D.abs()[q_1D.abs() > 0].min().item()
    if options['q_max'] is None:
        options['q_max'] = q_1D.max().item()

    # Define vector of chosen q values
    number_of_q_values = options['number_of_q_values']
    I_scaling_2d['q'] = torch.linspace(options['q_min'], options['q_max'], number_of_q_values)
    I_scaling_2d['q_voxel_scale'] = I_scaling_2d['q'] / I_scaling_2d['delta_q']

    # Fourier space grid arrays
    QX = q_1D.view(number_of_pixels, 1).repeat(1, number_of_pixels)
    QY = q_1D.view(1, number_of_pixels).repeat(number_of_pixels, 1)

    # Squared radius in Q space
    I_scaling_2d['Q2'] = QX ** 2 + QY ** 2
    q_x = torch.fft.fftfreq(number_of_pixels, d=1.0/number_of_pixels)* I_scaling_2d['delta_q']
    q_y = torch.fft.rfftfreq(number_of_pixels, d=1.0/number_of_pixels)* I_scaling_2d['delta_q']
    I_scaling_2d['Q22']= q_x.view(-1,1)**2 + q_y.view(1,-1)**2

    # Additional values for plotting
    I_scaling_2d['q_range_full'] = [0, torch.sqrt(I_scaling_2d['Q2'].max()).item()]
    I_scaling_2d['q_range_chosen'] = [options['q_min'], options['q_max']]

    return I_scaling_2d



def compute_saxs_curve_2d(RHO, I_scaling_2d, options=None, plot= True,sigma_smooth =0.5):
    # Set default options if not provided
    if options is None:
        options = {}
    options.setdefault('sigma_smooth', sigma_smooth)
    options.setdefault('plot', plot)
    options.setdefault('RHO_type', 'image')
    options.setdefault('constant_smoothing', True)

    device = RHO.device
    # Extract variables
    q = I_scaling_2d['q'].to(device)
    delta_q = I_scaling_2d['delta_q']
    Q_alt = torch.sqrt(I_scaling_2d['Q22']).to(device)
    q_voxel_scale = I_scaling_2d.get('q_voxel_scale', None).to(device)

    # Fourier transform or use GAMMA based on RHO_type
    U_alt = torch.fft.rfft2(RHO/I_scaling_2d['number_of_pixels']).abs()**2
    # Smoothing weights
    if options['constant_smoothing']:
        s = torch.tensor([options['sigma_smooth']] * len(q)).to(device)
    else:
        s = options['sigma_smooth'] * (1 + torch.log(q_voxel_scale)).to(device)
    s = s * delta_q

    # Ensure U is in high precision for accuracy
    U_alt = U_alt.to(torch.float64)

    # Now, compute the vectorized version with adjustments
    Q_alt_expanded = Q_alt.unsqueeze(-1).to(torch.float64)  # Ensure high precision

    q_expanded = q.view(1, 1, -1).to(torch.float64)
    s_expanded = s.view(1, 1, -1).to(torch.float64)

    # Compute weights W for all values of q at once

    W_alt = torch.exp(-0.5 / s_expanded**2 * (Q_alt_expanded - q_expanded)**2) / (Q_alt_expanded + 1e-12)
    W_alt[0, :] = 0  # Set weight of q = (0,0) to zero

    # Normalize weights for each q value independently
    W_alt /= W_alt.sum(dim=(0, 1), keepdim=True)

    # Compute SAXS response for all q values at once
    I_alt = torch.sum(U_alt.unsqueeze(-1) * W_alt, dim=(0, 1))

    return  I_alt



def compute_saxs_curve_2d_batch(RHOs, I_scaling_2d, options=None, plot=True, sigma_smooth=0.5):
    # Set default options if not provided
    if options is None:
        options = {}
    options.setdefault('sigma_smooth', sigma_smooth)
    options.setdefault('plot', plot)
    options.setdefault('RHO_type', 'image')
    options.setdefault('constant_smoothing', True)

    # Extract variables
    q = I_scaling_2d['q']
    delta_q = I_scaling_2d['delta_q']
    Q_alt = torch.sqrt(I_scaling_2d['Q22'])
    q_voxel_scale = I_scaling_2d.get('q_voxel_scale', None)

    # Fourier transform or use GAMMA based on RHO_type
    U_alt = torch.fft.rfft2(RHOs/I_scaling_2d['number_of_pixels']).abs()**2  # Shape: (b, H, W//2 + 1)


    # Ensure U is in high precision for accuracy
    U_alt = U_alt.to(torch.float64)

    # Smoothing weights
    if options['constant_smoothing']:
        s = torch.tensor([options['sigma_smooth']] * len(q), dtype=torch.float64)
    else:
        s = options['sigma_smooth'] * (1 + torch.log(q_voxel_scale))
    s = s * delta_q

    # Expand Q, q, and s for broadcasting
    Q_alt_expanded = Q_alt.unsqueeze(-1).to(torch.float64)  # Shape: (H, W, 1)
    q_expanded = q.view(1, 1, -1).to(torch.float64)  # Shape: (1, 1, len(q))
    s_expanded = s.view(1, 1, -1).to(torch.float64)  # Shape: (1, 1, len(q))

    # Compute weights W for all values of q at once
    W_alt = torch.exp(-0.5 / s_expanded**2 * (Q_alt_expanded - q_expanded)**2) / (Q_alt_expanded + 1e-12)
    W_alt[0, :] = 0  # Set weight of q = (0,0) to zero

    # Normalize weights for each q value independently
    W_alt /= W_alt.sum(dim=(0, 1), keepdim=True)

    # Compute SAXS response for all q values at once, across the batch dimension
    I_alt_batch = torch.sum(U_alt.unsqueeze(-1) * W_alt, dim=(1, 2))  # Shape: (b, len(q))

    return I_alt_batch  # Shape: (b, len(q))
