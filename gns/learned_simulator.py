import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict

mesh_width = 32# torch.sqrt(nparticles_per_example).mean().type(torch.int)

class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: Dict,
          nparticle_types: int,
          particle_type_embedding_size,
          device="cpu"):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._balls = None
    self._cloth_edge_index = None
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device
    self.mesh_size = mesh_width * mesh_width

  def set_balls(self, balls, neighbour_search_size, quad_size):
    self._balls = balls
    self._neighbour_search_size = neighbour_search_size
    self._quad_size = quad_size

  def get_cloth_edge_index_v0(self, batch_ids):
    if self._cloth_edge_index is None:
      bc = torch.bincount(batch_ids)
      assert torch.all(bc)
      DIM = int(bc[0].sqrt())

      np_cloth = np.arange(DIM*DIM*2).reshape(DIM*DIM, 2)
      for i in range(DIM*DIM):
          np_cloth[i] = [i//DIM, i%DIM]
      np_cloth = np.tile(np_cloth, (len(bc), 1))
      xy = torch.FloatTensor(np_cloth).to(self._device)
      self._cloth_edge_index = radius_graph(xy, self._neighbour_search_size, 
                batch=batch_ids, loop=False) #Ming TODO: try loop=True

      index_bc = torch.bincount(self._cloth_edge_index[0])
      mean_neigbors = index_bc.double().mean()
      assert mean_neigbors > 1 and mean_neigbors <= 100, 'shall within  rings'

      # generate the cloth edge distance, no need to normalize as each tile is a square of 1
      self._mesh_edge_distance = torch.norm(xy[self._cloth_edge_index[0]] - xy[self._cloth_edge_index[1]], dim=1).reshape(-1, 1)
      
    return self._cloth_edge_index


  def get_cloth_edge_index(self, batch_ids):
    if self._cloth_edge_index is None:
      bc = torch.bincount(batch_ids)
      # assert torch.all(bc)
      # assert mesh_width == int(bc[0].sqrt())

      np_cloth = np.arange(mesh_width*mesh_width*2).reshape(mesh_width*mesh_width, 2)
      for i in range(mesh_width*mesh_width):
          np_cloth[i] = [i//mesh_width, i%mesh_width]
      xy = torch.FloatTensor(np_cloth).to(self._device)
      base_mesh_graph = radius_graph(xy, self._neighbour_search_size, batch=None, loop=False) 
      self._cloth_edge_index = base_mesh_graph
      for i in range(1, len(bc)):
        next = base_mesh_graph + bc[0]*i
        self._cloth_edge_index = torch.cat([self._cloth_edge_index, next], axis=1)

      # np_cloth = np.tile(np_cloth, (len(bc), 1))
      # self.cloth_edge_index = radius_graph(xy, self._neighbour_search_size, 
      #           batch=batch_ids, loop=False) #Ming TODO: try loop=True

      index_bc = torch.bincount(self._cloth_edge_index[0])
      mean_neigbors = index_bc.double().mean()
      assert mean_neigbors > 1 and mean_neigbors <= 100, 'shall within  rings'

      # generate the cloth edge distance, no need to normalize as each tile is a square of 1
      # self._mesh_edge_distance = torch.norm(xy[self.cloth_edge_index[0]] - xy[self.cloth_edge_index[1]], dim=1).reshape(-1, 1)
      
    return self._cloth_edge_index

  
  def forward(self, 
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Forward hook runs on class instantiation"""
    res = self.predict_accelerations(
              next_positions,
              position_sequence_noise,
              position_sequence,
              nparticles_per_example,
              particle_types)
    return res
    pass

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = False):
    """Generate graph edges to all particles within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    self.batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    if True or self._balls is None:
      edge_index = radius_graph(
          node_features, r=radius, batch=self.batch_ids, loop=add_self_edges)
      self.get_cloth_edge_index(self.batch_ids)
    else:
      # cloth drop training scenario, the graph edges are precomputed
      edge_index = self.get_cloth_edge_index(self.batch_ids)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def get_mesh_distance(self, senders, receivers):
    mesh_dist = torch.norm(self.get_mesh_displacement(senders, receivers), dim=1)
    return mesh_dist.reshape(-1, 1)

  def get_mesh_displacement(self, senders, receivers):
    bc = self.batch_ids.bincount()[0]
    s = (senders % bc)
    r = (receivers % bc)

    # only return mesh distance when both sender and receiver are mesh particles
    d_x = torch.where( torch.logical_and(s < self.mesh_size, r < self.mesh_size), s % mesh_width - r % mesh_width, 100)
    d_y = torch.where( torch.logical_and(s < self.mesh_size, r < self.mesh_size), s // mesh_width - r // mesh_width, 100)

    self._normalized_relative_displacements = torch.cat([d_x.reshape(-1,1), d_y.reshape(-1,1)], axis=1) \
                                              * self._quad_size /self._connectivity_radius
    # self._normalized_relative_displacements = self._normalized_relative_displacements.reshape(-1, 2)
    return self._normalized_relative_displacements
  
  def get_mesh_distance_v0(self, senders, receivers):
    s_x = senders % mesh_width
    s_y = senders // mesh_width
    r_x = receivers % mesh_width
    r_y = receivers // mesh_width
    mesh_dist = ((s_x - r_x).square() + (s_y - r_y).square()).sqrt() * self._quad_size /self._connectivity_radius
    return mesh_dist.reshape(-1, 1)

  def get_mesh_displacement_v0(self, senders, receivers):
    s_x = senders % mesh_width
    s_y = senders // mesh_width
    r_x = receivers % mesh_width
    r_y = receivers // mesh_width

    normalized_relative_displacements = torch.cat([(s_x - r_x).reshape(-1,1),  (s_y - r_y).reshape(-1,1)], axis=1) \
                                        * self._quad_size /self._connectivity_radius
    return normalized_relative_displacements

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(
        nparticles, -1)
    # There are 5 previous steps, with dim 2
    # node_features shape (nparticles, 5 * 2 = 10)
    node_features.append(flat_velocity_sequence)

    if self._balls is None:
      # Normalized clipped distances to lower and upper boundaries.
      # boundaries are an array of shape [num_dimensions, 2], where the second
      # axis, provides the lower/upper boundaries.
      obs_features = self.get_boundary_feature(most_recent_position)
    else:
      # cloth drop training scenario, the boundary features are precomputed
      obs_features = self.get_ball_feature(most_recent_position)
    # The distance to 4 boundaries (top/bottom/left/right)
    # node_features shape (nparticles, 10+4)
    node_features.append(obs_features)

    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 30) for 2D
    # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    mesh_disp = self.get_mesh_displacement(senders, receivers)
    edge_features.append(mesh_disp)

    mesh_dist = self.get_mesh_distance(senders, receivers)
    edge_features.append(mesh_dist)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def get_boundary_feature(self, most_recent_position):
    boundaries = torch.tensor(
      self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
      most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
      boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
      [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
      distance_to_boundaries / self._connectivity_radius, -1., 1.)
        
    return normalized_clipped_distance_to_boundaries

  def get_ball_feature(self, most_recent_position):
    '''
    calculated normalized distance to the ball:
    1. calculate the distance to the ball center
    2. normalize the distance to the ball radius
    3. clip the distance to [-1, 1]
    '''
    ball_centers = torch.tensor(self._balls[0][:3], requires_grad=False).float().to(self._device)
    ball_radius = torch.tensor(self._balls[0][3], requires_grad=False).float().to(self._device)

    displacement_to_ball_center = (most_recent_position - ball_centers)
    normalized_displacement_to_ball_center = displacement_to_ball_center/self._connectivity_radius

    distance_to_ball_surface = torch.norm(displacement_to_ball_center, dim=-1, keepdim=True) - ball_radius
    normalized_distance_to_ball_surface = (distance_to_ball_surface/self._connectivity_radius).reshape(-1, 1)
    normalized_clipped_distance_to_ball_surface = torch.clamp(normalized_distance_to_ball_surface, -1., 1.)

    # normalized_clipped_displacement_to_ball = torch.clamp(displacement/self._connectivity_radius, -1., 1.)
    # normalized_displacement_to_ball = ((most_recent_position - ball_centers) * ball_radius / distance + ball_centers)/self._connectivity_radius

    r = torch.cat([normalized_clipped_distance_to_ball_surface.repeat(1,3), normalized_displacement_to_ball_center], dim=1)

    return r


  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types)
    predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

      also return the edge distances calculated from the predicted positions

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types)
    predicted_normalized_acceleration = self._encode_process_decode(
        node_features, edge_index, edge_features)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    if next_positions is not None:
      next_position_adjusted = next_positions + position_sequence_noise[:, -1]
      target_normalized_acceleration = self._inverse_decoder_postprocessor(
          next_position_adjusted, noisy_position_sequence)
    else:
      target_normalized_acceleration = None
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    pred_next_position = self._decoder_postprocessor(predicted_normalized_acceleration, noisy_position_sequence)
    mesh_graph = self.get_cloth_edge_index(self.batch_ids)
    world_distances = torch.norm(pred_next_position[mesh_graph[0]] - pred_next_position[mesh_graph[1]], dim=1) / self._connectivity_radius
    mesh_distances = self.get_mesh_distance(mesh_graph[0], mesh_graph[1]).reshape(-1)

    delta_dist_pct = (mesh_distances-world_distances)/(mesh_distances)# + 1e-6)

    return predicted_normalized_acceleration, target_normalized_acceleration, delta_dist_pct, pred_next_position

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
