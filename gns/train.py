import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from datetime import datetime

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
from statistics import mean

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

Stats = collections.namedtuple('Stats', ['mean', 'std'])
tbx = None

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device):
  """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
  """
  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in range(nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
    next_position = torch.where(
        kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy(),
  }

  return output_dict, loss


def predict(device: str, flags, model_fn=None, step=None, eval_count=0):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  metadata = reading_utils.read_metadata(flags["data_path"])
  simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)

  # Load simulator
  model_fn = flags["model_path"] + flags["model_file"] if model_fn is None else model_fn
  if os.path.exists(model_fn):
    simulator.load(model_fn)
  else:
    train(simulator)
  
  simulator.to(device)
  simulator.eval()

  # Output path
  if not os.path.exists(flags["output_path"]):
    os.makedirs(flags["output_path"])

  # Use `valid`` set for eval mode if not use `test`
  split = 'test' if "mode" in flags and flags["mode"] == 'rollout' else 'valid'
  data_path = flags["data_path"]
  ds = data_loader.get_data_loader_by_trajectories(path=f"{data_path}{split}.npz")

  eval_loss = []
  with torch.no_grad():
    for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
      if (eval_count != 0 and len(eval_loss) > eval_count):
        break
      # positions, particle_type = add_static_objects(positions, particle_type)
      # n_particles_per_example = len(particle_type)

      positions.to(device)
      particle_type.to(device)
      n_particles_per_example = torch.tensor([int(n_particles_per_example)], dtype=torch.int32).to(device)

      nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH
      # Predict example rollout
      example_rollout, loss = rollout(simulator, positions.to(device), particle_type.to(device),
                                      n_particles_per_example.to(device), nsteps, device)

      example_rollout['metadata'] = metadata
      print("Predicting example {} loss: {}".format(example_i, loss.mean()))
      eval_loss.append(torch.flatten(loss))
      
      # Save rollout in testing
      if "mode" in flags and flags["mode"] == 'rollout':
        example_rollout['metadata'] = metadata
        filename = f'rollout_{example_i}.pkl'
        filename = os.path.join(flags["output_path"], filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))

  if step is not None:
    get_tbx(flags).add_scalar('loss/3_eval', torch.mean(torch.cat(eval_loss)), step)

def add_static_objects(positions, particle_type):

    # p0 = (0.5, 0.2)
    # p1 = (0.8, 0.1)
    p0 = (0.3,0.2)
    p1 = (0.5,0.25)
    F = 40

    new_items = []
    for i in range(F):
      x = p0[0] + (p1[0] - p0[0])/F * i
      y = p0[1] + (p1[1] - p0[1])/F * i
      new_items.append((x,y))
    n = torch.tensor(new_items).reshape(F, 1, 2).repeat(1, positions.shape[1], 1)
    positions = torch.cat([positions, n], 0)
    particle_type = torch.cat([particle_type, torch.ones(F, dtype=torch.int32) * KINEMATIC_PARTICLE_ID], 0)

    return positions, particle_type
    # particle_type[:F] = KINEMATIC_PARTICLE_ID
    # positions[:F, :, :] = positions[:F, :1, :].repeat(1,positions.shape[1],1)



def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def train(rank, flags, world_size):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
  """
  distribute.setup(rank, world_size)

  metadata = reading_utils.read_metadata(flags["data_path"])
  serial_simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], rank)

  simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
  optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  step = 0

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      simulator.module.load(flags["model_path"] + flags["model_file"])

      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      # set optimizer state
      optimizer = torch.optim.Adam(simulator.module.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, rank)
      # set global train state
      step = train_state["global_train_state"].pop("step")
 
    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg) 

  simulator.train()
  simulator.to(rank)

  dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                             input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                             batch_size=flags["batch_size"],
                                                            )

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
  not_reached_nsteps = True
  tbx_graph_added = False
  try:
    while not_reached_nsteps:
      torch.distributed.barrier()
      for ((position, particle_type, n_particles_per_example), labels) in dl:
        position.to(rank)
        particle_type.to(rank)
        n_particles_per_example.to(rank)
        labels.to(rank)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(rank)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(rank)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        if not tbx_graph_added:
          # add_architecture_graph(flags, simulator)
          tbx_graph_added = True

        # if 'balls' in metadata:
        #   sampled_noise = torch.zeros_like(sampled_noise)

        # Get the predictions and target accelerations.
        position = position.to(rank)
        if data_loader.prediction_skip == 0:
          pred_acc, target_acc, delta_dist_pct, pred_next_position = simulator.module.predict_accelerations(
              next_positions=labels.to(rank),
              position_sequence_noise=sampled_noise.to(rank),
              position_sequence=position,
              nparticles_per_example=n_particles_per_example.to(rank),
              particle_types=particle_type.to(rank))

        elif data_loader.prediction_skip == 1:
          _, _, _, pred_next_position = simulator.module.predict_accelerations(
              None,
              position_sequence_noise=sampled_noise.to(rank),
              position_sequence=position,
              nparticles_per_example=n_particles_per_example.to(rank),
              particle_types=particle_type.to(rank))

          # append the pred_next_position to the position sequence, and remove the top of the sequence.
          position = torch.cat([position[:, 1:, :], pred_next_position.unsqueeze(1)], dim=1)

          pred_acc, target_acc, delta_dist_pct, pred_next_position = simulator.module.predict_accelerations(
              next_positions=labels.to(rank),
              position_sequence_noise=sampled_noise.to(rank),
              position_sequence=position,
              nparticles_per_example=n_particles_per_example.to(rank),
              particle_types=particle_type.to(rank))
        else:
          assert False, "prediction_skip must be 0 or 1"    

        num_non_kinematic = non_kinematic_mask.sum()
        if True or not 'balls' in metadata:
          # Calculate the loss and mask out loss on kinematic particles
          loss = (pred_acc - target_acc) ** 2
          loss = loss.sum(dim=-1)
          loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
          loss = loss.sum() 

          # Add the loss on the distance between nodes to enforce the mesh integrity: 
          # penalize nodes that with world_distance greater than the mesh distance
          # delta_dist_pct = (mesh_distance-world_distance) / mesh_distance
          tension_margin = 0.1 # hyper-parameter depends on the tension of the mesh
          weight = 0.1 

          delta_dist_pct = torch.clamp(delta_dist_pct + tension_margin, max=0)
          loss_2 = (delta_dist_pct ** 2).mean() * weight

          loss = loss / num_non_kinematic + loss_2
        else:
          pred_next_position = simulator.module._decoder_postprocessor(pred_acc, position.to(rank))
          loss = (pred_next_position - labels.to(rank)) ** 2
          loss = loss.sum(dim=-1)
          loss = loss.sum() / num_non_kinematic


        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if rank == 0 and step % 100 == 0:
          print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.')

        if rank == 0 and step % 10 == 0 and step > 100:
          get_tbx(flags).add_scalar('loss/1_train', loss, step)
          # get_tbx(flags).add_scalar('in_ball/1_train', torch.mean(-dist2ball), step)
          
          if step % 500 == 0:
            compute_valid_loss(flags, simulator, step)
          
        # Save model state
        if step % flags["nsave_steps"] == 0 and rank == 0:
          model_fn = flags["model_path"] + 'model-'+str(step)+'.pt'
          simulator.module.save(model_fn)
          train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
          torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')
          if step >= 100:
            predict('cuda', flags, model_fn=model_fn, step=step, eval_count = 20)

        # Complete training
        if (step >= flags["ntraining_steps"]):
          not_reached_nsteps = False
          break

        step += 1

  except KeyboardInterrupt:
    pass

  if rank == 0:
    simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step":step})
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

  distribute.cleanup()

def add_architecture_graph(flags, simulator):
    # input_to_model = (labels.to(rank), sampled_noise.to(rank), position.to(rank), n_particles_per_example.to(rank), particle_type.to(rank))
    # get_tbx(flags).add_graph(simulator.module, input_to_model=input_to_model)

    # # -------- Add graph for encoder -------- #
    # input_to_model = (torch.zeros([1,30], device='cuda'), 
    #                         torch.zeros([1, 3], device='cuda'))
    # get_tbx(flags).add_graph(
    #         simulator.module._encode_process_decode._encoder,
    #         input_to_model=input_to_model)

    # # -------- Add graph for processor -------- #
    input_to_model = (torch.zeros([1,128], device='cuda'), 
                            torch.zeros([2, 1], dtype=torch.long, device='cuda'), 
                            torch.zeros([1, 128], device='cuda'))
    get_tbx(flags).add_graph(
            simulator.module._encode_process_decode._processor,
            input_to_model=input_to_model)

    # -------- Add graph for decoder -------- #
    # input_to_model = (torch.zeros([1,128], device='cuda'))
    # get_tbx(flags).add_graph(
    #         simulator.module._encode_process_decode._decoder,
    #         input_to_model=input_to_model)


def compute_valid_loss(flags, simulator, step):
  rank = 0
  dl = data_loader.get_data_loader_by_samples(f'{flags["data_path"]}valid.npz',
                                          input_length_sequence=INPUT_SEQUENCE_LENGTH,  
                                          batch_size=flags["batch_size"],
                                          shuffle=True)

  with torch.no_grad():
    loss_lst = []
    for ((position, particle_type, n_particles_per_example), labels) in dl:    
      position.to(rank)
      particle_type.to(rank)
      n_particles_per_example.to(rank)
      labels.to(rank)

      # TODO (jpv): Move noise addition to data_loader
      # Sample the noise to add to the inputs to the model during training.
      sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(rank)
      non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(rank)
      sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
      
      pred_acc, target_acc, _, dist2ball = simulator.module.predict_accelerations(
          next_positions=labels.to(rank),
          position_sequence_noise=sampled_noise.to(rank),
          position_sequence=position.to(rank),
          nparticles_per_example=n_particles_per_example.to(rank),
          particle_types=particle_type.to(rank))

      # Calculate the loss and mask out loss on kinematic particles
      loss = (pred_acc - target_acc) ** 2
      loss = loss.sum(dim=-1)
      num_non_kinematic = non_kinematic_mask.sum()
      loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
      loss = loss.sum() / num_non_kinematic
      loss_lst.append(loss.item())
      if len(loss_lst) > 50:
        break

    get_tbx(flags).add_scalar('loss/2_valid', mean(loss_lst), step)
    # get_tbx(flags).add_scalar('in_ball/2_valid', torch.mean(-dist2ball), step)


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: str) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  # Normalization stats
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
  }

  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=37 if metadata['dim'] == 3 else 30,
      nedge_in=metadata['dim'] + 4, # already have world displacement, we add mesh distance, world distance, and mesh displacement
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      device=device)

  if 'balls' in metadata:
    simulator.set_balls(metadata['balls'], metadata['neighbour_search_size'], metadata['quad_size'])

  return simulator

def get_tbx(flags):
  global tbx
  if tbx is None:
    # save tensorboard log to model_path/tensorboard
    tbx_path = "/mnt/d/Stanford/tbx_logs/" + datetime.now().strftime('%Y%m%d-%H%M%S')
    tbx = SummaryWriter(tbx_path)  
  return tbx

def main(_):
  """Train or evaluates the model.

  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29600"#
  FLAGS = flags.FLAGS
  myflags = {}
  myflags["data_path"] = FLAGS.data_path
  myflags["noise_std"] = FLAGS.noise_std
  myflags["lr_init"] = FLAGS.lr_init
  myflags["lr_decay"] = FLAGS.lr_decay
  myflags["lr_decay_steps"] = FLAGS.lr_decay_steps
  myflags["batch_size"] = FLAGS.batch_size
  myflags["ntraining_steps"] = FLAGS.ntraining_steps
  myflags["nsave_steps"] = FLAGS.nsave_steps
  myflags["model_file"] = FLAGS.model_file
  myflags["model_path"] = FLAGS.model_path
  myflags["train_state_file"] = FLAGS.train_state_file
  myflags["output_path"] = FLAGS.output_path
  myflags["mode"] = FLAGS.mode

  # Read metadata
  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    world_size = torch.cuda.device_count()
    print(f"world_size = {world_size}")
    distribute.spawn_train(train, myflags, world_size)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    predict(device, myflags)

if __name__ == '__main__':
  app.run(main)
