from multihopkg.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
from multihopkg.utils import ops
import torch

class ContinuousPolicy():

    def __init__(
        self,
        # Goodness this is ugly:
        action_dropout_anneal_factor: float,
        action_dropout_anneal_interval: float,
        action_dropout_rate: float,
        baseline: str,
        beam_size: int,
        beta: float,
        gamma: float,
        num_rollouts: int,
        num_rollout_steps: int,
        use_action_space_bucketing: bool,
    ):
        # Training hyperparameters
        self.use_action_space_bucketing = use_action_space_bucketing
        self.num_rollouts = num_rollouts
        self.num_rollout_steps = num_rollout_steps
        self.baseline = baseline
        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor
        self.action_dropout_rate = action_dropout_rate
        self.action_dropout_anneal_factor = (
            action_dropout_anneal_factor  # Used in parent
        )
        self.action_dropout_anneal_interval = (
            action_dropout_anneal_interval  # Also used by parent
        )

        # TODO: PRepare more stuff here. You erased a lot
        # Inference hyperparameters
        self.beam_size = beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, r, e2, pred_e2):
        # TODO: Soft reward here.
        raise  NotImplementedError
        # return (pred_e2 == e2).float()

    def loss(self, mini_batch):
        
        # TODO: CHeck if we want to do that
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
    

        ##################################
        # Here we roll a batch of epusodes
        ##################################
        e1, e2, r = format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)


        ##################################
        #Compute metrics from output
        ##################################
        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

def format_batch(batch_data, num_labels=-1, num_tiles=1):
    """
    Convert batched tuples to the tensors accepted by the NN.
    """
    # TODO: Understand why this is needed

    # This is the tiling happening again. 
    def convert_to_binary_multi_subject(e1):
        e1_label = zeros_var_cuda([len(e1), num_labels])
        for i in range(len(e1)):
            e1_label[i][e1[i]] = 1
        return e1_label

    def convert_to_binary_multi_object(e2):
        e2_label = zeros_var_cuda([len(e2), num_labels])
        for i in range(len(e2)):
            e2_label[i][e2[i]] = 1
        return e2_label

    batch_e1, batch_e2, batch_r = [], [], []
    for i in range(len(batch_data)):
        e1, e2, r = batch_data[i]
        batch_e1.append(e1)
        batch_e2.append(e2)
        batch_r.append(r)
    batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
    batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
    if type(batch_e2[0]) is list:
        batch_e2 = convert_to_binary_multi_object(batch_e2)
    elif type(batch_e1[0]) is list:
        batch_e1 = convert_to_binary_multi_subject(batch_e1)
    else:
        batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
    # Rollout multiple times for each example
    if num_tiles > 1:
        batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
        batch_r = ops.tile_along_beam(batch_r, num_tiles)
        batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
    return batch_e1, batch_e2, batch_r
