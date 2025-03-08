import sys
import copy
import torch as th
import torch.nn as nn
import time
import itertools
import logging
from torch import Tensor
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, OrderedDict, Tuple, Union
from models.prefitter import Prefitter
from models.overfitter import OverfitterParameter, OverFitter
from encoding_management.presets import AVAILABLE_PRESETS, Preset, TrainerPhase
from utils.misc import ARMINT, FIXED_POINT_FRACTIONAL_MULT, TrainingExitCode, POSSIBLE_DEVICE, DescriptorOverfitter, DescriptorNN
from utils.helpers import pad_image
from utils.timer import Timer


@dataclass
class EncoderManager():
    """
    All the encoding option for a frame (loss, learning rate) as well as some
    counters monitoring the training time, the number of training iterations or the number
    of loops already done.
    """
    # ----- Encoding (i.e. training) options
    preset_name: str                                            # Preset name, should be a key in AVAILABLE_PRESETS src/encoding_management/presets.py
    start_lr: float = 1e-2                                      # Initial learning rate
    n_loops: int = 1                                            # Number of training loop
    n_itr: int = int(1e5)                                       # Maximum number of training iterations for a **single** phase
    
    # ==================== Not set by the init function ===================== #
    # ----- Actual preset, instantiated from its name
    preset: Preset = field(init=False)                          # It contains the learning rate in the different phase

    # ----- Monitoring
    idx_best_loop: int = field(default=0, init=False)           # Index of the loop which gives the best results (i.e. the best_loss)
    best_loss: float = field(default=1e6, init=False)           # Overall best loss (for all loops)
    loop_counter: int = field(default=0, init=False)            # Number of loops already done
    loop_start_time: float = field(default=0., init=False)      # Loop start time (before warm-up) ? What's this?
    iterations_counter: int = field(default=0, init=False)      # Total number of iterations done, including warm-up
    total_training_time_sec: float = field(default=0.0, init=False) # Total training time (second), including warm-up
    phase_idx: int = field(default=0, init=False)               # Index of the current training phase for the current loop
    warm_up_done: bool = field(default=False, init=False)       # True if the warm-up has already been done for this loop
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        assert self.preset_name in AVAILABLE_PRESETS, f'Preset named {self.preset_name} does not exist.' \
            f' List of available preset:\n{list(AVAILABLE_PRESETS.keys())}.'

        self.preset = AVAILABLE_PRESETS.get(self.preset_name)(start_lr= self.start_lr, n_itr_per_phase=self.n_itr)

        flag_quantize_model = False
        for training_phase in self.preset.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True
        assert flag_quantize_model, f'The selected preset ({self.preset_name}) does not include ' \
            f' a training phase with neural network quantization.\n{self.preset.pretty_string()}'

    def record_beaten(self, candidate_loss: float) -> bool:
        """Return True if the candidate loss is better (i.e. lower) than the best loss.

        Args:
            candidate_loss (float): Current candidate loss.

        Returns:
            bool: True if the candidate loss is better than the best loss
                (i.e. candidate < best).
        """
        return candidate_loss < self.best_loss

    def set_best_loss(self, new_best_loss: float):
        """Set the new best loss attribute. It automatically looks at the current loop_counter
        to fill the idx_best_loop attribute.

        Args:
            new_best_loss (float): The new best loss obtained at the current loop
        """
        self.best_loss = new_best_loss
        self.idx_best_loop = self.loop_counter
    
    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'EncoderManager value:\n'
        s += '--------------------------\n'
        for k in fields(self):
            if k.name == 'preset':
                # Don't print preset, it's quite ugly
                continue

            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s

@dataclass(kw_only=True)
class LossFunctionOutput():
    """Output for Encoder.loss_function"""
    # ----- This is the important output
    # Optional to allow easy inheritance by EncoderLogs
    loss: Optional[float] = None                                        # The RD cost to optimize

    # Any other data required to compute some logs, stored inside a dictionary
    rate_nn_bpd: Optional[float] = None                                 # Rate associated to the neural networks [bpd]
    rate_latent_bpd: Optional[float] = None                             # Rate associated to the latent          [bpd]
    rate_img_bpd: Optional[float] = None                                # Rate associated to the image           [bpd]

@dataclass
class EncoderOutput():
    """Dataclass representing the output of Encoder forward."""

    latent_bpd: Tensor              # Rate associated to each latent [total_latent_value]
    img_bpd: Tensor                 # Rate associated to the image [bpd]
    # Any other data required to compute some logs, stored inside a dictionary
    additional_data: Dict[str, Any] = field(default_factory = lambda: {})

@dataclass
class EncoderLogs(LossFunctionOutput):
    """Output of the test function i.e. the actual results of the encoding
    of one frame by the frame encoder.

    It inherits from LossFunctionOutput, meaning that all attributes of LossFunctionOutput
    are also attributes of EncoderLogs. A EncoderLogs is thus initialized
    from a LossFunctionOutput, all attribute of the LossFunctionOutput  will be copied as
    new attributes for the class.

    This is what is going to be saved to a log file.
    """
    loss_function_output: LossFunctionOutput        # All outputs from the loss function, will be copied is __post_init__
    encoder_output: EncoderOutput        # Output of frame encoder forward
    original_frame: Tensor                           # Non coded frame

    detailed_rate_nn: DescriptorOverfitter            # Rate for each NN weights & bias   [bit]
    quantization_param_nn: DescriptorOverfitter       # Quantization step for each NN weights & bias [ / ]

    encoding_time_second: float                     # Duration of the encoding          [sec]
    encoding_iterations_cnt: int                    # Number of encoding iterations     [ / ]

    # ==================== Not set by the init function ===================== #
    # Everything here is derived from encoder_output and original_frame

    # ----- FNLIC outputs
    # Spatial distribution of the rate, obtained by summing the rate of the different features
    # for each spatial location (in bit). [1, 1, H, W]
    spatial_rate_bit: Optional[Tensor] = field(init=False)
    # Feature distribution of the rate, obtained by the summing all the spatial location
    # of a given feature. [Number of latent resolution]
    feature_rate_bpd: Optional[List[float]] = field(init=False, default_factory=lambda: [])


    # ----- Miscellaneous quantities recovered from self.frame
    img_size: Tuple[int, int] = field(init=False)                   # [Height, Width]
    n_pixels: int = field(init=False)                               # Height x Width
    # ----- Neural network rate in bit per pixels
    detailed_rate_nn_bpd: DescriptorOverfitter = field(init=False)    # Rate for each NN weights & bias   [bpd]

    def __post_init__(self):
        # ----- Copy all the attributes of loss_function_output
        for field in fields(self.loss_function_output):
            setattr(self, field.name, getattr(self.loss_function_output, field.name))

        # ----- Retrieve info from the frame
        self.img_size = self.original_frame.shape[-2:]
        self.n_pixels = self.original_frame.shape[-2] * self.original_frame.shape[-1] * 3

        # ----- Convert rate in bpd
        # Divide each entry of self.detailed_rate_nn by the number of pixel
        self.detailed_rate_nn_bpd: DescriptorOverfitter = {
            module_name: {
                weight_or_bias: rate_in_bits / self.n_pixels
                for weight_or_bias, rate_in_bits in module.items()
            }
            for module_name, module in self.detailed_rate_nn.items()
        }

        # ------ Retrieve things related to the FNLIC from the additional
        # ------ outputs of the frame encoder.
        if 'detailed_rate_bit' in self.encoder_output.additional_data:
            detailed_rate_bit = self.encoder_output.additional_data.get('detailed_rate_bit')
            # Sum on the last three dimensions
            self.feature_rate_bpd = [
                x.sum(dim=(-1, -2, -3)) / (self.img_size[0] * self.img_size[1])
                for x in detailed_rate_bit
            ]

            upscaled_rate = []
            for rate in detailed_rate_bit:
                cur_c, cur_h, cur_w = rate.size()[-3:]

                # Ignore tensor with no channel
                if cur_c == 0:
                    continue

                # Rate is in bit, but since we're going to upsampling the rate values to match
                # the actual image size, we want to keep the total number of bit consistent.
                # To do so, we divide the rate by the upsampling ratio.
                # Example:
                # 2x2 feature maps with 8 bits for each sample gives a 4x4 visualisation
                # with 2 bits per sample. This make the total number of bits stay identical
                rate /=  (self.img_size[0] * self.img_size[1]) / (cur_h * cur_w)
                upscaled_rate.append(
                    nn.functional.interpolate(rate, size=self.img_size, mode='nearest')
                )

            upscaled_rate = th.cat(upscaled_rate, dim=1)
            self.spatial_rate_bit = upscaled_rate.sum(dim=1, keepdim=True)


    def pretty_string(
        self,
        show_col_name: bool = False,
        mode: Literal['all', 'short'] = 'all',
        additional_data: Dict[str, Any] = {}
    ) -> str:
        """Return a pretty string formatting the data within the class.

        Args:
            show_col_name (bool, optional): True to also display col name. Defaults to False.
            mode (str, optional): Either "short" or "all". Defaults to 'all'.

        Returns:
            str: The formatted results
        """
        col_name = ''
        values = ''
        COL_WIDTH = 10
        INTER_COLUMN_SPACE = ' '

        for k in fields(self):
            if not self.should_be_printed(k.name, mode=mode):
                continue

            # ! Deep copying is needed but i don't know why?
            val = copy.deepcopy(getattr(self, k.name))

            if k.name == 'feature_rate_bpd':
                for i in range(len(val)):
                    col_name += f'{k.name + f"_{str(i).zfill(2)}":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    values += f'{self.format_value(val[i], attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

            elif k.name == 'detailed_rate_nn_bpd':
                for subnetwork_name, subnetwork_detailed_rate in val.items():
                    col_name += f'{subnetwork_name + "_rate_bpd":<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                    sum_weight_and_bias = sum([tmp for _, tmp in subnetwork_detailed_rate.items()])
                    values += f'{self.format_value(sum_weight_and_bias, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

            else:
                col_name += f'{self.format_column_name(k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
                values += f'{self.format_value(val, attribute_name=k.name):<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

        for k, v in additional_data.items():
            col_name += f'{k:<{COL_WIDTH}}{INTER_COLUMN_SPACE}'
            values += f'{v:<{COL_WIDTH}}{INTER_COLUMN_SPACE}'

        if show_col_name:
            return col_name + '\n' + values
        else:
            return values

    def should_be_printed(self, attribute_name: str, mode: str) -> bool:
        """Return True if the attribute named <attribute_name> should be printed
        in mode <mode>.

        Args:
            attribute_name (str): Candidate attribute to print
            mode (str): Either "short" or "all"

        Returns:
            bool: True if the attribute should be printed, False otherwise
        """

        # Syntax: {'attribute': [printed in mode xxx]}
        ATTRIBUTES = {
            # ----- This is printed in every modes
            'loss': ['short', 'all'],
            'total_rate_bpd': ['short', 'all'],
            'rate_img_bpd': ['short', 'all'],
            'rate_latent_bpd': ['short', 'all'],
            'rate_nn_bpd': ['short', 'all'],
            'encoding_time_second': ['short', 'all'],
            'encoding_iterations_cnt': ['short', 'all'],

            # ----- This is only printed in mode all
            'feature_rate_bpd': ['all'],
            'detailed_rate_nn_bpd': ['all'],
            'n_pixels': ['all'],
            'img_size': ['all'],
            'mac_decoded_pixel': ['all'],
        }

        if attribute_name not in ATTRIBUTES:
            return False

        if mode not in ATTRIBUTES.get(attribute_name):
            return False

        return True

    def format_value(
        self,
        value: Union[str, int, float, Tensor],
        attribute_name: str = ''
    ) -> str:

        if attribute_name == 'img_size':
            value = 'x'.join([str(tmp) for tmp in value])

        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f'{value:.6f}'
        elif isinstance(value, Tensor):
            return f'{value.item():.6f}'

    def format_column_name(self, col_name: str) -> str:

        # Syntax: {'long_name': 'short_name'}
        LONG_TO_SHORT = {
            'rate_latent_bpd': 'latent_bpd',
            'rate_img_bpd': 'img_bpd',
            'rate_nn_bpd': 'nn_bpd',
            'encoding_time_second': 'time_sec',
            'encoding_iterations_cnt': 'itr',
        }

        if col_name not in LONG_TO_SHORT:
            return col_name
        else:
            return LONG_TO_SHORT.get(col_name)

class FNLIC(nn.Module):
    def __init__(
        self,
        encoder_param: OverfitterParameter,
        encoder_manager: EncoderManager,
        prefitter: Prefitter,
        img_t:Tensor,
    ):
        super().__init__()
        self.img_t_ori = img_t
        self.img_t = pad_image(img_t, encoder_param.n_latents-1)
        self.prefitter = prefitter
        self.prefitter.to_device('cpu')
        self.prefitter.eval()
        th.use_deterministic_algorithms(True)
        self.prior = prefitter.get_prior(self.img_t)
        th.use_deterministic_algorithms(False)
        self.encoder_param = encoder_param
        self.encoder = OverFitter(encoder_param)
        self.encoder_manager = encoder_manager
    
    def set_to_train(self):
        self.encoder.train()
    
    def set_to_eval(self):
        self.prefitter.eval()
        self.encoder.eval()
    
    def forward(self,
                use_ste_quant: bool = False) -> EncoderOutput:
        out = self.encoder(self.img_t, self.prior, use_ste_quant)
        return EncoderOutput(img_bpd=out['img_bpd'], latent_bpd=out['latent_bpd'], additional_data={})

    def loss_function(
        self,
        encoder_out: EncoderOutput,
        rate_mlp_bpd: float = 0.,
    )->LossFunctionOutput:
        loss = encoder_out.img_bpd + rate_mlp_bpd + encoder_out.latent_bpd
        return LossFunctionOutput(
            loss=loss,
            rate_nn_bpd=rate_mlp_bpd,
            rate_latent_bpd=encoder_out.latent_bpd,
            rate_img_bpd=encoder_out.img_bpd
        )

    @th.no_grad()
    def test(self, training=True) -> EncoderLogs:
        # 1. Get the rate associated to the network ----------------------------- #
        # The rate associated with the network is zero if it has not been quantize
        # before calling the test functions
        rate_mlp = 0.
        rate_per_module = self.encoder.get_network_rate()
        for _, module_rate in rate_per_module.items():
            for _, param_rate in module_rate.items():   # weight, bias
                rate_mlp += param_rate

        # 2. Measure performance ------------------------------------------------ #
        self.set_to_eval()

        # flag_additional_outputs set to True to obtain more output
        encoder_out = self.forward(use_ste_quant=False)

        loss_fn_output = self.loss_function(
            encoder_out,
            rate_mlp_bpd=rate_mlp/self.encoder.img_size,
        )

        encoder_logs = EncoderLogs(
            loss_function_output=loss_fn_output,
            encoder_output=encoder_out,
            original_frame=self.img_t_ori,
            detailed_rate_nn=rate_per_module,
            quantization_param_nn=self.encoder.get_network_quantization_step(),
            encoding_time_second=self.encoder_manager.total_training_time_sec,
            encoding_iterations_cnt=self.encoder_manager.iterations_counter,
        )

        # 3. Restore training mode ---------------------------------------------- #
        if training:
            self.set_to_train()

        return encoder_logs

    def warmup(self, device: POSSIBLE_DEVICE, alpha_init:float=1.0):
        """

        Perform the warm-up i.e. N different mini training to select the best
        starting point. At the end of the warm-up, the starting point is registered
        as an attribute in self.encoder.

        Args:
            device (POSSIBLE_DEVICE): The device on which the model should run.
        """
        start_time = time.time()

        training_preset = self.encoder_manager.preset
        msg = '\nStarting warm up...'
        msg += f' Number of warm-up iterations: {training_preset.get_total_warmup_iterations()}\n'
        logging.info(msg)

        _col_width = 14

        for idx_warmup_phase, warmup_phase in enumerate(training_preset.all_warmups):
            logging.info(f'{"-" * 30}  Warm-up phase: {idx_warmup_phase:>2} {"-" * 30}')

            # mem_info(f"Warmup-{idx_warmup_phase:02d}")

            # At the beginning of the first warmup phase, we must initialize all the models
            if idx_warmup_phase == 0:
                all_candidates = [
                    {
                        'model': OverFitter(self.encoder_param, alpha_init=alpha_init),
                        'metrics': None,
                        'id': idx_model
                    }
                    for idx_model in range(warmup_phase.candidates)
                ]

            # At the beginning of the other warm-up phases, keep the desired number of best candidates
            else:
                all_candidates = all_candidates[:warmup_phase.candidates]

            # Construct the training phase object describing the options of this particular warm-up phase
            training_phase = TrainerPhase(
                lr=warmup_phase.lr,
                max_itr=warmup_phase.iterations,
                start_temperature_softround=0.3,
                end_temperature_softround=0.3,
                start_kumaraswamy=2.0,
                end_kumaraswamy=2.0,
            )

            # ! idx_candidate is just the index of one candidate in the all_candidates list. It is **not** a
            # ! unique identifier for this candidate. This is given by:
            # !         all_candidates[idx_candidate].get('id')
            # ! the all_candidates list gives the ordered list of the best performing models so its order may change.
            for idx_candidate, candidate in enumerate(all_candidates):
                logging.info(f'\nCandidate n° {idx_candidate:<2}, ID = {candidate.get("id"):<2}:')
                logging.info(f'-------------------------\n')
                # mem_info(f"Warmup-cand-in {idx_warmup_phase:02d}-{idx_candidate:02d}")

                # Use the current candidate as our actual Cool-chic encoder
                self.encoder = candidate.get('model')
                self.encoder.to_device(device)

                # ! One training phase goes here!
                encoder_logs = self.one_training_phase(training_phase)

                self.encoder.to_device('cpu')

                # Store the updated candidate on CPU
                all_candidates[idx_candidate] = {
                    'model': self.encoder,
                    'metrics': encoder_logs,
                    'id': candidate.get('id')
                }
                # mem_info(f"Warmup-cand-out{idx_warmup_phase:02d}-{idx_candidate:02d}")

            # Sort all the models by ascending loss. The best one is all_candidates[0]
            all_candidates = sorted(all_candidates, key=lambda x: x.get('metrics').loss)

            # Print the results of this warm-up phase
            s = f'\n\nPerformance at the end of the warm-up phase:\n\n'
            s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"img_bpd":^{_col_width}}|{"latent_bpd":^{_col_width}}|\n'
            s += f'------|{"-" * _col_width}|{"-" * _col_width}|{"-" * _col_width}|\n'
            for candidate in all_candidates:
                s += f'{candidate.get("id"):^{6}}|'
                s += f'{candidate.get("metrics").loss.item():^{_col_width}.4f}|'
                s += f'{candidate.get("metrics").rate_img_bpd:^{_col_width}.4f}|'
                s += f'{candidate.get("metrics").rate_latent_bpd:^{_col_width}.4f}|'
                s += '\n'
            logging.info(s)

        # Keep only the best model
        best_model = all_candidates[0].get('model')
        self.encoder = best_model

        # We've already worked for that many second during warm up
        warmup_duration =  time.time() - start_time

        logging.info(f'Warm-up is done!')
        logging.info(f'Warm-up time [s]: {warmup_duration:.2f}')
        logging.info(f'Winner ID       : {all_candidates[0].get("id")}\n')

    def one_training_phase(self, trainer_phase: TrainerPhase):
        start_time = time.time()

        # ==== Keep track of the best loss and model for *THIS* current phase ==== #
        # Perform a first test to get the current best logs (it includes the loss)
        initial_encoder_logs = self.test()
        encoder_logs_best = initial_encoder_logs
        # ! Maybe self.cool_chic_encoder.state_dict()?
        this_phase_best_model = OrderedDict(
            (k, v.detach().clone()) for k, v in self.state_dict().items()
        )
        # ==== Keep track of the best loss and model for *THIS* current phase ==== #

        self.set_to_train()

        # =============== Build the list of parameters to optimize ============== #
        # Iteratively construct the list of required parameters... This is kind of a
        # strange syntax, which has been found quite empirically

        parameters_to_optimize = []
        if 'arm' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.encoder.arm.parameters()]
        if 'upsampling' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.encoder.upsampling.parameters()]
        if 'synthesis' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.encoder.synthesis.parameters()]
        if 'latent' in trainer_phase.optimized_module:
            parameters_to_optimize += [*self.encoder.latents.parameters()]
        if 'all' in trainer_phase.optimized_module:
            parameters_to_optimize = self.parameters()

        optimizer = th.optim.Adam(parameters_to_optimize, lr=trainer_phase.lr)
        # =============== Build the list of parameters to optimize ============== #

        scheduler = False
        # Scheduler for a single monotonic decrease of the learning rate from
        # trainer_phase.lr to 0.
        if trainer_phase.scheduling_period:
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trainer_phase.max_itr / trainer_phase.freq_valid,
                eta_min=0,
                last_epoch=-1,
                verbose=False
            )

        # Custom scheduling function for the soft rounding temperature and the kumaraswamy param
        def linear_schedule(initial_value, final_value, cur_itr, max_itr):
            return cur_itr * (final_value - initial_value) / max_itr + initial_value

        # Initialize soft rounding temperature and kumaraswamy parameter
        cur_tmp = linear_schedule(
            trainer_phase.start_temperature_softround,
            trainer_phase.end_temperature_softround,
            0,
            trainer_phase.max_itr
        )
        kumaraswamy_param = linear_schedule(
            trainer_phase.start_kumaraswamy,
            trainer_phase.end_kumaraswamy,
            0,
            trainer_phase.max_itr
        )

        self.encoder.noise_quantizer.soft_round_temperature = cur_tmp
        self.encoder.ste_quantizer.soft_round_temperature = cur_tmp
        self.encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param

        cnt_record = 0
        show_col_name = True

        # phase optimization
        for cnt in range(trainer_phase.max_itr):
            if cnt - cnt_record > trainer_phase.patience:

                # if no scheduler, exceeding the patience level ends the phase
                if not scheduler:
                    break

                # If we have a scheduler, we reload the previous best mode and resume from the current lr
                else:
                    self.load_state_dict(this_phase_best_model)
                    current_lr = scheduler.get_last_lr()[0]

                if False:
                    # we re-initiate the scheduler
                    T_max = ( cnt - trainer_phase.max_itr) // trainer_phase.freq_valid
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0, last_epoch=-1, verbose=False)
                    logging.info(f"Reload best model and new schedule tmax={T_max} and lr={current_lr:.6f}")

                else:
                    # we keep on with the current scheduler
                    logging.info(f"Reload best model lr={current_lr:.6f}")

                cnt_record = cnt

            # This is slightly faster than optimizer.zero_grad()
            for param in self.parameters():
                param.grad = None

            # forward / backward
            out_forward = self.forward(use_ste_quant=trainer_phase.ste)
            loss_function_output = self.loss_function(out_forward)
            loss_function_output.loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1e-1, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

            self.encoder_manager.iterations_counter += 1

            # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
            if ((cnt + 1) % trainer_phase.freq_valid == 0) or (cnt + 1 == trainer_phase.max_itr):
                #  a. Update iterations counter and training time and test model
                self.encoder_manager.total_training_time_sec += time.time() - start_time
                start_time = time.time()

                # b. Test the model and check whether we've beaten our record
                encoder_logs = self.test()

                flag_new_record = False

                if encoder_logs.loss < encoder_logs_best.loss:
                    # A record must have at least -0.0005 bpd. A smaller improvement
                    # does not matter.
                    delta_bpd  = encoder_logs.loss - encoder_logs_best.loss
                    flag_new_record = delta_bpd < -0.0005

                if flag_new_record:
                    # Save best model
                    for k, v in self.state_dict().items():
                        this_phase_best_model[k].copy_(v)

                    # ========================= reporting ========================= #
                    this_phase_loss_gain =  100 * (encoder_logs.loss - initial_encoder_logs.loss) / encoder_logs.loss

                    log_new_record = ''
                    log_new_record += f'{this_phase_loss_gain:+6.3f} % '
                    # ========================= reporting ========================= #

                    # Update new record
                    encoder_logs_best = encoder_logs
                    cnt_record = cnt
                else:
                    log_new_record = ''

                # Show column name a single time
                additional_data = {
                    'STE': trainer_phase.ste,
                    'lr': f'{trainer_phase.lr if not scheduler else scheduler.get_last_lr()[0]:.8f}',
                    'optim': ','.join(trainer_phase.optimized_module),
                    'patience': (trainer_phase.patience - cnt + cnt_record) // trainer_phase.freq_valid,
                    'sr_temp': f'{cur_tmp:.5f}',
                    'kumara': f'{kumaraswamy_param:.5f}',
                    'record': log_new_record
                }
                logging.info(
                    encoder_logs.pretty_string(
                        show_col_name=show_col_name,
                        mode='short',
                        additional_data=additional_data
                    )
                )
                show_col_name = False


                # Update soft rounding temperature and kumaraswamy noise
                cur_tmp = linear_schedule(
                    trainer_phase.start_temperature_softround,
                    trainer_phase.end_temperature_softround,
                    cnt,
                    trainer_phase.max_itr
                )
                kumaraswamy_param = linear_schedule(
                    trainer_phase.start_kumaraswamy,
                    trainer_phase.end_kumaraswamy,
                    cnt,
                    trainer_phase.max_itr
                )

                self.encoder.noise_quantizer.soft_round_temperature = cur_tmp
                self.encoder.ste_quantizer.soft_round_temperature = cur_tmp
                self.encoder.noise_quantizer.kumaraswamy_param = kumaraswamy_param

                # Update scheduler
                if scheduler:
                    scheduler.step()

                # Restore training mode
                self.set_to_train()

        # Load best model found for this encoding loop
        self.load_state_dict(this_phase_best_model)

        # Quantize the model parameters at the end of the training phase
        
        if trainer_phase.quantize_model:
            self.quantize_model()

        # Final test to eventual retrieve the performance of the model
        encoder_logs = self.test()

        return encoder_logs

    @th.no_grad()
    def quantize_model(self):
        """Quantize the current model, in place!.
        # ! We also obtain the integerized ARM here!"""

        start_time = time.time()
        self.set_to_eval()

        # We have to quantize all the modules that we want to send
        module_to_quantize = {
            module_name: getattr(self.encoder, module_name)
            for module_name in self.encoder.modules_to_send
        }

        best_q_step = {k: None for k in module_to_quantize}

        for module_name, module in module_to_quantize.items():
            # Start the RD optimization for the quantization step of each module with an
            # arbitrary high value for the RD cost.
            best_loss = 1e6

            # Save full precision parameters before quantizing
            module.save_full_precision_param()

            # Try to find the best quantization step
            all_q_step = module._POSSIBLE_Q_STEP
            for q_step_w, q_step_b in itertools.product(all_q_step, all_q_step):
                # Quantize
                current_q_step: DescriptorNN = {'weight': q_step_w, 'bias': q_step_b}
                quantization_success = module.quantize(current_q_step)

                if not quantization_success:
                    continue

                # Measure rate
                rate_per_module = module.measure_laplace_rate()
                total_rate_module_bit = sum([v for _, v in rate_per_module.items()])

                # Evaluate

                # ===================== Integerization of the ARM ===================== #
                if module_name == 'arm':
                    if ARMINT:
                        self.encoder = self.encoder.to_device('cpu')
                    module.set_quant(FIXED_POINT_FRACTIONAL_MULT)
                # ===================== Integerization of the ARM ===================== #

                encoder_out = self.forward()

                # Compute results
                loss_function_output = self.loss_function(encoder_out, total_rate_module_bit/self.encoder.img_size)

                # Store best quantization steps
                if loss_function_output.loss < best_loss:
                    best_loss = loss_function_output.loss
                    best_q_step[module_name] = current_q_step

            # Once we've tested all the possible quantization step: quantize one last
            # time with the best one we've found to actually use it.
            quantization_success = module.quantize(best_q_step[module_name])

            if not quantization_success:
                logging.info(f'Greedy quantization failed!')
                sys.exit(0)

        logging.info(f'\nTime greedy_quantization: {time.time() - start_time:4.1f} seconds\n')

        # Re-apply integerization of the module
        self.encoder.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)
       
    def one_training_loop(self,
                           device: POSSIBLE_DEVICE,
                           frame_workdir: str,
                           start_time: str=0.,
                           alpha_init:float=1.0):
        """Main training function of a Encoder. It requires a encoder_save_path
        in order to save the encoder periodically to allow for checkpoint.

        Args:
            device (POSSIBLE_DEVICE): On which device should the training run
            encoder_save_path (str): Where to checkpoint the model
            path_original_sequence (str): Path to the raw .yuv file with the video to code.
                This should not really be seen by a Encoder, but we need it to perform the
                inter warm-up where the references are shifted.
            start_time (float): Keep track of the when we started the overall training to
                requeue if need be

        Returns:
            (TrainingExitCode): Exit code

        """
        msg = '-' * 80 + '\n'
        msg += f'{" " * 30} Training loop {self.encoder_manager.loop_counter + 1} / {self.encoder_manager.n_loops}\n'
        msg += '-' * 80
        logging.info(msg)
        self.to_device(device)

        # if not self.encoder_manager.warm_up_done:
        self.warmup(device, alpha_init=alpha_init)
        self.to_device(device)
            # self.encoder_manager.warm_up_done = True

        # Save model after checkpoint
        # if is_job_over(start_time):
        #     return TrainingExitCode.REQUEUE

        # Perform the successive training phase from phase_encoder_manager.phase_idx to
        # the total number of phase.
        # The counter phase_encoder_manager.phase_idx is incremented by the function
        # self.one_training_phase()
        for idx_phase in range(self.encoder_manager.phase_idx, len(self.encoder_manager.preset.all_phases)):
            logging.info(f'{"-" * 30} Training phase: {idx_phase:>2} {"-" * 30}\n')
            self.one_training_phase(self.encoder_manager.preset.all_phases[idx_phase])
            self.encoder_manager.phase_idx += 1

            logging.info(f'\nResults at the end of the phase:')
            logging.info('--------------------------------')
            logging.info(f'\n{self.test().pretty_string(show_col_name=True, mode="short")}\n')

            # if is_job_over(start_time):
            #     return TrainingExitCode.REQUEUE

        # At the end of each loop, compute the final loss
        encoder_logs = self.test()

        # Write results file
        with open(f'{frame_workdir}results_loop_{self.encoder_manager.loop_counter + 1}.tsv', 'w') as f_out:
            f_out.write(encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

        # We've beaten our record
        if self.encoder_manager.record_beaten(encoder_logs.loss):
            logging.info(f'Best loss beaten at loop {self.encoder_manager.loop_counter + 1}')
            logging.info(f'Previous best loss: {self.encoder_manager.best_loss :.6f}')
            logging.info(f'New best loss     : {encoder_logs.loss.cpu().item() :.6f}')

            self.encoder_manager.set_best_loss(encoder_logs.loss.cpu().item())

            # Save best results
            with open(f'{frame_workdir}results_best.tsv', 'w') as f_out:
                f_out.write(encoder_logs.pretty_string(show_col_name=True, mode='all') + '\n')

            # # Generate the visualisation for the best frame encoder
            # self.generate_visualisation(f'{frame_workdir}')

        # Increment the loop counter, reset the warm up flag and the phase idx counter
        self.encoder_manager.loop_counter += 1
        self.encoder_manager.warm_up_done = False
        self.encoder_manager.phase_idx = 0

        # We're done with this frame!
        return TrainingExitCode.END
        
    def overfit(self,
              device:POSSIBLE_DEVICE,
              work_dir: str, alpha_init:float) -> TrainingExitCode:
        start_time = time.time()
        exit_code = self.one_training_loop(device, work_dir, start_time, alpha_init=alpha_init)
        return exit_code
        
    def save(self, path: str):
        self.encoder.save(path)

    @th.no_grad()
    def test_inference_time(self) -> float:
        self.set_to_eval()
        latent = self.encoder.get_quantized_latent()
        latent = th.cat([cur_latent.flatten() for cur_latent in latent])
        max_latent_v = int(th.ceil(latent.abs().max()).item())
        with Timer(str(self.img_t.device)) as t:
            prior = self.prefitter.get_prior(self.img_t)
            self.encoder.inference_for_decode(self.img_t, prior, max_latent_v)
        return t.result

    def to_device(self, device: POSSIBLE_DEVICE):
        self.encoder.to_device(device)
        self.img_t = self.img_t.to(device)
        self.prior = self.prior.to(device)
        self.prefitter.to_device(device)

def load_fnlic(src: str, overfitter_param: OverfitterParameter, img_t:Tensor, prefitter:Prefitter) -> FNLIC:
    """
        Load FNLIC for encoding
    """
    # Reset the stream position to the beginning of the BytesIO object & load it

    encoder = OverFitter(overfitter_param)
    encoder.load(src)

    # Create a frame encoder from the stored parameters
    fnlic = FNLIC(
        encoder_param=overfitter_param,
        encoder_manager=None,
        img_t=img_t,
        prefitter=prefitter
    )
    fnlic.encoder = encoder

    return fnlic