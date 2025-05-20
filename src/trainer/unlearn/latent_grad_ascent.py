from trainer.unlearn.base import UnlearnTrainer

import torch
import torch.nn.functional as F
from torch import Tensor
from enum import Enum
import os
import math

from captum.attr import LayerIntegratedGradients

from transformer_lens.utils import get_act_name
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint


# INTEGRATED GRADIENTS #


def run_from_layer_fn(
    model: HookedTransformer,
    original_input: Tensor,
    patch_layer: HookPoint,
    patch_output: Tensor,
    metric: callable,
    metric_labels: Tensor,
    reset_hooks_end=True,
):
    """
    Runs the model with a specified input to an internal layer.
    Runs with original_input, but patches in patch_output at patch_layer.
    """

    def fwd_hook(act, hook):
        assert (
            patch_output.shape == act.shape
        ), f"Patch shape {patch_output.shape} != activation shape {act.shape}"
        return patch_output + 0 * act  # Trick to ensure gradients propagate

    logits = model.run_with_hooks(
        original_input,
        fwd_hooks=[(patch_layer.name, fwd_hook)],
        reset_hooks_end=reset_hooks_end,
    )
    diff = metric(logits, metric_labels)
    return diff


def compute_layer_to_output_attributions(
    model,
    original_input,
    layer_input,
    layer_baseline,
    target_layer,
    prev_layer,
    metric,
    metric_labels,
):
    """
    Calculates layerwise integrated gradient scores for target_layer.
    Uses layer_input / layer_baseline as IG input / baseline parameters.
    """
    n_samples = original_input.size(0)

    # Take the model starting from the target layer
    def forward_fn(x):
        return run_from_layer_fn(
            model, original_input, prev_layer, x, metric, metric_labels
        )

    # Attribute to the target_layer's output
    ig_embed = LayerIntegratedGradients(
        forward_fn, target_layer, multiply_by_inputs=True
    )
    attributions, error = ig_embed.attribute(
        inputs=layer_input,
        baselines=layer_baseline,
        internal_batch_size=n_samples,
        attribute_to_layer_input=False,
        return_convergence_delta=True,
    )
    print(f"\nError (delta) for {target_layer.name} attribution: {error}")
    return attributions


def integrated_gradients(
    model: HookedTransformer,
    baseline_tokens: torch.Tensor,
    baseline_cache: ActivationCache,
    input_cache: ActivationCache,
    metric: callable,
    metric_labels,
):
    """
    Calculates layerwise integrated gradients for every MLP neuron and
    attention head in the model. Uses corrupt_cache activations as input
    and clean_cache activations as baseline.
    """
    n_samples = baseline_tokens.size(0)

    # Gradient attribution for neurons in MLP layers
    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    # Gradient attribution for attention heads
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    # Calculate integrated gradients for each layer
    for layer in range(model.cfg.n_layers):

        # Gradient attribution on heads
        hook_name = get_act_name("result", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("z", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_input = input_cache[prev_layer_hook]
        layer_baseline = baseline_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_head, d_model]
        attributions = compute_layer_to_output_attributions(
            model,
            baseline_tokens,
            layer_input,
            layer_baseline,
            target_layer,
            prev_layer,
            metric,
            metric_labels,
        )

        # Calculate score based on mean over each embedding, for each token
        per_token_score = attributions.mean(dim=3)
        score = per_token_score.mean(dim=1)
        attn_results[:, layer] = score

        # Gradient attribution on MLP neurons
        hook_name = get_act_name("post", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("mlp_in", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_input = input_cache[prev_layer_hook]
        layer_baseline = baseline_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_model]
        attributions = compute_layer_to_output_attributions(
            model,
            baseline_tokens,
            layer_input,
            layer_baseline,
            target_layer,
            prev_layer,
            metric,
            metric_labels,
        )
        score = attributions.mean(dim=1)
        mlp_results[:, layer] = score

    return mlp_results, attn_results


def highlight_components(attribution_scores):
    """
    Return a binary tensor of the same shape as attribution_scores, with 1s in components
    with high attribution scores ("important" components).
    Also returns the indices of the highlighted components.
    """
    mean_scores = torch.mean(attribution_scores, dim=(1, 2), keepdim=True)
    std_scores = torch.std(attribution_scores, dim=(1, 2), keepdim=True)

    highlighted_components = attribution_scores.abs() > (mean_scores + 1.96 * std_scores)
    highlighted_indices = highlighted_components.nonzero()
    return highlighted_components, highlighted_indices


def correct_logit_metric(logits, forget_tokens):
    """
    Return cross entropy between the logits and the correct answer.
    """
    return F.cross_entropy(logits, forget_tokens, reduction="sum")


class AttributionMethod(Enum):
    IG_REWRITE_ORIGINAL = 1  # IG with rewrite baseline and original input
    IG_ORIGINAL_REWRITE = 2  # IG with original baseline and rewrite input
    AP_ORIGINAL_REWRITE = 3  # AP with original clean and rewrite corrupt


def get_save_location(method: AttributionMethod, sample_index: int):
    if method == AttributionMethod.IG_REWRITE_ORIGINAL:
        return f"results/counterfact/ig_rewrite_original_{sample_index}.pt"
    elif method == AttributionMethod.IG_ORIGINAL_REWRITE:
        return f"results/counterfact/ig_original_rewrite_{sample_index}.pt"


def run_attribution_steps(
    model: HookedTransformer,
    original_tokens: Tensor,
    rewrite_tokens: Tensor,
    answer_labels: Tensor,
    original_cache: ActivationCache,
    rewrite_cache: ActivationCache,
    sample_id: str,
    overwrite=False,
):
    """
    Run three types of attribution methods on the given data samples.
    Returns a dictionary of highlighted components per attribution method, for MLP and attention heads each.
    Warning: do not use "overwrite" if working with many batches - inefficient!
    """
    mlp_attribution_highlights = dict()
    attn_attribution_highlights = dict()

    # Run integrated gradients with original baseline and rewrite input
    ig_original_rewrite_path = get_save_location(AttributionMethod.IG_ORIGINAL_REWRITE, sample_id)
    if not overwrite and os.path.exists(ig_original_rewrite_path):
        ig_original_rewrite_mlp, ig_original_rewrite_attn = torch.load(
            ig_original_rewrite_path
        )
    else:
        ig_original_rewrite_mlp, ig_original_rewrite_attn = integrated_gradients(
            model,
            original_tokens,
            original_cache,
            rewrite_cache,
            correct_logit_metric,
            answer_labels,
        )
        torch.save(
            (ig_original_rewrite_mlp, ig_original_rewrite_attn),
            ig_original_rewrite_path,
        )

    mlp_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_attn)
    )

    # Run integrated gradients with rewrite baseline and original input
    ig_rewrite_original_path = get_save_location(AttributionMethod.IG_REWRITE_ORIGINAL, sample_id)
    if not overwrite and os.path.exists(ig_rewrite_original_path):
        ig_rewrite_original_mlp, ig_rewrite_original_attn = torch.load(
            ig_rewrite_original_path
        )
    else:
        ig_rewrite_original_mlp, ig_rewrite_original_attn = integrated_gradients(
            model,
            rewrite_tokens,
            rewrite_cache,
            original_cache,
            correct_logit_metric,
            answer_labels,
        )
        torch.save(
            (ig_rewrite_original_mlp, ig_rewrite_original_attn),
            ig_rewrite_original_path,
        )

    mlp_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_attn)
    )

    return mlp_attribution_highlights, attn_attribution_highlights


class LatentGradAscent(UnlearnTrainer):

    def __init__(self):
        super().__init__()
        self.hooked_model = None
        self.target_mlp_components = None
        self.target_attn_components = None


    def localise_forget_components(self, model, forget_inputs, labels):
        assert model.config.name_or_path == self.hooked_model.cfg.model_name, \
            "Model and hooked model must be the same"

        original_input = forget_inputs["input_ids"]
        original_input = [f"{inp}: " for inp in original_input]
        n_samples = original_input.size(0)

        rewrite_input = forget_inputs["input_ids"].clone()
        rewrite_input = [f"{inp} (DO NOT ANSWER CORRECTLY): " for inp in original_input]

        tokenised = self.hooked_model.to_tokens(original_input + rewrite_input, prepend_bos=False)
        original_tokens, rewrite_tokens = [tokenised[i:i + n_samples] for i in range(0, len(tokenised), n_samples)]
        labels = forget_inputs["labels"]

        _, original_cache = self.hooked_model.run_with_cache(original_tokens)
        _, rewrite_cache = self.hooked_model.run_with_cache(rewrite_tokens)

        sample_id = str(hash(original_tokens))

        answer_tokens = self.hooked_model.to_tokens(labels, prepend_bos=False)

        mlp_attribution_highlights, attn_attribution_highlights = run_attribution_steps(
            self.hooked_model,
            original_tokens,
            rewrite_tokens,
            answer_tokens,
            original_cache,
            rewrite_cache,
            sample_id,
        )

        return mlp_attribution_highlights, attn_attribution_highlights
    

    def identify_target_components(self, highlighted_dict: dict):
        ig_rewrite_original_highlighted = highlighted_dict[
            AttributionMethod.IG_REWRITE_ORIGINAL
        ]
        ig_original_rewrite_highlighted = highlighted_dict[
            AttributionMethod.IG_ORIGINAL_REWRITE
        ]
        return ig_rewrite_original_highlighted | ig_original_rewrite_highlighted
    

    def freeze_non_target_components(self, target_attn_components, target_mlp_components):
        # Mask out gradients at non-target components
        with torch.no_grad():
            for layer_idx in range(self.hooked_model.cfg.n_layers):
                # Attention components: W_K, W_Q, W_V, W_O matrices
                # Match attention weight shape [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
                layer_attn_weight_mask = target_attn_components[:, layer_idx].view(
                    self.hooked_model.cfg.n_heads, 1, 1
                )
                # Match attention bias shape [n_heads, d_head]
                layer_attn_bias_mask = target_attn_components[:, layer_idx].view(
                    self.hooked_model.cfg.n_heads, 1
                )

                self.hooked_model.blocks[layer_idx].attn.W_K.grad *= layer_attn_weight_mask  # shape []
                self.hooked_model.blocks[layer_idx].attn.b_K.grad *= layer_attn_bias_mask

                self.hooked_model.blocks[layer_idx].attn.W_Q.grad *= layer_attn_weight_mask
                self.hooked_model.blocks[layer_idx].attn.b_Q.grad *= layer_attn_bias_mask

                self.hooked_model.blocks[layer_idx].attn.W_V.grad *= layer_attn_weight_mask
                self.hooked_model.blocks[layer_idx].attn.b_V.grad *= layer_attn_bias_mask

                self.hooked_model.blocks[layer_idx].attn.W_O.grad *= layer_attn_weight_mask
                # Attention output biases of shape [d_model,] - no need to mask on specific head

                # MLP neuron components: W_in, W_out matrices
                layer_mlp_mask = target_mlp_components[layer_idx]  # shape [d_mlp,]
                self.hooked_model.blocks[layer_idx].mlp.W_in.grad *= layer_mlp_mask.view(
                    1, self.hooked_model.cfg.d_mlp
                )  # shape [d_model, d_mlp]
                self.hooked_model.blocks[layer_idx].mlp.W_out.grad *= layer_mlp_mask.view(
                    self.hooked_model.cfg.d_mlp, 1
                )  # shape [d_mlp, d_model]
                self.hooked_model.blocks[layer_idx].mlp.b_in.grad *= layer_mlp_mask  # shape [d_mlp,]
                # MLP output biases of shape [d_model,] - no need to mask on specific neuron


    def inverted_hinge_loss(output_logits, target_index):
        logit_probs = torch.softmax(output_logits, dim=-1)
        # Get probability of target token for each sample
        # target_prob = torch.gather(logit_probs, dim=1, index=target_index.unsqueeze(1))
        target_prob = logit_probs[:, target_index]
        # Get max probability of non-target tokens
        nontarget_probs = logit_probs.clone()
        nontarget_probs[:, target_index] = -math.inf
        max_nontarget_prob = torch.max(nontarget_probs, dim=-1)[0]
        # Calculate IHL
        return (1 + target_prob - max_nontarget_prob).sum()


    def compute_loss(self, model, inputs, return_outputs=False):

        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = self.inverted_hinge_loss(outputs.logits, forget_inputs["labels"]) + 
        
        return (loss, outputs) if return_outputs else loss
    

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass
        loss = self.compute_loss(model, inputs)

        # Backward pass
        loss.backward()

        # Load a hooked model for localisation
        self.hooked_model = HookedTransformer.from_pretrained(hf_model=model)
        # Explicitly calculate and expose the result for each attention head
        self.hooked_model.set_use_attn_result(True)
        self.hooked_model.set_use_hook_mlp_in(True)

        # Identify target components for unlearning
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "labels": forget_inputs["labels"],
        }

        mlp_attribution_highlights, attn_attribution_highlights = self.localise_forget_components(model, forget_inputs)
        target_mlp = self.identify_target_components(mlp_attribution_highlights)
        target_attn = self.identify_target_components(attn_attribution_highlights)

        # Only fine tune target components
        self.freeze_non_target_components(target_attn, target_mlp)

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss.detach()
