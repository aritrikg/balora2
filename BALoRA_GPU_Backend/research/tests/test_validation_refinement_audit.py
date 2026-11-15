import hashlib
import json
from pathlib import Path
import time
import zipfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.validation_audit import (compare_validation, locate_results, prepare_reference,
                                         prepare_audit_resume, validate_saved_split)
from models.refinement_guarded import GuardedRefinementConfig, refine_guarded
from tests.test_sam2_wrapper_fake import build_fake
from tests.test_training_and_refinement_fake import OneClip, collate
from utils.runtime_budget import BudgetExpired


class ProposalModel:
    def __init__(self, probability=0.95, varied=False, invalid=False):
        self.probability = probability
        self.varied = varied
        self.invalid = invalid
        self.points = []
        self.calls = []

    def select_ranks_from_uncertainty(self, embedding, points, labels, **kwargs):
        self.points.append(points.clone())
        assert labels.tolist() == [[1]]
        return {"layer": 4}, None

    def decode_region(self, embedding, point, bbox, ranks, k_samples):
        self.calls.append((point, bbox))
        y0, x0, y1, x1 = bbox
        probabilities = torch.full((k_samples, 1, y1-y0+1, x1-x0+1), self.probability)
        if self.varied:
            probabilities[0] = 0.01
            probabilities[1] = 0.99
        result = torch.logit(probabilities)
        if self.invalid:
            result[0, 0, 0, 0] = float("nan")
        return result


def maps():
    probability = torch.full((32, 32), 0.01)
    probability[3:26, 3:26] = 0.6
    uncertainty = torch.zeros_like(probability)
    uncertainty[10:15, 10:15] = 0.5
    point = torch.tensor([[[4., 4.]]])
    return probability, uncertainty, point


def config(**kwargs):
    return GuardedRefinementConfig(max_iters=1, crop_padding=2, region_min_area=1, k_samples=2, **kwargs)


def test_preserves_original_target_prompt_and_crop_context():
    p, u, point = maps()
    model = ProposalModel()
    result, _, trace = refine_guarded(model, {}, p, u, point, config())
    assert model.points[0].tolist() == point.tolist()
    assert model.calls[0][0] == (4, 4)
    assert model.calls[0][1] == (1, 1, 27, 27)
    assert trace[0].accepted and float(result[12, 12]) > float(p[12, 12])


def test_never_overwrites_confident_pixels_inside_crop():
    p, u, point = maps()
    original_p, original_u = p.clone(), u.clone()
    refined, epi, trace = refine_guarded(ProposalModel(), {}, p, u, point, config())
    active = u > 0
    assert trace[0].accepted
    assert torch.equal(refined[~active], p[~active])
    assert torch.equal(epi[~active], u[~active])
    assert torch.equal(p, original_p) and torch.equal(u, original_u)


def test_rejects_increased_predictive_entropy_without_changing_mask():
    p, u, point = maps()
    result, epi, trace = refine_guarded(ProposalModel(0.5), {}, p, u, point, config())
    assert trace[0].reason == "predictive_entropy_increased"
    assert not trace[0].accepted
    assert torch.equal(result, p) and torch.equal(epi, u)


def test_rejects_a_proposal_with_greater_epistemic_uncertainty():
    p, u, point = maps()
    result, _, trace = refine_guarded(ProposalModel(varied=True), {}, p, u, point, config())
    assert trace[0].reason == "epistemic_not_improved"
    assert torch.equal(result, p)


def test_rejects_loss_of_original_positive_prompt():
    p, u, point = maps()
    point = torch.tensor([[[12., 12.]]])
    result, _, trace = refine_guarded(ProposalModel(0.05), {}, p, u, point, config())
    assert trace[0].reason == "positive_prompt_lost"
    assert torch.equal(result, p)


def test_nonfinite_proposal_is_recorded_and_baseline_retained():
    p, u, point = maps()
    result, _, trace = refine_guarded(ProposalModel(invalid=True), {}, p, u, point, config())
    assert trace[0].reason == "nonfinite_proposal"
    assert torch.equal(result, p) and torch.isfinite(result).all()


def test_remote_background_uncertainty_does_not_invent_a_new_organ_prompt():
    p, u, point = maps()
    u.zero_(); u[29:32, 0:3] = 0.5
    model = ProposalModel()
    result, _, trace = refine_guarded(model, {}, p, u, point, config())
    assert model.calls == [] and trace == [] and torch.equal(result, p)


def test_missing_predicted_component_at_prompt_keeps_baseline():
    p, u, point = maps()
    p[4, 4] = 0.4
    model = ProposalModel()
    result, _, trace = refine_guarded(model, {}, p, u, point, config())
    assert model.calls == [] and trace == [] and torch.equal(result, p)


def test_expired_deadline_does_not_start_crop_work():
    p, u, point = maps()
    model = ProposalModel()
    with pytest.raises(BudgetExpired):
        refine_guarded(model, {}, p, u, point, config(), deadline=time.time()-1)
    assert model.calls == []


def test_rejects_bad_inputs_even_if_zero_iterations_requested():
    p, u, point = maps()
    cfg = config(); cfg.max_iters = 0; cfg.k_samples = 1
    with pytest.raises(ValueError, match="sample count"):
        refine_guarded(ProposalModel(), {}, p, u, point, cfg)


def saved_config():
    splits = {"train": [f"train{i:02d}" for i in range(19)],
              "validation": [f"val{i:02d}" for i in range(5)],
              "test": [f"test{i:02d}" for i in range(6)]}
    digest = hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"experiment": {"split_case_ids": splits, "split_hash": digest, "seed": 42},
            "refinement": {"confidence_threshold": 0.85, "max_iters": 1, "crop_padding": 2},
            "confidence": {"region_min_area": 1}, "bayesian": {"k_infer": 2}}


class FiveValidationClips(Dataset):
    def __init__(self, cfg):
        self.case_ids = cfg["experiment"]["split_case_ids"]["validation"]
        self._clip_index = [(cid, 16, 17) for cid in self.case_ids]
    def __len__(self):
        return len(self.case_ids)
    def __getitem__(self, i):
        item = OneClip()[0]
        item["case_id"] = self.case_ids[i]
        return item


def make_loader(cfg):
    return DataLoader(FiveValidationClips(cfg), batch_size=1, collate_fn=collate, shuffle=False)


def test_validation_only_comparison_preserves_weights_rng_and_paired_baseline(tmp_path):
    cfg = saved_config()
    model = build_fake()
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    report = compare_validation(model, make_loader(cfg), cfg, tmp_path, "cpu")
    assert report["complete"] and report["test_evaluated"] is False
    assert report["training_updates"] == 0 and report["paired_initial_predictions_verified"]
    assert report["n_scored_slice_organ_predictions"] == 5
    assert report["selection"]["selected_policy"] in ("none", "guarded")
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())


def test_completed_validation_progress_reuses_predictions_without_decoding(tmp_path, monkeypatch):
    cfg = saved_config(); model = build_fake(); loader = make_loader(cfg)
    first = compare_validation(model, loader, cfg, tmp_path, "cpu")
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed prediction was decoded again")
    monkeypatch.setattr(model, "decode_with_adaptive_rank", forbidden)
    again = compare_validation(model, loader, cfg, tmp_path, "cpu")
    assert again["selection"] == first["selection"]


def test_heldout_case_is_blocked_before_any_prediction(tmp_path):
    cfg = saved_config(); loader = make_loader(cfg)
    loader.dataset.case_ids = cfg["experiment"]["split_case_ids"]["test"]
    with pytest.raises(ValueError, match="Only the saved validation"):
        compare_validation(build_fake(), loader, cfg, tmp_path, "cpu")


def test_partial_validation_cannot_choose_an_inference_policy(tmp_path):
    cfg = saved_config()
    report = compare_validation(build_fake(), make_loader(cfg), cfg, tmp_path, "cpu", deadline=time.time()-1)
    assert report["complete"] is False
    assert report["selection"]["selected_policy"] is None
    assert report["selection"]["complete"] is False


def test_mismatched_paired_base_predictions_fail_instead_of_comparing(tmp_path):
    cfg = saved_config(); model = build_fake(); loader = make_loader(cfg)
    compare_validation(model, loader, cfg, tmp_path, "cpu")
    path = tmp_path/"guarded_progress.json"
    progress = json.loads(path.read_text())
    progress["rows"][0]["base_seg"]["dice"] += 0.1
    path.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="Initial predictions differ"):
        compare_validation(model, loader, cfg, tmp_path, "cpu")


def reference_zip(path, unsafe=False):
    cfg = saved_config()
    checkpoint = b"metadata-only fixture; not a trained model"
    digest = hashlib.sha256(checkpoint).hexdigest()
    manifest = {"protocol_complete": True, "execution_status": "FOLD1_PROTOCOL_COMPLETE", "configuration": cfg}
    usage = {"closed_cleanly": True, "execution_status": "FOLD1_PROTOCOL_COMPLETE", "accounted_runtime_seconds": 49668.25}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("reproducibility_manifest.json", json.dumps(manifest))
        z.writestr("selected_checkpoint.json", json.dumps({"sha256": digest, "epoch": 10}))
        z.writestr("session_state.json", json.dumps(usage))
        z.writestr("research_fold_01/best.pt", checkpoint)
        z.writestr("heldout_evaluation.json", "DO NOT READ THIS FOR VALIDATION SELECTION")
        if unsafe:
            z.writestr("../outside.txt", "bad")
    return digest


def test_rootless_results_zip_is_supported_and_test_scores_are_not_extracted(tmp_path):
    archive = tmp_path/"BALoRA_fold1_20epochs_v1_results.zip"
    digest = reference_zip(archive)
    assert locate_results(tmp_path) == archive
    root = tmp_path/"local_inputs"
    context = prepare_reference(archive, root, digest)
    assert context["checkpoint_sha256"] == digest
    assert context["prior_accounted_seconds"] == 49668.25
    assert not (root/"heldout_evaluation.json").exists()
    assert Path(context["checkpoint"]).is_file()


def test_archive_traversal_is_rejected_before_extraction(tmp_path):
    archive = tmp_path/"bad.zip"
    digest = reference_zip(archive, unsafe=True)
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        prepare_reference(archive, tmp_path/"dest", digest)
    assert not (tmp_path/"outside.txt").exists()


def test_reference_weight_hash_mismatch_stops_before_model_allocation(tmp_path):
    archive = tmp_path/"reference.zip"
    reference_zip(archive)
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_reference(archive, tmp_path/"dest", "0"*64)


def test_saved_split_overlap_is_rejected():
    cfg = saved_config()
    cfg["experiment"]["split_case_ids"]["validation"][0] = "test00"
    with pytest.raises(ValueError, match="overlap"):
        validate_saved_split(cfg)


def test_saved_case_list_hash_is_verified():
    cfg = saved_config()
    cfg["experiment"]["split_case_ids"]["validation"][0] = "changed_case"
    with pytest.raises(ValueError, match="hash"):
        validate_saved_split(cfg)


@pytest.mark.parametrize("clean,expected", [(True, 52000), (False, 49668.25+43200)])
def test_resume_accounts_for_clean_or_interrupted_audit(tmp_path, clean, expected):
    previous = tmp_path/"inputs"/"previous_audit"
    previous.mkdir(parents=True)
    from evaluation.validation_audit import BUILD
    state = {"build": BUILD, "checkpoint_sha256": "abc", "closed_cleanly": clean,
             "prior_accounted_seconds": 49668.25, "accounted_runtime_seconds": 52000}
    (previous/"audit_session_state.json").write_text(json.dumps(state))
    (previous/"legacy_progress.json").write_text('{"rows": []}')
    output = tmp_path/"working"; output.mkdir()
    prior = prepare_audit_resume(tmp_path/"inputs", output,
                                 {"checkpoint_sha256": "abc", "prior_accounted_seconds": 49668.25})
    assert prior == expected
    assert (output/"legacy_progress.json").is_file()


@pytest.mark.parametrize('scenario', ['fresh', 'prior_floor', 'same_session_retry'])
def test_audit_entrypoint_loads_saved_weights_exports_and_closes_usage_ledger(tmp_path, monkeypatch, scenario):
    """Exercise the complete orchestration with CPU fake SAM2, not a GPU claim."""
    import evaluation.validation_audit as audit
    from utils.checkpoint import save_checkpoint
    cfg = saved_config()
    base = tmp_path/"base.pt"; base.write_bytes(b"fake frozen backbone identity")
    cfg["sam2"] = {"checkpoint_sha256": audit.sha256_file(base), "resolved_commit": "fake"}
    cfg["data"] = {}
    model = build_fake()
    checkpoint = tmp_path/"selected.pt"
    save_checkpoint(model, None, 10, str(checkpoint), adapter_only=False)
    reference = {"checkpoint": str(checkpoint), "checkpoint_sha256": audit.sha256_file(checkpoint),
                 "configuration": cfg, "prior_accounted_seconds": 49668.25, "selected_epoch": 10}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU fake orchestration test")
    def build_cpu(config, device):
        result = build_fake(); result._run_config = config
        return result
    monkeypatch.setattr(audit, "build_model", build_cpu)
    monkeypatch.setattr(audit, "make_validation_loader", lambda config, *_: make_loader(config))
    original_compare = audit.compare_validation
    monkeypatch.setattr(audit, "compare_validation", lambda model, loader, config, root, device, deadline:
                        original_compare(model, loader, config, root, "cpu", deadline))
    inputs = tmp_path/"inputs"; inputs.mkdir()
    output = tmp_path/"audit_output"
    start = time.time()-30
    floor = 13.8398 if scenario == 'prior_floor' else 0
    expected_prior = max(49668.25, floor*3600)
    if scenario == 'same_session_retry':
        output.mkdir()
        start = time.time()-1000
        finished = time.time()-2
        expected_prior = 50000
        (output/'audit_session_state.json').write_text(json.dumps({
            'build': audit.BUILD, 'checkpoint_sha256': reference['checkpoint_sha256'],
            'closed_cleanly': True, 'accounted_runtime_seconds':expected_prior,
            'prior_accounted_seconds':49668.25,'finished_unix':finished,
        }))
    result = audit.run_audit(reference, tmp_path, output, base, "fake", inputs,
                             notebook_start=start, minimum_prior_hours=floor)
    state = json.loads((output/"audit_session_state.json").read_text())
    assert result["complete"] and result["training_updates"] == 0
    assert state["closed_cleanly"] and state["execution_status"] == "VALIDATION_AUDIT_COMPLETE"
    assert state['prior_accounted_seconds'] == expected_prior
    if scenario == 'same_session_retry':
        assert state['accounting_start_unix'] == finished
        assert expected_prior+2 <= state['accounted_runtime_seconds'] < expected_prior+60
        assert state['work_deadline_unix'] <= start+3*3600
    else:
        assert state["accounted_runtime_seconds"] >= expected_prior+30
    with zipfile.ZipFile(output/"BALoRA_validation_audit_results.zip") as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        ledger = json.loads(z.read(output.name+"/audit_session_state.json"))
        assert ledger["closed_cleanly"]
        assert not any("heldout" in name for name in z.namelist())
