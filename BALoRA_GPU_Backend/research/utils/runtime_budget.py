"""Cooperative wall-time limits and atomic, strict JSON reports."""
from __future__ import annotations
import json, math, os, time
from pathlib import Path

class BudgetExpired(RuntimeError):
    """An intentional pause with saved state, not successful completion."""

def check_deadline(deadline, stage):
    if deadline is not None and time.time() >= float(deadline):
        raise BudgetExpired(f'{stage}: wall-time allocation reached')

def _clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(k): _clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_clean(v) for v in value]
    return value

def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False, default=str), encoding='utf-8')
    os.replace(tmp, path)

def plan_runtime(train_seconds, validation_seconds, elapsed_seconds,
                 budget_hours=29.5, final_hours=4.0, safety_hours=0.5,
                 requested_epochs=20, min_epochs=8, already_completed=0,
                 *, final_seconds_estimate=None, training_complete=False):
    """Fit training AND final evaluation inside one accounted time budget.

    ``final_hours`` is a minimum reserve, not a separate capacity ceiling.
    Enlarge it to the measured final-stage estimate, rounded UP to the next
    half-hour, and compute the remaining training window from that allocation.
    Requested epochs/data/K are never reduced to make the estimate fit.
    """
    inputs = (train_seconds, validation_seconds, elapsed_seconds, budget_hours,
              final_hours, safety_hours)
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in inputs):
        raise ValueError('Runtime durations must be finite and non-negative')
    if budget_hours <= 0:
        raise ValueError('budget_hours must be positive')
    if requested_epochs < 1 or min_epochs < 1 or min_epochs > requested_epochs or already_completed < 0:
        raise ValueError('Invalid epoch counts for runtime planning')
    if final_seconds_estimate is not None and (
            not math.isfinite(float(final_seconds_estimate)) or final_seconds_estimate < 0):
        raise ValueError('Final evaluation estimate must be finite and non-negative')
    epoch = (float(train_seconds) + float(validation_seconds)) * 1.20
    if not math.isfinite(epoch) or epoch <= 0:
        raise ValueError('Positive finite measured epoch time is required')
    estimate = float(final_seconds_estimate) if final_seconds_estimate is not None else final_hours*3600
    final_reserve = math.ceil(max(final_hours*3600,estimate)/1800)*1800
    remaining = budget_hours*3600 - elapsed_seconds - safety_hours*3600
    available = remaining - final_reserve
    final_fits = remaining >= final_reserve
    feasible = max(0, math.floor((available+1e-6)/epoch))
    needed = 0 if training_complete else max(0, int(requested_epochs) - int(already_completed))
    fits = final_fits and feasible >= needed
    reasons = []
    if not final_fits:
        reasons.append('insufficient_total_time_for_final_evaluation')
    elif feasible < needed:
        reasons.append('insufficient_time_for_requested_training_and_final_evaluation')
    return {'requested_total_epochs': int(requested_epochs),
            'completed_epochs_on_resume': int(already_completed),
            'minimum_total_epochs': int(min_epochs),
            'training_protocol_already_complete':bool(training_complete),
            'required_additional_epochs':needed,
            'estimated_epoch_seconds': epoch, 'available_training_seconds': max(0,available),
            'estimated_feasible_additional_epochs': feasible,
            'requested_protocol_estimated_to_fit': fits,
            'minimum_protocol_estimated_to_fit': final_fits and feasible >= max(0,min_epochs-already_completed),
            'budget_hours': budget_hours, 'minimum_final_reserve_hours': final_hours,
            'estimated_final_seconds': estimate, 'reserved_final_hours': final_reserve/3600,
            'final_evaluation_window_estimated_to_fit': final_fits,
            'training_allocation_hours_from_start': (budget_hours*3600-safety_hours*3600-final_reserve)/3600,
            'estimated_total_accounted_hours': (elapsed_seconds+needed*epoch+final_reserve+safety_hours*3600)/3600,
            'rejection_reasons': reasons,
            'safety_hours': safety_hours,
            'note': 'Estimates only. Deadline pauses are PARTIAL and resumable; no runtime or convergence guarantee.'}
