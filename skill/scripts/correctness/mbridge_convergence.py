# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored verbatim from NVIDIA Megatron-Bridge, scripts/performance/utils/evaluate.py, commit
# c3c974d4f2e315db8db01945acab6555049c40d7 (2026-10-05): the licence header above (lines 1-13), the numpy import
# (lines 26-31) and validate_convergence (lines 145-329), unmodified. Nothing else of that file is used. ATTUNE's
# correctness check calls it, for review only, on a check arm's per-step loss against its baseline arm's, with the
# thresholds Megatron-Bridge's pipeline passes (scripts/performance/argument_parser.py:1092-1125).
import logging
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    HAVE_NUMPY = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMPY = False


def validate_convergence(
    current_values: "np.ndarray",
    golden_values: "np.ndarray",
    steps: List[str],
    logger: logging.Logger,
    wandb_run: Optional["wandb.Run"] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Comprehensive loss curve convergence validation using multiple metrics.

    This function implements a robust multi-metric approach to validate that
    the current training run produces statistically equivalent results to the
    golden reference, accounting for training variability and different loss ranges.

    Args:
        current_values: Current training loss values
        golden_values: Golden reference loss values
        steps: Training step identifiers
        logger: Logger instance for detailed reporting
        config: Optional configuration dict with custom thresholds

    Returns:
        Dict with 'passed' boolean and detailed results
    """

    # Default configuration
    default_config = {
        # Statistical significance threshold
        "correlation_threshold": 0.95,
        # Point-wise tolerances (adaptive based on loss magnitude)
        "high_loss_tolerance": 0.10,  # 10% for loss > 2.0
        "medium_loss_tolerance": 0.05,  # 5% for loss 0.5-2.0
        "low_loss_tolerance": 0.02,  # 2% for loss < 0.5
        # Curve shape metrics
        "final_loss_tolerance": 0.03,  # 3% for final loss
        # Outlier handling
        "max_outlier_ratio": 0.1,  # Max 10% of points can be outliers
        "outlier_threshold": 3.0,  # 3-sigma outlier detection
        # Loss curve analysis
        "skip_first_percent_loss": 0.0,  # Percentage of loss points to skip from beginning
    }

    if config:
        default_config.update(config)
    config = default_config

    results = {"passed": True, "failed_metrics": [], "summary": "", "details": "", "metrics": {}}

    logger.info("Starting comprehensive loss curve validation...")

    # 1. SKIP FIRST PERCENT OF LOSS POINTS (if configured)
    skip_first_n_percent = max(0, int(len(current_values) * config["skip_first_percent_loss"]))
    if skip_first_n_percent > 0:
        current_values = current_values[skip_first_n_percent:]
        golden_values = golden_values[skip_first_n_percent:]
        steps = steps[skip_first_n_percent:]
        logger.info(f"Skipped first {skip_first_n_percent} loss points for analysis")

    # 2. STATISTICAL CORRELATION TEST
    correlation = np.corrcoef(current_values, golden_values)[0, 1]
    results["metrics"]["correlation"] = correlation

    if correlation < config["correlation_threshold"]:
        results["passed"] = False
        results["failed_metrics"].append("correlation")
        logger.warning(f"Correlation {correlation:.4f} < threshold {config['correlation_threshold']}")
    else:
        logger.info(f"✓ Correlation test passed: {correlation:.4f} >= {config['correlation_threshold']:.4f}")

    # 3. ADAPTIVE POINT-WISE TOLERANCE CHECK
    point_wise_failures = []
    for i, (current_val, golden_val) in enumerate(zip(current_values, golden_values)):
        # Determine tolerance based on loss magnitude
        if golden_val > 2.0:
            tolerance = config["high_loss_tolerance"]
        elif golden_val > 0.5:
            tolerance = config["medium_loss_tolerance"]
        else:
            tolerance = config["low_loss_tolerance"]

        # Calculate relative difference
        if golden_val != 0:
            relative_diff = abs(current_val - golden_val) / abs(golden_val)
        else:
            relative_diff = abs(current_val) if current_val != 0 else 0

        if relative_diff > tolerance:
            point_wise_failures.append(
                {
                    "step": steps[i],
                    "current": current_val,
                    "golden": golden_val,
                    "relative_diff": relative_diff,
                    "tolerance": tolerance,
                }
            )

    results["metrics"]["point_wise_failures"] = len(point_wise_failures)
    results["metrics"]["total_points"] = len(current_values)

    if len(point_wise_failures) > 0:
        failure_ratio = len(point_wise_failures) / len(current_values)
        if failure_ratio > config["max_outlier_ratio"]:
            results["passed"] = False
            results["failed_metrics"].append("point_wise_tolerance")
            logger.warning(
                f"Point-wise failures: {len(point_wise_failures)}/{len(current_values)} "
                f"({failure_ratio:.2%}) > max allowed {config['max_outlier_ratio']:.2%}"
            )
        else:
            logger.info(f"✓ Point-wise tolerance: {len(point_wise_failures)} outliers within acceptable range")
    else:
        logger.info("✓ Point-wise tolerance: All points within tolerance")

    # 4. FINAL LOSS VALIDATION
    final_current = current_values[-1]
    final_golden = golden_values[-1]
    final_diff = abs(final_current - final_golden) / final_golden if final_golden != 0 else abs(final_current)

    results["metrics"]["final_loss_current"] = final_current
    results["metrics"]["final_loss_golden"] = final_golden
    results["metrics"]["final_loss_diff"] = final_diff

    if final_diff > config["final_loss_tolerance"]:
        results["passed"] = False
        results["failed_metrics"].append("final_loss")
        logger.warning(f"Final loss difference {final_diff:.4f} > threshold {config['final_loss_tolerance']}")
    else:
        logger.info(f"✓ Final loss validation passed: {final_diff:.4f} <= {config['final_loss_tolerance']:.4f}")

    # 5. OUTLIER DETECTION (3-sigma rule)
    residuals = current_values - golden_values
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    outliers = np.abs(residuals - mean_residual) > config["outlier_threshold"] * std_residual
    outlier_count = np.sum(outliers)

    results["metrics"]["outlier_count"] = outlier_count
    results["metrics"]["outlier_ratio"] = outlier_count / len(current_values)

    if outlier_count / len(current_values) > config["max_outlier_ratio"]:
        results["passed"] = False
        results["failed_metrics"].append("outliers")
        logger.warning(
            f"Too many outliers: {outlier_count}/{len(current_values)} "
            f"({outlier_count / len(current_values):.2%}) > max {config['max_outlier_ratio']:.2%}"
        )
    else:
        logger.info(f"✓ Outlier detection passed: {outlier_count} outliers <= {config['max_outlier_ratio']:.2%}")

    # Generate summary
    if results["passed"]:
        results["summary"] = "All convergence tests passed"
        logger.info("🎉 All convergence validation tests PASSED!")
    else:
        results["summary"] = f"Failed {len(results['failed_metrics'])} out of 5 validation tests"
        logger.error(f"❌ Convergence validation FAILED: {results['summary']}")

        # Add detailed failure information
        details = []
        if point_wise_failures:
            details.append(f"Point-wise failures ({len(point_wise_failures)}):")
            for failure in point_wise_failures[:5]:  # Show first 5 failures
                details.append(
                    f"  Step {failure['step']}: {failure['current']:.6f} vs {failure['golden']:.6f} "
                    f"(diff: {failure['relative_diff']:.4f})"
                )
            if len(point_wise_failures) > 5:
                details.append(f"  ... and {len(point_wise_failures) - 5} more")

        results["details"] = "\n".join(details)

    if wandb_run is not None:
        wandb_run.summary["convergence_passed"] = results["passed"]
        wandb_run.summary["convergence_failed_metrics"] = ",".join(results["failed_metrics"])

    for key, value in results["metrics"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    return results

