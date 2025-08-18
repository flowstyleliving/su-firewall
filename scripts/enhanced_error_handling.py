#!/usr/bin/env python3
"""
Enhanced Error Handling and Logging
===================================

Implements comprehensive error handling, detailed logging, and robust
failure recovery mechanisms for the hallucination detection system.
"""

import json
import requests
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import time
from pathlib import Path
from functools import wraps
import sys
from contextlib import contextmanager

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hallucination_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    timestamp: str
    function_name: str
    error_type: str
    error_message: str
    stack_trace: str
    input_data: Dict[str, Any]
    recovery_attempted: bool
    recovery_successful: bool
    impact_level: str  # "low", "medium", "high", "critical"

@dataclass
class OperationResult:
    """Standardized operation result with error tracking"""
    success: bool
    data: Any = None
    error_context: Optional[ErrorContext] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = None

class ErrorHandlingDecorator:
    """Decorator for comprehensive error handling"""
    
    def __init__(self, impact_level: str = "medium", retry_count: int = 2):
        self.impact_level = impact_level
        self.retry_count = retry_count
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> OperationResult:
            start_time = time.time()
            
            for attempt in range(self.retry_count + 1):
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000
                    
                    logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1} ({execution_time:.1f}ms)")
                    
                    return OperationResult(
                        success=True,
                        data=result,
                        execution_time_ms=execution_time,
                        metadata={"attempt": attempt + 1}
                    )
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Create error context
                    error_context = ErrorContext(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        function_name=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                        input_data={
                            "args": str(args)[:500] + "..." if len(str(args)) > 500 else str(args),
                            "kwargs": str(kwargs)[:500] + "..." if len(str(kwargs)) > 500 else str(kwargs)
                        },
                        recovery_attempted=attempt < self.retry_count,
                        recovery_successful=False,
                        impact_level=self.impact_level
                    )
                    
                    if attempt < self.retry_count:
                        wait_time = (attempt + 1) * 0.5  # Exponential backoff
                        logger.warning(f"⚠️ {func.__name__} failed on attempt {attempt + 1}: {e}")
                        logger.info(f"🔄 Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final failure
                        logger.error(f"❌ {func.__name__} failed after {self.retry_count + 1} attempts: {e}")
                        log_error_details(error_context)
                        
                        return OperationResult(
                            success=False,
                            error_context=error_context,
                            execution_time_ms=execution_time
                        )
            
        return wrapper

def log_error_details(error_context: ErrorContext):
    """Log comprehensive error details"""
    
    logger.error(f"🚨 ERROR DETAILS:")
    logger.error(f"   Function: {error_context.function_name}")
    logger.error(f"   Type: {error_context.error_type}")
    logger.error(f"   Message: {error_context.error_message}")
    logger.error(f"   Impact: {error_context.impact_level}")
    logger.error(f"   Time: {error_context.timestamp}")
    
    # Save error to file for analysis
    error_log_path = Path("error_analysis.jsonl")
    with open(error_log_path, "a") as f:
        f.write(json.dumps(asdict(error_context)) + "\n")

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    logger.info(f"🚀 Starting {operation_name}")
    
    try:
        yield
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"✅ {operation_name} completed in {execution_time:.1f}ms")
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"❌ {operation_name} failed after {execution_time:.1f}ms: {e}")
        raise

@ErrorHandlingDecorator(impact_level="high", retry_count=3)
def robust_api_call(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = 5,
    api_base: str = "http://localhost:8080/api/v1"
) -> Dict[str, Any]:
    """Robust API call with comprehensive error handling"""
    
    # Input validation
    if not endpoint or not payload:
        raise ValueError("Endpoint and payload are required")
    
    # Sanitize payload
    sanitized_payload = sanitize_api_payload(payload)
    
    # Make request
    response = requests.post(
        f"{api_base}/{endpoint}",
        json=sanitized_payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"}
    )
    
    # Validate response
    if response.status_code != 200:
        raise requests.HTTPError(f"API error {response.status_code}: {response.text}")
    
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    
    # Validate response structure
    validate_api_response(result, endpoint)
    
    return result

def sanitize_api_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize API payload to prevent errors"""
    
    sanitized = {}
    
    for key, value in payload.items():
        if key in ["prompt", "output"]:
            # Ensure strings and limit length
            if isinstance(value, str):
                sanitized[key] = value[:10000]  # Prevent oversized inputs
            else:
                sanitized[key] = str(value)[:10000]
                
        elif key in ["model_id"]:
            # Ensure valid model ID
            sanitized[key] = str(value) if value else "mistral-7b"
            
        elif key in ["ensemble", "intelligent_routing"]:
            # Ensure boolean
            sanitized[key] = bool(value) if value is not None else True
            
        elif key in ["lambda_override", "tau_override"]:
            # Ensure numeric and within bounds
            if value is not None:
                try:
                    num_val = float(value)
                    if 0.01 <= num_val <= 10.0:
                        sanitized[key] = num_val
                except (ValueError, TypeError):
                    pass  # Skip invalid values
                    
        else:
            # Pass through other values
            sanitized[key] = value
    
    return sanitized

def validate_api_response(response: Dict[str, Any], endpoint: str):
    """Validate API response structure"""
    
    if endpoint == "analyze":
        # Check required fields for analyze endpoint
        if "ensemble_result" not in response:
            logger.warning("⚠️ Missing ensemble_result in API response")
        
        ensemble = response.get("ensemble_result", {})
        if "hbar_s" not in ensemble or "p_fail" not in ensemble:
            logger.warning("⚠️ Missing semantic uncertainty metrics in response")

@ErrorHandlingDecorator(impact_level="medium", retry_count=1)
def robust_dataset_loading(
    dataset_name: str,
    max_samples: Optional[int] = None
) -> List[Any]:
    """Robust dataset loading with fallback mechanisms"""
    
    logger.info(f"📊 Loading dataset: {dataset_name}")
    
    if dataset_name == "truthfulqa":
        from comprehensive_dataset_loader import load_truthfulqa_fixed
        return load_truthfulqa_fixed(max_samples)
        
    elif dataset_name.startswith("halueval_"):
        task = dataset_name.replace("halueval_", "")
        from comprehensive_dataset_loader import load_halueval_fixed
        return load_halueval_fixed(task, max_samples)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def analyze_error_patterns(error_log_path: str = "error_analysis.jsonl") -> Dict[str, Any]:
    """Analyze error patterns from logged errors"""
    
    if not Path(error_log_path).exists():
        return {"message": "No error log found"}
    
    errors = []
    with open(error_log_path, 'r') as f:
        for line in f:
            if line.strip():
                errors.append(json.loads(line))
    
    if not errors:
        return {"message": "No errors in log"}
    
    # Analyze patterns
    error_analysis = {
        "total_errors": len(errors),
        "by_function": {},
        "by_error_type": {},
        "by_impact_level": {},
        "recent_errors": errors[-10:],  # Last 10 errors
        "most_common_errors": {}
    }
    
    # Group by various categories
    for error in errors:
        func_name = error.get("function_name", "unknown")
        error_type = error.get("error_type", "unknown")
        impact = error.get("impact_level", "unknown")
        
        # Count by function
        error_analysis["by_function"][func_name] = error_analysis["by_function"].get(func_name, 0) + 1
        
        # Count by error type
        error_analysis["by_error_type"][error_type] = error_analysis["by_error_type"].get(error_type, 0) + 1
        
        # Count by impact
        error_analysis["by_impact_level"][impact] = error_analysis["by_impact_level"].get(impact, 0) + 1
    
    logger.info(f"🔍 Error analysis complete: {len(errors)} errors analyzed")
    
    return error_analysis

def create_health_check_endpoint() -> Dict[str, Any]:
    """Create comprehensive health check for the system"""
    
    health_status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "unknown",
        "components": {}
    }
    
    # Test API connectivity
    try:
        response = requests.get("http://localhost:8080/health", timeout=3)
        if response.status_code == 200:
            health_status["components"]["api"] = {
                "status": "healthy",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
        else:
            health_status["components"]["api"] = {
                "status": "degraded",
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        health_status["components"]["api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Test dataset availability
    dataset_status = {}
    for dataset in ["truthfulqa", "halueval_qa", "halueval_general"]:
        try:
            result = robust_dataset_loading(dataset, max_samples=1)
            if result.success and result.data:
                dataset_status[dataset] = {"status": "available", "sample_count": len(result.data)}
            else:
                dataset_status[dataset] = {"status": "error", "error": str(result.error_context.error_message if result.error_context else "Unknown")}
        except Exception as e:
            dataset_status[dataset] = {"status": "error", "error": str(e)}
    
    health_status["components"]["datasets"] = dataset_status
    
    # Test ensemble functionality
    try:
        test_response = robust_api_call(
            "analyze",
            {
                "prompt": "Test prompt",
                "output": "Test output",
                "model_id": "mistral-7b",
                "ensemble": True
            }
        )
        
        if test_response.success:
            health_status["components"]["ensemble"] = {
                "status": "healthy",
                "response_time_ms": test_response.execution_time_ms
            }
        else:
            health_status["components"]["ensemble"] = {
                "status": "unhealthy",
                "error": test_response.error_context.error_message if test_response.error_context else "Unknown"
            }
            
    except Exception as e:
        health_status["components"]["ensemble"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Determine overall status
    component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
    if isinstance(health_status["components"]["datasets"], dict):
        dataset_statuses = [ds.get("status", "unknown") for ds in health_status["components"]["datasets"].values()]
        component_statuses.extend(dataset_statuses)
    
    if all(status == "healthy" or status == "available" for status in component_statuses):
        health_status["overall_status"] = "healthy"
    elif any(status == "unhealthy" or status == "error" for status in component_statuses):
        health_status["overall_status"] = "unhealthy"
    else:
        health_status["overall_status"] = "degraded"
    
    return health_status

def enhanced_evaluation_with_error_handling(
    max_samples: int = 100,
    enable_detailed_logging: bool = True
) -> Dict[str, Any]:
    """Run evaluation with comprehensive error handling"""
    
    if enable_detailed_logging:
        logger.info("🔧 Enhanced error handling enabled")
    
    evaluation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "error_summary": {},
        "performance_metrics": {},
        "health_check": {}
    }
    
    with performance_monitor("Enhanced Evaluation"):
        # Health check first
        health_status = create_health_check_endpoint()
        evaluation_results["health_check"] = health_status
        
        if health_status["overall_status"] != "healthy":
            logger.warning(f"⚠️ System health check failed: {health_status['overall_status']}")
        
        # Load datasets with error handling
        dataset_result = robust_dataset_loading("truthfulqa", max_samples // 2)
        if not dataset_result.success:
            logger.error("❌ Failed to load datasets")
            evaluation_results["error_summary"]["dataset_loading"] = "Failed"
            return evaluation_results
        
        truthfulqa_pairs = dataset_result.data
        
        halueval_result = robust_dataset_loading("halueval_qa", max_samples // 2)
        halueval_pairs = halueval_result.data if halueval_result.success else []
        
        all_pairs = truthfulqa_pairs + halueval_pairs
        evaluation_results["total_samples"] = len(all_pairs)
        
        logger.info(f"📊 Loaded {len(all_pairs)} evaluation pairs")
        
        # Run analysis with error tracking
        successful_count = 0
        failed_count = 0
        error_types = {}
        
        for i, pair in enumerate(all_pairs[:max_samples]):
            # Test both correct and hallucinated answers
            test_cases = [
                (pair.correct_answer, False),
                (pair.hallucinated_answer, True)
            ]
            
            for answer, is_hallucination in test_cases:
                # Robust API call
                api_result = robust_api_call(
                    "analyze",
                    {
                        "prompt": pair.prompt,
                        "output": answer,
                        "model_id": "mistral-7b",
                        "ensemble": True,
                        "intelligent_routing": True
                    }
                )
                
                if api_result.success:
                    successful_count += 1
                    logger.debug(f"✅ Analysis {i+1} successful")
                else:
                    failed_count += 1
                    error_type = api_result.error_context.error_type if api_result.error_context else "Unknown"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    logger.warning(f"⚠️ Analysis {i+1} failed: {error_type}")
            
            # Progress reporting
            if (i + 1) % 10 == 0:
                progress = ((i + 1) / min(max_samples, len(all_pairs))) * 100
                logger.info(f"📈 Progress: {i + 1}/{min(max_samples, len(all_pairs))} ({progress:.1f}%)")
        
        # Update results
        evaluation_results["successful_analyses"] = successful_count
        evaluation_results["failed_analyses"] = failed_count
        evaluation_results["error_summary"] = error_types
        
        # Calculate reliability metrics
        total_analyses = successful_count + failed_count
        reliability_rate = successful_count / total_analyses if total_analyses > 0 else 0
        
        evaluation_results["performance_metrics"] = {
            "reliability_rate": reliability_rate,
            "total_analyses": total_analyses,
            "error_rate": failed_count / total_analyses if total_analyses > 0 else 0
        }
        
        logger.info(f"📊 Evaluation complete: {reliability_rate:.3f} reliability rate")
    
    return evaluation_results

def generate_error_report() -> str:
    """Generate comprehensive error analysis report"""
    
    print("📋 GENERATING ERROR ANALYSIS REPORT")
    print("=" * 50)
    
    # Analyze error patterns
    error_analysis = analyze_error_patterns()
    
    # Generate report
    report_lines = [
        "# Error Analysis Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Error Summary",
        f"- Total Errors: {error_analysis.get('total_errors', 0)}",
        ""
    ]
    
    # Function error breakdown
    if "by_function" in error_analysis:
        report_lines.extend([
            "## Errors by Function",
            ""
        ])
        for func, count in sorted(error_analysis["by_function"].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- {func}: {count} errors")
        report_lines.append("")
    
    # Error type breakdown
    if "by_error_type" in error_analysis:
        report_lines.extend([
            "## Errors by Type",
            ""
        ])
        for error_type, count in sorted(error_analysis["by_error_type"].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- {error_type}: {count} occurrences")
        report_lines.append("")
    
    # Impact level breakdown
    if "by_impact_level" in error_analysis:
        report_lines.extend([
            "## Errors by Impact Level",
            ""
        ])
        for impact, count in error_analysis["by_impact_level"].items():
            report_lines.append(f"- {impact}: {count} errors")
        report_lines.append("")
    
    # Recommendations
    report_lines.extend([
        "## Recommendations",
        ""
    ])
    
    total_errors = error_analysis.get('total_errors', 0)
    if total_errors > 50:
        report_lines.append("- High error count detected - investigate system stability")
    if error_analysis.get("by_error_type", {}).get("requests.ConnectionError", 0) > 10:
        report_lines.append("- Connection errors detected - check API server health")
    if error_analysis.get("by_error_type", {}).get("TimeoutError", 0) > 5:
        report_lines.append("- Timeout errors detected - consider increasing timeout values")
    
    if not report_lines[-1]:  # No recommendations added
        report_lines.append("- System appears stable")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open("error_analysis_report.md", "w") as f:
        f.write(report_text)
    
    print("💾 Error report saved: error_analysis_report.md")
    print(f"📊 Total errors analyzed: {total_errors}")
    
    return report_text

if __name__ == "__main__":
    print("🚀 Starting Enhanced Error Handling System...")
    
    # Run enhanced evaluation
    results = enhanced_evaluation_with_error_handling(max_samples=50)
    
    print(f"\n📊 ENHANCED EVALUATION RESULTS")
    print("-" * 50)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Successful: {results['successful_analyses']}")
    print(f"Failed: {results['failed_analyses']}")
    print(f"Reliability: {results['performance_metrics'].get('reliability_rate', 0):.3f}")
    print(f"System Health: {results['health_check'].get('overall_status', 'unknown')}")
    
    # Generate error report
    print(f"\n" + "=" * 60)
    generate_error_report()
    
    print(f"\n✅ Enhanced error handling evaluation complete!")