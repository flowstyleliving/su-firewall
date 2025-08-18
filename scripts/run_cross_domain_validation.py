#!/usr/bin/env python3
"""
Cross-Domain Validation Runner for Semantic Entropy Hallucination Detection

This script provides a comprehensive interface for running cross-domain validation
across medical, legal, and scientific domains with various configuration options.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional

class CrossDomainValidationRunner:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.results_dir = Path("validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_validation_via_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation via REST API"""
        print(f"🔬 Starting cross-domain validation via API...")
        print(f"🎯 Domains: {config['domains']}")
        print(f"📊 Samples per domain: {config.get('samples_per_domain', 1000)}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/cross_domain_validation",
                json=config,
                timeout=1800  # 30 minutes timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            raise
    
    def run_validation_via_binary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation via Rust binary"""
        print(f"🦀 Starting cross-domain validation via Rust binary...")
        
        # Build command arguments
        cmd = [
            "cargo", "run", "-p", "realtime", "--bin", "cross_domain_validation", "--"
        ]
        
        # Add domain arguments
        if config.get("domains"):
            domains_str = ",".join(config["domains"])
            cmd.extend(["--domains", domains_str])
        
        # Add other arguments
        if config.get("samples_per_domain"):
            cmd.extend(["--samples", str(config["samples_per_domain"])])
        
        if config.get("include_baselines", False):
            cmd.append("--baselines")
        
        if config.get("enable_transfer_analysis", False):
            cmd.append("--transfer")
        
        if config.get("enable_parameter_optimization", False):
            cmd.append("--optimize")
        
        # Set output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"cross_domain_validation_{timestamp}.json"
        cmd.extend(["--output", str(output_file)])
        
        print(f"🏃‍♂️ Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                print("✅ Binary execution successful")
                print(result.stdout)
                
                # Load results from file
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    return {"status": "completed", "output_file": str(output_file)}
            else:
                print(f"❌ Binary execution failed with code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)
                
        except subprocess.TimeoutExpired:
            print("⏰ Validation timed out after 30 minutes")
            raise
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            raise
    
    def run_comprehensive_validation(
        self,
        domains: List[str],
        samples_per_domain: int = 1000,
        use_api: bool = True,
        include_baselines: bool = True,
        enable_transfer_analysis: bool = True,
        enable_parameter_optimization: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive cross-domain validation"""
        
        config = {
            "domains": domains,
            "samples_per_domain": samples_per_domain,
            "include_baselines": include_baselines,
            "detailed_analysis": True,
            "enable_transfer_analysis": enable_transfer_analysis,
            "enable_parameter_optimization": enable_parameter_optimization,
        }
        
        print("🧪 CROSS-DOMAIN VALIDATION FOR SEMANTIC ENTROPY")
        print("=" * 55)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        start_time = time.time()
        
        try:
            if use_api:
                # First try to start the server if not running
                if not self.check_server_health():
                    print("🚀 Starting server...")
                    self.start_server()
                    time.sleep(5)  # Give server time to start
                
                results = self.run_validation_via_api(config)
            else:
                results = self.run_validation_via_binary(config)
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ Total validation time: {elapsed_time:.2f} seconds")
            
            if generate_report:
                self.generate_comprehensive_report(results, config)
            
            return results
            
        except Exception as e:
            print(f"💥 Validation failed: {e}")
            raise
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def start_server(self) -> None:
        """Start the realtime server"""
        print("🚀 Starting realtime server...")
        subprocess.Popen(
            ["cargo", "run", "-p", "server"],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def generate_comprehensive_report(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Generate a comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"validation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Cross-Domain Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration summary
            f.write("## Configuration\n\n")
            f.write(f"- **Domains:** {', '.join(config['domains'])}\n")
            f.write(f"- **Samples per domain:** {config.get('samples_per_domain', 1000)}\n")
            f.write(f"- **Baseline comparison:** {config.get('include_baselines', False)}\n")
            f.write(f"- **Transfer analysis:** {config.get('enable_transfer_analysis', False)}\n")
            f.write(f"- **Parameter optimization:** {config.get('enable_parameter_optimization', False)}\n\n")
            
            # Overall performance
            if "overall_summary" in results:
                summary = results["overall_summary"]
                f.write("## Overall Performance\n\n")
                f.write(f"- **Average F1 Score:** {summary.get('avg_cross_domain_f1', 0):.3f}\n")
                f.write(f"- **Average AUROC:** {summary.get('avg_cross_domain_auroc', 0):.3f}\n")
                f.write(f"- **Domains meeting threshold:** {summary.get('domains_meeting_threshold', 0)}/{summary.get('total_domains_tested', 0)}\n")
                f.write(f"- **Best performing domain:** {summary.get('best_performing_domain', 'N/A')}\n")
                f.write(f"- **Most challenging domain:** {summary.get('most_challenging_domain', 'N/A')}\n")
                f.write(f"- **Production ready:** {'✅ Yes' if summary.get('ready_for_production', False) else '⚠️ No'}\n\n")
            
            # Domain-specific results
            if "domain_results" in results:
                f.write("## Domain-Specific Results\n\n")
                for domain, result in results["domain_results"].items():
                    f.write(f"### {domain}\n\n")
                    f.write(f"- **F1 Score:** {result.get('f1_score', 0):.3f}\n")
                    f.write(f"- **AUROC:** {result.get('auroc', 0):.3f}\n")
                    f.write(f"- **Precision:** {result.get('precision', 0):.3f}\n")
                    f.write(f"- **Recall:** {result.get('recall', 0):.3f}\n")
                    f.write(f"- **Threshold met:** {'✅ Yes' if result.get('threshold_met', False) else '⚠️ No'}\n")
                    f.write(f"- **Samples processed:** {result.get('samples_processed', 0)}\n\n")
                    
                    if "key_findings" in result:
                        f.write("**Key Findings:**\n")
                        for finding in result["key_findings"]:
                            f.write(f"- {finding}\n")
                        f.write("\n")
            
            # Recommendations
            if "recommendations" in results:
                f.write("## Recommendations\n\n")
                for recommendation in results["recommendations"]:
                    f.write(f"- {recommendation}\n")
                f.write("\n")
            
            # Transfer analysis
            if "transfer_analysis" in results and results["transfer_analysis"]:
                transfer = results["transfer_analysis"]
                f.write("## Cross-Domain Transfer Analysis\n\n")
                f.write(f"- **Robustness score:** {transfer.get('cross_domain_robustness_score', 0):.3f}\n")
                f.write(f"- **Best source domain:** {transfer.get('best_source_domain', 'N/A')}\n\n")
                
                if "adaptation_requirements" in transfer:
                    f.write("### Domain Adaptation Requirements\n\n")
                    for domain, needed in transfer["adaptation_requirements"].items():
                        status = "Required" if needed else "Not needed"
                        f.write(f"- **{domain}:** {status}\n")
                    f.write("\n")
        
        print(f"📄 Comprehensive report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Domain Validation Runner for Semantic Entropy Hallucination Detection"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["medical", "legal", "scientific", "general"],
        default=["medical", "legal", "scientific"],
        help="Domains to validate"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per domain"
    )
    parser.add_argument(
        "--use-binary",
        action="store_true",
        help="Use Rust binary instead of API"
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline method comparisons"
    )
    parser.add_argument(
        "--no-transfer",
        action="store_true",
        help="Skip cross-domain transfer analysis"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip parameter optimization"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating comprehensive report"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL for API calls"
    )
    
    args = parser.parse_args()
    
    runner = CrossDomainValidationRunner(base_url=args.base_url)
    
    try:
        results = runner.run_comprehensive_validation(
            domains=args.domains,
            samples_per_domain=args.samples,
            use_api=not args.use_binary,
            include_baselines=not args.no_baselines,
            enable_transfer_analysis=not args.no_transfer,
            enable_parameter_optimization=not args.no_optimize,
            generate_report=not args.no_report
        )
        
        print("\n🎉 Cross-domain validation completed successfully!")
        
        # Print key metrics
        if "overall_summary" in results:
            summary = results["overall_summary"]
            print(f"📊 Average F1: {summary.get('avg_cross_domain_f1', 0):.3f}")
            print(f"📈 Average AUROC: {summary.get('avg_cross_domain_auroc', 0):.3f}")
            print(f"✅ Production ready: {summary.get('ready_for_production', False)}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()