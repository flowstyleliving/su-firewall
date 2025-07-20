// 🚀 Streamlined Semantic Uncertainty Runtime
// Zero dependencies, deterministic, sub-10ms performance

use std::env;
use clap::{App, Arg};
use semantic_uncertainty_runtime::streamlined_engine::{StreamlinedEngine, StreamlinedResult, FisherAnalysisResult};

#[derive(Debug, Clone)]
pub enum CalculationMethod {
    JsdKl,        // Default: JSD/KL method (practical)
    Fisher,       // Fisher Information method (rigorous)
    Both,         // Both methods (comparison)
}

impl Default for CalculationMethod {
    fn default() -> Self {
        CalculationMethod::JsdKl
    }
}

fn main() {
    let matches = App::new("Semantic Uncertainty Analyzer")
        .version("1.0")
        .about("Analyze semantic uncertainty using precision and flexibility metrics")
        .arg(
            Arg::with_name("method")
                .long("method")
                .value_name("METHOD")
                .help("Calculation method: jsd-kl, fisher, or both")
                .takes_value(true)
                .default_value("jsd-kl")
        )
        .arg(
            Arg::with_name("prompt")
                .short('p')
                .long("prompt")
                .value_name("PROMPT")
                .help("Input prompt text")
                .required(true)
                .takes_value(true)
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT")
                .help("Output text to analyze")
                .required(true)
                .takes_value(true)
        )
        .get_matches();

    let method = match matches.value_of("method").unwrap() {
        "jsd-kl" => CalculationMethod::JsdKl,
        "fisher" => CalculationMethod::Fisher,
        "both" => CalculationMethod::Both,
        _ => {
            eprintln!("Error: Invalid method. Use 'jsd-kl', 'fisher', or 'both'");
            std::process::exit(1);
        }
    };

    let prompt = matches.value_of("prompt").unwrap();
    let output = matches.value_of("output").unwrap();

    let engine = StreamlinedEngine::new();
    
    match method {
        CalculationMethod::JsdKl => {
            println!("🧮 Using JSD/KL method (practical/statistical)");
            let result = engine.analyze(prompt, output);
            print_jsd_kl_results(&result);
        }
        CalculationMethod::Fisher => {
            println!("🔬 Using Fisher Information method (rigorous/mathematical)");
            let result = engine.analyze_with_fisher(prompt, output);
            print_fisher_results(&result);
        }
        CalculationMethod::Both => {
            println!("📊 Comparing both methods");
            let jsd_result = engine.analyze(prompt, output);
            let fisher_result = engine.analyze_with_fisher(prompt, output);
            print_comparison_results(&jsd_result, &fisher_result);
        }
    }
}

fn print_jsd_kl_results(result: &StreamlinedResult) {
    println!("\n📈 JSD/KL Analysis Results:");
    println!("Precision (JSD): {:.4}", result.delta_mu);
    println!("Flexibility (KL): {:.4}", result.delta_sigma);
    println!("Semantic Uncertainty (ℏₛ): {:.4}", result.calibrated_hbar);
    println!("Risk Level: {:?}", result.risk_level);
}

fn print_fisher_results(result: &FisherAnalysisResult) {
    println!("\n🔬 Fisher Information Analysis Results:");
    println!("Precision (Fisher): {:.4}", result.fisher_precision);
    println!("Flexibility (Fisher): {:.4}", result.fisher_flexibility);
    println!("Semantic Uncertainty (ℏₛ_fisher): {:.4}", result.fisher_semantic_uncertainty);
    println!("Risk Level: {:?}", result.risk_level);
}

fn print_comparison_results(
    jsd_result: &StreamlinedResult,
    fisher_result: &FisherAnalysisResult,
) {
    println!("\n📊 Method Comparison:");
    println!("┌─────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Metric          │ JSD/KL      │ Fisher      │ Difference  │");
    println!("├─────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ Precision       │ {:11.4} │ {:11.4} │ {:11.4} │", 
        jsd_result.delta_mu, fisher_result.fisher_precision, 
        (jsd_result.delta_mu - fisher_result.fisher_precision).abs());
    println!("│ Flexibility     │ {:11.4} │ {:11.4} │ {:11.4} │", 
        jsd_result.delta_sigma, fisher_result.fisher_flexibility,
        (jsd_result.delta_sigma - fisher_result.fisher_flexibility).abs());
    println!("│ Semantic ℏₛ     │ {:11.4} │ {:11.4} │ {:11.4} │", 
        jsd_result.calibrated_hbar, fisher_result.fisher_semantic_uncertainty,
        (jsd_result.calibrated_hbar - fisher_result.fisher_semantic_uncertainty).abs());
    println!("└─────────────────┴─────────────┴─────────────┴─────────────┘");
    
    // Confidence interval analysis
    let precision_agreement = calculate_agreement(jsd_result.delta_mu, fisher_result.fisher_precision);
    let flexibility_agreement = calculate_agreement(jsd_result.delta_sigma, fisher_result.fisher_flexibility);
    let uncertainty_agreement = calculate_agreement(jsd_result.calibrated_hbar, fisher_result.fisher_semantic_uncertainty);
    
    println!("\n🎯 Method Agreement Analysis:");
    println!("Precision Agreement: {:.1}%", precision_agreement);
    println!("Flexibility Agreement: {:.1}%", flexibility_agreement);
    println!("Uncertainty Agreement: {:.1}%", uncertainty_agreement);
}

fn calculate_agreement(value1: f64, value2: f64) -> f64 {
    let max_val = value1.max(value2);
    if max_val == 0.0 { 100.0 } else {
        (1.0 - (value1 - value2).abs() / max_val) * 100.0
    }
} 