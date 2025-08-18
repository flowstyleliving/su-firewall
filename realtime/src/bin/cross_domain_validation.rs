use std::collections::HashMap;
use clap::{Arg, Command};
use realtime::validation::cross_domain::{CrossDomainValidator, CrossDomainValidationConfig};
use common::DomainType;
use tokio;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
    
    // Parse command line arguments
    let matches = Command::new("cross_domain_validation")
        .about("Cross-Domain Validation for Semantic Entropy Hallucination Detection")
        .arg(
            Arg::new("domains")
                .long("domains")
                .help("Comma-separated list of domains to validate (medical,legal,scientific,general)")
                .value_delimiter(',')
                .default_value("medical,legal,scientific")
                .required(false)
        )
        .arg(
            Arg::new("samples")
                .long("samples")
                .help("Number of samples per domain")
                .default_value("1000")
                .value_parser(clap::value_parser!(usize))
        )
        .arg(
            Arg::new("folds")
                .long("folds")
                .help("Number of cross-validation folds")
                .default_value("5")
                .value_parser(clap::value_parser!(usize))
        )
        .arg(
            Arg::new("baselines")
                .long("baselines")
                .help("Include baseline method comparisons")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("transfer")
                .long("transfer")
                .help("Enable cross-domain transfer analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("optimize")
                .long("optimize")
                .help("Enable parameter optimization")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("output")
                .long("output")
                .help("Output file for validation results")
                .default_value("cross_domain_validation_results.json")
        )
        .get_matches();
    
    // Parse domains
    let domain_strings: Vec<&str> = matches.get_many::<String>("domains")
        .unwrap_or_default()
        .map(|s| s.as_str())
        .collect();
    
    let domains: Result<Vec<DomainType>, String> = domain_strings.iter()
        .map(|&domain_str| match domain_str.to_lowercase().as_str() {
            "medical" => Ok(DomainType::Medical),
            "legal" => Ok(DomainType::Legal),
            "scientific" => Ok(DomainType::Scientific),
            "general" => Ok(DomainType::General),
            _ => Err(format!("Unknown domain: {}", domain_str)),
        })
        .collect();
    
    let domains = match domains {
        Ok(domains) => domains,
        Err(e) => {
            error!("❌ Invalid domain specification: {}", e);
            std::process::exit(1);
        }
    };
    
    let samples_per_domain = *matches.get_one::<usize>("samples").unwrap();
    let validation_splits = *matches.get_one::<usize>("folds").unwrap();
    let include_baselines = matches.get_flag("baselines");
    let enable_transfer_analysis = matches.get_flag("transfer");
    let enable_parameter_optimization = matches.get_flag("optimize");
    let output_file = matches.get_one::<String>("output").unwrap();
    
    info!("🔬 Starting Cross-Domain Validation");
    info!("📊 Domains: {:?}", domains);
    info!("📈 Samples per domain: {}", samples_per_domain);
    info!("🔄 Cross-validation folds: {}", validation_splits);
    info!("🏗️ Include baselines: {}", include_baselines);
    info!("🔄 Transfer analysis: {}", enable_transfer_analysis);
    info!("🎯 Parameter optimization: {}", enable_parameter_optimization);
    
    // Create validation configuration
    let mut performance_thresholds = HashMap::new();
    performance_thresholds.insert(DomainType::Medical, 0.70);
    performance_thresholds.insert(DomainType::Legal, 0.65);
    performance_thresholds.insert(DomainType::Scientific, 0.60);
    performance_thresholds.insert(DomainType::General, 0.55);
    
    let config = CrossDomainValidationConfig {
        domains: domains.clone(),
        samples_per_domain,
        validation_splits,
        baseline_methods: if include_baselines {
            vec![
                "diag_fim_dir".to_string(),
                "scalar_js_kl".to_string(),
                "base_semantic_entropy".to_string(),
            ]
        } else {
            vec![]
        },
        performance_thresholds,
        enable_transfer_analysis,
        enable_parameter_optimization,
        statistical_significance_threshold: 0.05,
    };
    
    // Create validator and run validation
    let mut validator = CrossDomainValidator::new(config);
    
    info!("🚀 Executing cross-domain validation...");
    let start_time = std::time::Instant::now();
    
    match validator.run_cross_domain_validation().await {
        Ok(results) => {
            let elapsed = start_time.elapsed();
            info!("✅ Cross-domain validation completed in {:.2}s", elapsed.as_secs_f64());
            
            // Print summary to console
            print_validation_summary(&results);
            
            // Save detailed results to file
            let results_json = serde_json::to_string_pretty(&results)?;
            tokio::fs::write(output_file, results_json).await?;
            info!("💾 Detailed results saved to: {}", output_file);
            
            // Print recommendations
            print_recommendations(&results);
            
            // Exit with appropriate status code
            if results.overall_performance_summary.avg_cross_domain_f1 > 0.7 {
                info!("🎉 Validation successful - system ready for production!");
                std::process::exit(0);
            } else if results.overall_performance_summary.avg_cross_domain_f1 > 0.6 {
                info!("⚠️ Validation partially successful - improvements recommended");
                std::process::exit(1);
            } else {
                error!("❌ Validation failed - significant improvements needed");
                std::process::exit(2);
            }
        },
        Err(e) => {
            error!("❌ Cross-domain validation failed: {}", e);
            std::process::exit(3);
        }
    }
}

fn print_validation_summary(results: &realtime::validation::cross_domain::CrossDomainResults) {
    println!("\n🔬 CROSS-DOMAIN VALIDATION SUMMARY");
    println!("═══════════════════════════════════════");
    
    let summary = &results.overall_performance_summary;
    println!("📊 Total samples processed: {}", summary.total_samples_processed);
    println!("🎯 Average F1 score: {:.3}", summary.avg_cross_domain_f1);
    println!("📈 Average AUROC: {:.3}", summary.avg_cross_domain_auroc);
    println!("✅ Domains meeting threshold: {}/{}", summary.domains_meeting_threshold, summary.total_domains_tested);
    println!("🏆 Best performing domain: {:?}", summary.best_performing_domain);
    println!("🔧 Most challenging domain: {:?}", summary.most_challenging_domain);
    
    println!("\n📋 DOMAIN-SPECIFIC RESULTS");
    println!("───────────────────────────");
    
    for (domain, result) in &results.domain_results {
        let status = if result.performance_threshold_met { "✅" } else { "⚠️" };
        println!("{} {:?}: F1={:.3}, AUROC={:.3}, Precision={:.3}, Recall={:.3}", 
                status, domain, result.avg_f1, result.avg_auroc, result.avg_precision, result.avg_recall);
        
        // Print domain-specific metrics
        let metrics = &result.domain_specific_metrics;
        println!("  📝 Terminology accuracy: {:.3}", metrics.terminology_accuracy);
        println!("  📚 Citation verification: {:.3}", metrics.citation_verification_rate);
        println!("  🚨 Dangerous misinformation catch rate: {:.3}", metrics.dangerous_misinformation_catch_rate);
        
        match domain {
            DomainType::Medical => {
                if let Some(drug_detection) = metrics.drug_interaction_detection {
                    println!("  💊 Drug interaction detection: {:.3}", drug_detection);
                }
                if let Some(contraindication) = metrics.contraindication_flagging {
                    println!("  ⚠️ Contraindication flagging: {:.3}", contraindication);
                }
            },
            DomainType::Legal => {
                if let Some(precedent) = metrics.precedent_consistency {
                    println!("  ⚖️ Precedent consistency: {:.3}", precedent);
                }
                if let Some(jurisdiction) = metrics.jurisdiction_accuracy {
                    println!("  🏛️ Jurisdiction accuracy: {:.3}", jurisdiction);
                }
            },
            DomainType::Scientific => {
                if let Some(methodology) = metrics.methodology_validation_accuracy {
                    println!("  🔬 Methodology validation: {:.3}", methodology);
                }
                if let Some(statistical) = metrics.statistical_claim_verification {
                    println!("  📊 Statistical claim verification: {:.3}", statistical);
                }
            },
            DomainType::General => {
                println!("  📄 General domain metrics calculated");
            }
        }
        println!();
    }
    
    if let Some(transfer) = &results.transfer_analysis {
        println!("🔄 CROSS-DOMAIN TRANSFER ANALYSIS");
        println!("────────────────────────────────────");
        println!("🌐 Cross-domain robustness score: {:.3}", transfer.cross_domain_robustness_score);
        
        println!("📈 Transfer performance matrix:");
        for ((source, target), score) in &transfer.transfer_matrix {
            println!("  {:?} → {:?}: {:.3}", source, target, score);
        }
        
        println!("🔧 Domain adaptation requirements:");
        for (domain, needed) in &transfer.domain_adaptation_needed {
            let status = if *needed { "⚠️ Required" } else { "✅ Not needed" };
            println!("  {:?}: {}", domain, status);
        }
        println!();
    }
    
    if let Some(universal) = &results.universal_parameters {
        println!("🌍 UNIVERSAL PARAMETERS");
        println!("─────────────────────────");
        println!("λ (lambda): {:.3}", universal.lambda);
        println!("τ (tau): {:.3}", universal.tau);
        println!("Similarity threshold: {:.3}", universal.similarity_threshold);
        println!("Terminology weight: {:.3}", universal.terminology_weight);
        
        println!("🎯 Cross-domain performance with universal parameters:");
        for (domain, performance) in &universal.cross_domain_performance {
            println!("  {:?}: {:.3}", domain, performance);
        }
        println!();
    }
}

fn print_recommendations(results: &realtime::validation::cross_domain::CrossDomainResults) {
    println!("💡 RECOMMENDATIONS");
    println!("═══════════════════");
    
    let summary = &results.overall_performance_summary;
    println!("{}", summary.recommendation);
    
    // Performance-based recommendations
    if summary.avg_cross_domain_f1 > 0.8 {
        println!("🚀 Outstanding performance! Ready for immediate production deployment.");
    } else if summary.avg_cross_domain_f1 > 0.7 {
        println!("👍 Good performance. Consider minor optimizations before production.");
    } else if summary.avg_cross_domain_f1 > 0.6 {
        println!("⚠️ Moderate performance. Domain-specific tuning recommended.");
    } else {
        println!("🔧 Performance below expectations. Significant improvements needed.");
    }
    
    // Domain-specific recommendations
    for (domain, result) in &results.domain_results {
        if !result.performance_threshold_met {
            println!("🎯 {:?} domain improvement suggestions:", domain);
            
            if result.avg_precision < 0.7 {
                println!("  • Reduce false positives by adjusting similarity thresholds");
            }
            if result.avg_recall < 0.7 {
                println!("  • Improve hallucination detection sensitivity");
            }
            if result.domain_specific_metrics.terminology_accuracy < 0.8 {
                println!("  • Enhance domain-specific terminology recognition");
            }
        }
    }
    
    // Transfer learning recommendations
    if let Some(transfer) = &results.transfer_analysis {
        if transfer.cross_domain_robustness_score < 0.7 {
            println!("🔄 Cross-domain transfer recommendations:");
            println!("  • Implement domain-specific adaptation layers");
            println!("  • Consider domain-aware pre-training");
            println!("  • Increase domain-specific vocabulary coverage");
        }
    }
    
    println!("\n🎯 NEXT STEPS");
    println!("─────────────");
    if summary.domains_meeting_threshold == summary.total_domains_tested {
        println!("1. Deploy to staging environment for integration testing");
        println!("2. Configure domain auto-detection in production API");
        println!("3. Set up monitoring for domain-specific performance metrics");
    } else {
        println!("1. Focus improvements on underperforming domains");
        println!("2. Increase domain-specific training data");
        println!("3. Re-run validation after improvements");
    }
}

fn parse_domains(domain_string: &str) -> Result<Vec<DomainType>, String> {
    domain_string
        .split(',')
        .map(|s| match s.trim().to_lowercase().as_str() {
            "medical" => Ok(DomainType::Medical),
            "legal" => Ok(DomainType::Legal),
            "scientific" => Ok(DomainType::Scientific),
            "general" => Ok(DomainType::General),
            invalid => Err(format!("Invalid domain: {}. Valid options: medical, legal, scientific, general", invalid)),
        })
        .collect()
}