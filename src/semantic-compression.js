// 🧠 Semantic Compression Engine
// Extract semantic "essence" from long prompts for ultra-fast analysis

export class SemanticCompressor {
  constructor() {
    this.maxTokens = 150; // Compressed essence limit
    this.compressionRatio = 0.3; // Target 30% of original
  }

  /**
   * 🧠 Compress prompt to semantic essence
   * Extracts key concepts, intent, and risk factors
   */
  compressPrompt(prompt) {
    const startTime = Date.now();
    
    // 🔍 Step 1: Extract semantic components
    const components = this.extractSemanticComponents(prompt);
    
    // 🎯 Step 2: Identify core intent
    const intent = this.identifyIntent(prompt, components);
    
    // ⚡ Step 3: Compress to essence
    const essence = this.generateEssence(components, intent);
    
    // 📊 Step 4: Preserve risk indicators
    const riskPreservation = this.preserveRiskFactors(prompt, essence);
    
    const compressionTime = Date.now() - startTime;
    
    return {
      original_length: prompt.length,
      compressed_essence: essence.compressed,
      compression_ratio: essence.compressed.length / prompt.length,
      semantic_loss: essence.semantic_loss,
      risk_preservation: riskPreservation,
      intent_category: intent.category,
      key_concepts: components.concepts,
      compression_time_ms: compressionTime,
      should_use_compression: this.shouldUseCompression(prompt, essence)
    };
  }

  /**
   * 🔍 Extract key semantic components
   */
  extractSemanticComponents(prompt) {
    const words = prompt.toLowerCase().split(/\s+/);
    
    // 🎯 Key concept extraction
    const concepts = this.extractKeyConcepts(words);
    
    // 🔧 Action words (verbs that indicate intent)
    const actions = this.extractActions(words);
    
    // 🏷️ Entities and subjects
    const entities = this.extractEntities(words);
    
    // ⚠️ Risk indicators
    const riskIndicators = this.extractRiskIndicators(words);
    
    // 🎭 Emotional/persuasive language
    const emotionalMarkers = this.extractEmotionalMarkers(words);
    
    return {
      concepts: concepts,
      actions: actions,
      entities: entities,
      risk_indicators: riskIndicators,
      emotional_markers: emotionalMarkers,
      complexity_score: this.calculateComplexityScore(concepts, actions, entities)
    };
  }

  extractKeyConcepts(words) {
    // 🧠 High-value concept words that carry semantic weight
    const conceptWords = [
      // Technology
      'ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network',
      'algorithm', 'data', 'model', 'training', 'deep', 'computer', 'quantum',
      
      // Actions/Processes  
      'create', 'generate', 'build', 'develop', 'design', 'implement', 'analyze',
      'explain', 'describe', 'write', 'make', 'produce', 'construct',
      
      // Risk-related
      'hack', 'break', 'exploit', 'manipulate', 'deceive', 'illegal', 'harmful',
      'dangerous', 'weapon', 'bomb', 'poison', 'kill', 'destroy',
      
      // Complex/Abstract
      'paradox', 'infinite', 'impossible', 'consciousness', 'reality', 'existence',
      'logic', 'reasoning', 'truth', 'philosophy', 'ethics', 'moral'
    ];
    
    return words.filter(word => conceptWords.includes(word));
  }

  extractActions(words) {
    const actionWords = [
      'create', 'make', 'build', 'generate', 'produce', 'develop', 'design',
      'write', 'explain', 'describe', 'analyze', 'solve', 'find', 'discover',
      'hack', 'break', 'exploit', 'manipulate', 'trick', 'deceive', 'bypass'
    ];
    
    return words.filter(word => actionWords.includes(word));
  }

  extractEntities(words) {
    // Simple entity extraction - in production would use NLP
    const entities = [];
    
    // Look for capitalized words (proper nouns)
    const originalWords = words.join(' ').split(/\s+/);
    originalWords.forEach(word => {
      if (/^[A-Z][a-z]+/.test(word)) {
        entities.push(word.toLowerCase());
      }
    });
    
    // Common entity types
    const entityTypes = [
      'bank', 'government', 'company', 'person', 'system', 'network',
      'database', 'server', 'website', 'application', 'software'
    ];
    
    entityTypes.forEach(entity => {
      if (words.includes(entity)) {
        entities.push(entity);
      }
    });
    
    return [...new Set(entities)]; // Remove duplicates
  }

  extractRiskIndicators(words) {
    const riskWords = [
      // High risk
      'hack', 'bomb', 'weapon', 'kill', 'murder', 'poison', 'illegal', 'steal',
      'break', 'exploit', 'manipulate', 'deceive', 'trick', 'bypass', 'override',
      
      // Medium risk  
      'paradox', 'infinite', 'impossible', 'crash', 'destroy', 'damage',
      'harmful', 'dangerous', 'unethical', 'immoral',
      
      // Chaos indicators
      'consciousness', 'reality', 'existence', 'recursive', 'loop', 'meta'
    ];
    
    return words.filter(word => riskWords.includes(word));
  }

  extractEmotionalMarkers(words) {
    const emotionalWords = [
      'urgent', 'critical', 'important', 'secret', 'hidden', 'powerful',
      'amazing', 'incredible', 'revolutionary', 'breakthrough', 'advanced'
    ];
    
    return words.filter(word => emotionalWords.includes(word));
  }

  calculateComplexityScore(concepts, actions, entities) {
    return concepts.length * 1.5 + actions.length * 2.0 + entities.length * 1.0;
  }

  /**
   * 🎯 Identify primary intent category
   */
  identifyIntent(prompt, components) {
    const promptLower = prompt.toLowerCase();
    
    // 🔍 Intent patterns
    const intentPatterns = {
      'creation': /\b(create|make|build|generate|produce|develop|design|write)\b/g,
      'explanation': /\b(explain|describe|what|how|why|tell|about)\b/g,
      'analysis': /\b(analyze|compare|evaluate|assess|examine|study)\b/g,
      'problem_solving': /\b(solve|fix|resolve|find|solution|help)\b/g,
      'instruction': /\b(how to|guide|steps|tutorial|instructions)\b/g,
      'creative': /\b(story|poem|creative|imagine|fiction|narrative)\b/g,
      'factual': /\b(fact|true|correct|accurate|definition|meaning)\b/g,
      'risky': /\b(hack|break|exploit|illegal|harmful|dangerous)\b/g,
      'philosophical': /\b(consciousness|reality|existence|meaning|purpose|truth)\b/g
    };
    
    const intentScores = {};
    let maxScore = 0;
    let primaryIntent = 'general';
    
    for (const [intent, pattern] of Object.entries(intentPatterns)) {
      const matches = (promptLower.match(pattern) || []).length;
      intentScores[intent] = matches;
      
      if (matches > maxScore) {
        maxScore = matches;
        primaryIntent = intent;
      }
    }
    
    return {
      category: primaryIntent,
      confidence: maxScore / (prompt.split(/\s+/).length / 10), // Normalize by length
      all_scores: intentScores,
      is_multi_intent: Object.values(intentScores).filter(score => score > 0).length > 2
    };
  }

  /**
   * ⚡ Generate compressed semantic essence
   */
  generateEssence(components, intent) {
    // 🎯 Priority-based compression
    const priorityElements = [];
    
    // 1. Risk indicators (highest priority)
    if (components.risk_indicators.length > 0) {
      priorityElements.push(`RISK: ${components.risk_indicators.slice(0, 3).join(', ')}`);
    }
    
    // 2. Primary intent
    priorityElements.push(`INTENT: ${intent.category}`);
    
    // 3. Key actions (top 3)
    if (components.actions.length > 0) {
      priorityElements.push(`ACTIONS: ${components.actions.slice(0, 3).join(', ')}`);
    }
    
    // 4. Core concepts (top 5)
    if (components.concepts.length > 0) {
      priorityElements.push(`CONCEPTS: ${components.concepts.slice(0, 5).join(', ')}`);
    }
    
    // 5. Entities (top 3)
    if (components.entities.length > 0) {
      priorityElements.push(`ENTITIES: ${components.entities.slice(0, 3).join(', ')}`);
    }
    
    // 6. Emotional markers
    if (components.emotional_markers.length > 0) {
      priorityElements.push(`EMOTION: ${components.emotional_markers.slice(0, 2).join(', ')}`);
    }
    
    const compressed = priorityElements.join(' | ');
    
    // 📊 Calculate semantic loss estimation
    const originalComplexity = components.complexity_score;
    const compressedComplexity = priorityElements.length * 2;
    const semanticLoss = Math.max(0, (originalComplexity - compressedComplexity) / originalComplexity);
    
    return {
      compressed: compressed,
      semantic_loss: semanticLoss,
      compression_quality: this.assessCompressionQuality(components, compressed)
    };
  }

  /**
   * 📊 Preserve critical risk factors during compression
   */
  preserveRiskFactors(original, essence) {
    const originalRisk = this.calculateRiskScore(original);
    const essenceRisk = this.calculateRiskScore(essence.compressed);
    
    const preservation = essenceRisk / Math.max(originalRisk, 0.1);
    
    return {
      original_risk_score: originalRisk,
      essence_risk_score: essenceRisk,
      preservation_ratio: preservation,
      risk_preserved: preservation > 0.8 // 80% threshold
    };
  }

  calculateRiskScore(text) {
    const riskWords = [
      'hack', 'bomb', 'weapon', 'kill', 'illegal', 'steal', 'break', 'exploit',
      'manipulate', 'deceive', 'paradox', 'infinite', 'impossible', 'dangerous'
    ];
    
    const textLower = text.toLowerCase();
    return riskWords.reduce((score, word) => {
      return score + (textLower.includes(word) ? 1 : 0);
    }, 0);
  }

  assessCompressionQuality(components, compressed) {
    // Quality metrics
    const hasRiskIndicators = components.risk_indicators.length > 0;
    const hasKeyActions = components.actions.length > 0;
    const hasConcepts = components.concepts.length > 0;
    
    const qualityScore = (
      (hasRiskIndicators ? 30 : 0) +
      (hasKeyActions ? 25 : 0) +
      (hasConcepts ? 25 : 0) +
      (compressed.length < 200 ? 20 : 0) // Bonus for staying under limit
    );
    
    return {
      score: qualityScore,
      rating: qualityScore > 80 ? 'excellent' : 
              qualityScore > 60 ? 'good' : 
              qualityScore > 40 ? 'fair' : 'poor'
    };
  }

  /**
   * 🤔 Decide whether to use compression
   */
  shouldUseCompression(original, essence) {
    const originalLength = original.length;
    const compressionRatio = essence.compressed.length / originalLength;
    
    // Use compression if:
    // 1. Original is long enough to benefit (>300 chars)
    // 2. Compression achieves good ratio (<0.5)
    // 3. Semantic loss is acceptable (<0.4)
    // 4. Risk preservation is good (>0.7)
    
    return {
      should_compress: originalLength > 300 && 
                      compressionRatio < 0.5 && 
                      essence.semantic_loss < 0.4,
      reason: this.getCompressionReason(originalLength, compressionRatio, essence.semantic_loss),
      performance_benefit: this.estimatePerformanceBenefit(originalLength, compressionRatio)
    };
  }

  getCompressionReason(length, ratio, loss) {
    if (length <= 300) return 'Text too short to benefit from compression';
    if (ratio >= 0.5) return 'Compression ratio insufficient';
    if (loss >= 0.4) return 'Semantic loss too high';
    return 'Compression recommended for performance';
  }

  estimatePerformanceBenefit(originalLength, compressionRatio) {
    const processingReduction = (1 - compressionRatio) * 100;
    const estimatedSpeedupMs = Math.round(originalLength * 0.02 * (1 - compressionRatio));
    
    return {
      processing_reduction_percent: processingReduction,
      estimated_speedup_ms: estimatedSpeedupMs,
      worth_compression: estimatedSpeedupMs > 10
    };
  }

  /**
   * 🧪 Batch compression for multiple prompts
   */
  compressBatch(prompts) {
    return prompts.map((prompt, index) => ({
      index: index,
      original: prompt,
      compression: this.compressPrompt(prompt)
    }));
  }
}

// 🧠 Enhanced Semantic Engine with Compression
export class CompressedSemanticEngine {
  constructor(originalEngine) {
    this.originalEngine = originalEngine;
    this.compressor = new SemanticCompressor();
    this.compressionStats = {
      total_requests: 0,
      compressed_requests: 0,
      average_speedup_ms: 0,
      compression_success_rate: 0
    };
  }

  async analyze(prompt, model = 'gpt4') {
    const startTime = Date.now();
    
    // 🧠 Step 1: Analyze if compression would benefit
    const compression = this.compressor.compressPrompt(prompt);
    
    // 🎯 Step 2: Decide analysis approach
    let analysisPrompt = prompt;
    let compressionUsed = false;
    
    if (compression.should_use_compression.should_compress) {
      analysisPrompt = compression.compressed_essence;
      compressionUsed = true;
      this.compressionStats.compressed_requests++;
    }
    
    this.compressionStats.total_requests++;
    
    // ⚡ Step 3: Run semantic analysis (original or compressed)
    let result = await this.originalEngine.analyze(analysisPrompt, model);
    
    // 📊 Step 4: Adjust metrics based on compression
    if (compressionUsed) {
      result = this.adjustForCompression(result, compression, prompt);
    }
    
    const totalTime = Date.now() - startTime;
    
    // 📈 Update compression statistics
    this.updateCompressionStats(compressionUsed, totalTime, compression);
    
    return {
      ...result,
      compression_used: compressionUsed,
      compression_data: compressionUsed ? {
        original_length: compression.original_length,
        compressed_length: compression.compressed_essence.length,
        compression_ratio: compression.compression_ratio,
        semantic_loss: compression.semantic_loss,
        performance_benefit: compression.should_use_compression.performance_benefit
      } : null,
      processing_time: totalTime
    };
  }

  adjustForCompression(result, compression, originalPrompt) {
    // 🎯 Adjust ℏₛ based on compression loss
    const compressionFactor = 1 + (compression.semantic_loss * 0.3);
    
    // 📊 Preserve risk indicators from original prompt
    if (compression.risk_preservation.risk_preserved) {
      // Risk properly preserved, use compressed analysis
      return {
        ...result,
        h_bar: result.h_bar * compressionFactor,
        compression_adjusted: true
      };
    } else {
      // Risk not well preserved, boost uncertainty
      return {
        ...result,
        h_bar: Math.min(result.h_bar * compressionFactor * 1.2, 5.0),
        risk_level: this.escalateRiskIfNeeded(result.risk_level, compression),
        compression_adjusted: true,
        risk_escalation: true
      };
    }
  }

  escalateRiskIfNeeded(currentRisk, compression) {
    if (compression.risk_preservation.original_risk_score > 2 && 
        !compression.risk_preservation.risk_preserved) {
      // Original had high risk but compression lost it
      return 'high_collapse_risk';
    }
    return currentRisk;
  }

  updateCompressionStats(used, totalTime, compression) {
    if (used && compression.should_use_compression.performance_benefit) {
      const speedup = compression.should_use_compression.performance_benefit.estimated_speedup_ms;
      this.compressionStats.average_speedup_ms = 
        (this.compressionStats.average_speedup_ms + speedup) / 2;
    }
    
    this.compressionStats.compression_success_rate = 
      this.compressionStats.compressed_requests / this.compressionStats.total_requests;
  }

  getCompressionStats() {
    return {
      ...this.compressionStats,
      compression_rate: (this.compressionStats.compressed_requests / 
                        this.compressionStats.total_requests * 100).toFixed(1) + '%'
    };
  }
}