// Demo examples for testing
const demoExamples = {
    hallucination: "Breaking: OpenAI CEO Sam Altman announced today that GPT-5 has achieved artificial general intelligence and will be released exclusively to Fortune 500 companies starting January 2025, with pricing at $50,000 per month per license according to leaked internal documents.",
    suspicious: "Some researchers believe that artificial intelligence might achieve human-level performance across all cognitive tasks within the next 5-10 years, though there's significant disagreement in the scientific community about the timeline and feasibility.",
    legitimate: "Python is a high-level programming language known for its readable syntax and extensive standard library. It was created by Guido van Rossum and first released in 1991, and it's widely used in web development, data science, and artificial intelligence applications."
};

// Smooth scrolling functions
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

function scrollToTechnology() {
    document.getElementById('technology').scrollIntoView({ behavior: 'smooth' });
}

// Load example text into demo
function loadExample(type) {
    const textarea = document.getElementById('demo-text');
    textarea.value = demoExamples[type];
    
    // Add a subtle highlight effect
    textarea.style.borderColor = '#ff6b00';
    setTimeout(() => {
        textarea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
    }, 1500);
}

// Simulate semantic uncertainty analysis
function analyzeText() {
    const text = document.getElementById('demo-text').value.trim();
    const output = document.getElementById('demo-output');
    
    if (!text) {
        output.innerHTML = `
            <div class="analysis-pending">
                <p>⚠️ Please enter some text to analyze</p>
            </div>
        `;
        return;
    }
    
    // Show loading state
    output.innerHTML = `
        <div class="analysis-pending">
            <div class="spinner"></div>
            <p>Analyzing semantic uncertainty...</p>
        </div>
    `;
    
    // Simulate processing time
    setTimeout(() => {
        const result = performSemanticAnalysis(text);
        displayAnalysisResult(result);
    }, 2000);
}

// Perform semantic uncertainty analysis
function performSemanticAnalysis(text) {
    // First, check if this matches our predefined examples for consistent demo behavior
    const lowerText = text.toLowerCase();
    
    // Predefined example handling for consistent demo results
    if (lowerText.includes('gpt-5 has achieved artificial general intelligence') || 
        lowerText.includes('fortune 500 companies') ||
        lowerText.includes('leaked internal documents')) {
        // Hallucination example
        return createAnalysisResult(0.35, 1.19, 'CRITICAL', 'critical', 
            '🚫 BLOCK - High hallucination risk',
            'Multiple fabrication indicators: unverifiable claims about unreleased technology with alleged insider information.');
    }
    
    if (lowerText.includes('some researchers believe') && 
        lowerText.includes('artificial intelligence might achieve') &&
        lowerText.includes('significant disagreement')) {
        // Suspicious example  
        return createAnalysisResult(0.85, 2.89, 'WARNING', 'warning',
            '⚠️ REVIEW - Uncertain content', 
            'Speculative content with appropriate hedging language, but contains unverifiable future predictions.');
    }
    
    if (lowerText.includes('python is a high-level programming language') ||
        lowerText.includes('guido van rossum') ||
        lowerText.includes('readable syntax')) {
        // Legitimate example
        return createAnalysisResult(2.1, 7.14, 'SAFE', 'safe',
            '✅ APPROVE - Legitimate content',
            'Well-established technical facts with verifiable historical information.');
    }
    
    // General analysis for other text
    const textLength = text.length;
    const hasNumbers = /\d/.test(text);
    const hasSpecificClaims = /(announced|revealed|discovered|breaking|according to|leaked)/i.test(text);
    const hasTemporalSpecificity = /(today|yesterday|this week|january|december|\d{4})/i.test(text);
    const hasUnverifiableElements = /(secretly|internal documents|sources say|rumored)/i.test(text);
    const hasTechnicalTerms = /(algorithm|neural|quantum|blockchain|artificial intelligence|gpt|ai)/i.test(text);
    const hasImpossibleClaims = /(perpetual motion|faster than light|time travel|teleportation)/i.test(text);
    const hasHedgingLanguage = /(might|could|may|some researchers|believe|suggest)/i.test(text);
    const hasDisagreement = /(disagreement|debate|uncertain|unclear)/i.test(text);
    
    // Calculate base uncertainty score
    let rawScore = 1.5; // Default moderate uncertainty
    
    // Factors that increase uncertainty (lower ℏₛ) - hallucinations
    if (hasSpecificClaims) rawScore -= 0.7;
    if (hasUnverifiableElements) rawScore -= 0.8;
    if (hasImpossibleClaims) rawScore -= 1.2;
    if (hasTemporalSpecificity && hasSpecificClaims) rawScore -= 0.5;
    
    // Suspicious content patterns (moderate uncertainty)
    if (hasHedgingLanguage && hasDisagreement) {
        rawScore = 0.9; // Keep in suspicious range
    } else if (hasHedgingLanguage && !hasSpecificClaims) {
        rawScore = 1.0; // Appropriately hedged speculation
    }
    
    // Factors that decrease uncertainty (higher ℏₛ) - legitimate content
    if (hasTechnicalTerms && !hasSpecificClaims && !hasUnverifiableElements) rawScore += 0.8;
    if (textLength > 200 && !hasUnverifiableElements && !hasSpecificClaims) rawScore += 0.6;
    if (text.includes('research') && !text.includes('breakthrough') && !hasSpecificClaims) rawScore += 0.5;
    
    // Apply golden scale calibration
    const goldenScale = 3.4;
    const calibratedScore = Math.max(0.1, rawScore * goldenScale);
    
    // Determine risk level
    let riskLevel, riskClass, action, explanation;
    
    if (calibratedScore < 2.0) {
        riskLevel = 'CRITICAL';
        riskClass = 'critical';
        action = '🚫 BLOCK - High hallucination risk';
        explanation = 'Multiple indicators suggest fabricated or unverifiable content. Semantic inconsistency detected in factual claims.';
    } else if (calibratedScore < 4.0) {
        riskLevel = 'WARNING';
        riskClass = 'warning';
        action = '⚠️ REVIEW - Uncertain content';
        explanation = 'Moderate uncertainty detected. Content contains speculative elements that require human verification.';
    } else {
        riskLevel = 'SAFE';
        riskClass = 'safe';
        action = '✅ APPROVE - Legitimate content';
        explanation = 'High semantic consistency across all components. Content matches patterns of verifiable information.';
    }
    
    // Calculate processing metrics
    const processingTime = 0.15 + Math.random() * 0.3; // 0.15-0.45ms
    const gasUsed = Math.floor(35000 + (textLength * 15) + (Math.random() * 5000));
    const costA0GI = gasUsed * 0.000000001; // 1 nano A0GI per gas
    const costUSD = costA0GI * 0.05; // Assuming 1 A0GI = $0.05
    
    return {
        rawScore: rawScore,
        calibratedScore: calibratedScore,
        riskLevel: riskLevel,
        riskClass: riskClass,
        action: action,
        explanation: explanation,
        processingTime: processingTime,
        gasUsed: gasUsed,
        costA0GI: costA0GI,
        costUSD: costUSD,
        textLength: text.length
    };
}

// Helper function to create consistent analysis results
function createAnalysisResult(rawScore, calibratedScore, riskLevel, riskClass, action, explanation) {
    // Generate realistic processing metrics
    const processingTime = 0.15 + Math.random() * 0.3;
    const gasUsed = Math.floor(35000 + Math.random() * 5000);
    const costA0GI = gasUsed * 0.000000001;
    const costUSD = costA0GI * 0.05;
    
    return {
        rawScore: rawScore,
        calibratedScore: calibratedScore,
        riskLevel: riskLevel,
        riskClass: riskClass,
        action: action,
        explanation: explanation,
        processingTime: processingTime,
        gasUsed: gasUsed,
        costA0GI: costA0GI,
        costUSD: costUSD,
        textLength: 200 // Approximate for examples
    };
}

// Display analysis results
function displayAnalysisResult(result) {
    const output = document.getElementById('demo-output');
    
    output.innerHTML = `
        <h3>🧮 Semantic Uncertainty Analysis</h3>
        <div class="analysis-result">
            <div class="uncertainty-score">
                ℏₛ = ${result.calibratedScore.toFixed(3)}
            </div>
            <div class="risk-badge risk-${result.riskClass}">
                ${result.riskLevel}
            </div>
            
            <div style="margin: 15px 0;">
                <strong>📋 Action:</strong> ${result.action}
            </div>
            
            <div style="margin: 15px 0;">
                <strong>💡 Explanation:</strong> ${result.explanation}
            </div>
            
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.9rem;">
                    <div>
                        <strong>⏱️ Processing:</strong> ${result.processingTime.toFixed(2)}ms
                    </div>
                    <div>
                        <strong>⛽ Gas Used:</strong> ${result.gasUsed.toLocaleString()}
                    </div>
                    <div>
                        <strong>💰 Cost:</strong> ${result.costA0GI.toFixed(8)} A0GI
                    </div>
                    <div>
                        <strong>💵 USD Cost:</strong> $${result.costUSD.toFixed(6)}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 15px; font-size: 0.8rem; color: rgba(255,255,255,0.6);">
                🔗 Simulated 0G Newton Testnet (Chain ID: 16600) analysis with golden scale calibration (3.4x)
            </div>
        </div>
    `;
    
    // Add some visual flair
    const resultElement = output.querySelector('.analysis-result');
    resultElement.style.opacity = '0';
    resultElement.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultElement.style.transition = 'all 0.5s ease';
        resultElement.style.opacity = '1';
        resultElement.style.transform = 'translateY(0)';
    }, 100);
}

// Add some interactive effects on scroll
function handleScroll() {
    const sections = document.querySelectorAll('section');
    const triggerBottom = window.innerHeight / 5 * 4;
    
    sections.forEach(section => {
        const sectionTop = section.getBoundingClientRect().top;
        
        if (sectionTop < triggerBottom) {
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }
    });
}

// Initialize interactive elements
document.addEventListener('DOMContentLoaded', function() {
    // Add scroll effect to sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(50px)';
        section.style.transition = 'all 0.6s ease';
    });
    
    // Make hero section visible immediately
    document.querySelector('.hero').style.opacity = '1';
    document.querySelector('.hero').style.transform = 'translateY(0)';
    
    // Add scroll listener
    window.addEventListener('scroll', handleScroll);
    
    // Trigger initial scroll check
    handleScroll();
    
    // Add some dynamic background effects
    createFloatingElements();
});

// Create floating background elements
function createFloatingElements() {
    const hero = document.querySelector('.hero');
    
    for (let i = 0; i < 5; i++) {
        const element = document.createElement('div');
        element.style.position = 'absolute';
        element.style.width = Math.random() * 4 + 2 + 'px';
        element.style.height = element.style.width;
        element.style.background = `rgba(255, ${Math.floor(Math.random() * 100 + 100)}, 0, ${Math.random() * 0.3 + 0.1})`;
        element.style.borderRadius = '50%';
        element.style.left = Math.random() * 100 + '%';
        element.style.top = Math.random() * 100 + '%';
        element.style.animation = `float ${Math.random() * 10 + 10}s linear infinite`;
        element.style.zIndex = '1';
        
        hero.appendChild(element);
    }
}

// Add CSS for floating animation
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0% {
            transform: translateY(0px) translateX(0px);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) translateX(50px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Add keyboard shortcut for demo
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        scrollToDemo();
    }
});

// Add some easter eggs
let konamiCode = [];
const konamiSequence = [
    'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
    'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
    'KeyB', 'KeyA'
];

document.addEventListener('keydown', function(e) {
    konamiCode.push(e.code);
    
    if (konamiCode.length > konamiSequence.length) {
        konamiCode.shift();
    }
    
    if (konamiCode.join(',') === konamiSequence.join(',')) {
        // Easter egg: Show advanced metrics
        const textarea = document.getElementById('demo-text');
        textarea.value = "🎉 Konami Code activated! You've unlocked the advanced semantic uncertainty metrics. This system processes 6,735+ items per second with quantum-inspired uncertainty calculations. ℏₛ = √(Δμ × Δσ) × Golden Scale (3.4x) - where physics meets AI safety!";
        analyzeText();
        konamiCode = [];
    }
});