// DOM Elements
const textarea = document.getElementById('newsText');
const analyzeBtn = document.getElementById('analyzeBtn');
const charCount = document.getElementById('charCount');
const resultSection = document.getElementById('resultSection');
const loadingOverlay = document.getElementById('loadingOverlay');

// Accordion elements
const howHeader = document.getElementById('howHeader');
const howBody = document.getElementById('howBody');
const howArrow = document.getElementById('howArrow');

// Accordion state
let isHowOpen = false;

// Character counter
textarea.addEventListener('input', function() {
    charCount.textContent = this.value.length;
});

// Accordion click
howHeader.addEventListener('click', function() {
    isHowOpen = !isHowOpen;
    howBody.style.maxHeight = isHowOpen ? howBody.scrollHeight + 'px' : '0';
    howBody.style.padding = isHowOpen ? '0 0 0.5rem 0' : '0';
    howArrow.style.transform = isHowOpen ? 'rotate(180deg)' : 'rotate(0deg)';
});

// Initialize accordion (closed by default)
howBody.style.maxHeight = '0';
howBody.style.padding = '0';

// Analyze function
async function analyzeNews(text) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) throw new Error('Server error');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Update UI with results
function displayResults(data) {
    const resultCard = document.querySelector('.result-card');
    const resultIcon = document.getElementById('resultIcon');
    const resultLabel = document.getElementById('resultLabel');
    const resultVerdict = document.getElementById('resultVerdict');
    const scoreBar = document.getElementById('scoreBar');
    const scoreValue = document.getElementById('scoreValue');
    const resultExplanation = document.getElementById('resultExplanation');
    
    if (!data) {
        resultVerdict.textContent = '⚠️ Error';
        resultExplanation.textContent = 'Could not connect to server. Make sure the backend is running.';
        scoreBar.style.width = '0%';
        scoreValue.textContent = '0%';
        resultSection.style.display = 'block';
        return;
    }
    
    const isReal = data.is_real;
    const score = data.authenticity_score;
    
    // Update card class for colors
    resultCard.classList.remove('real', 'fake');
    resultCard.classList.add(isReal ? 'real' : 'fake');
    
    // Update content
    resultIcon.textContent = isReal ? '📰' : '⚠️';
    resultLabel.textContent = isReal ? 'Likely Real Pattern' : 'Suspicious Pattern';
    resultVerdict.textContent = data.verdict;
    scoreBar.style.width = score + '%';
    scoreValue.textContent = Math.round(score) + '%';
    
    // Add explanation
    if (isReal) {
        resultExplanation.textContent = 'Writing pattern matches factual, neutral reporting style. Low sensationalism detected.';
    } else {
        resultExplanation.textContent = 'Writing pattern shows signs of sensationalism: excessive punctuation, emotional language, or ALL CAPS.';
    }
    
    // Show result section with animation
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Main analyze handler
async function handleAnalyze() {
    const text = textarea.value.trim();
    
    if (!text) {
        alert('Please paste some text to analyze');
        return;
    }
    
    // Show loading
    loadingOverlay.classList.add('active');
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.7';
    
    // Hide previous result
    resultSection.style.display = 'none';
    
    // Simulate minimum loading time for UX
    const result = await analyzeNews(text);
    
    setTimeout(() => {
        displayResults(result);
        loadingOverlay.classList.remove('active');
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
    }, 400);
}

// Example buttons
function setExample(type) {
    const examples = {
        fake: "BREAKING!!! The government just announced a free smartphone for every citizen!!! This is 100% confirmed by a secret source!!! Share this news before they delete it!!! Don't let them hide the truth!!!",
        real: "The Ministry of Electronics and Information Technology announced today a new initiative to improve digital literacy in rural areas. The program will partner with state governments and aims to train 10 million citizens over the next two years."
    };
    textarea.value = examples[type];
    charCount.textContent = textarea.value.length;
    
    // Auto-analyze after a short delay
    setTimeout(() => handleAnalyze(), 100);
}

// Event listeners
analyzeBtn.addEventListener('click', handleAnalyze);
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const type = btn.getAttribute('data-type');
        setExample(type);
    });
});

// Allow Enter+Ctrl to submit
textarea.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        handleAnalyze();
    }
});