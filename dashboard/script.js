/* ═══════════════════════════════════════════════════════════════
   EquiChurn Dashboard — Interactive JavaScript
   Handles: scroll reveals, counters, radar chart, fairness chart
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initScrollReveal();
    initCounters();
    initRadarChart();
    initFairnessChart();
});

/* ─── NAVBAR SCROLL EFFECT ─────────────────────────────────── */
function initNavbar() {
    const navbar = document.getElementById('navbar');
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        // Shrink on scroll
        if (window.scrollY > 60) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active link
        let current = '';
        sections.forEach(section => {
            const top = section.offsetTop - 200;
            if (window.scrollY >= top) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });
}

/* ─── SCROLL REVEAL ────────────────────────────────────────── */
function initScrollReveal() {
    const elements = document.querySelectorAll('[data-reveal]');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger siblings
                const siblings = entry.target.parentElement.querySelectorAll('[data-reveal]');
                let delay = 0;
                siblings.forEach((sib, i) => {
                    if (sib === entry.target) delay = i * 100;
                });

                setTimeout(() => {
                    entry.target.classList.add('revealed');
                }, delay);

                // Trigger metric bar fills
                const fill = entry.target.querySelector('.metric-fill');
                if (fill) {
                    setTimeout(() => fill.classList.add('animate'), delay + 200);
                }

                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    });

    elements.forEach(el => observer.observe(el));
}

/* ─── ANIMATED COUNTERS ────────────────────────────────────── */
function initCounters() {
    const counters = document.querySelectorAll('.metric-value');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseFloat(entry.target.dataset.count);
                const counterEl = entry.target.querySelector('.counter');
                animateCounter(counterEl, 0, target, 1800);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(c => observer.observe(c));
}

function animateCounter(element, start, end, duration) {
    const isDecimal = end < 10;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;

        if (isDecimal) {
            element.textContent = current.toFixed(2);
        } else {
            element.textContent = Math.round(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/* ─── RADAR CHART (Hero Visual) ────────────────────────────── */
function initRadarChart() {
    const canvas = document.getElementById('radarCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 400 * dpr;
    canvas.height = 400 * dpr;
    ctx.scale(dpr, dpr);

    const cx = 200, cy = 200, maxR = 150;
    const labels = ['Accuracy', 'Recall', 'Precision', 'Fairness', 'Speed', 'Calibration'];
    const values = [0.91, 0.88, 0.85, 0.95, 0.92, 0.89];
    const n = labels.length;

    let animProgress = 0;
    const animDuration = 2000;
    let startTime = null;

    function drawFrame(timestamp) {
        if (!startTime) startTime = timestamp;
        animProgress = Math.min((timestamp - startTime) / animDuration, 1);
        const eased = 1 - Math.pow(1 - animProgress, 3);

        ctx.clearRect(0, 0, 400, 400);

        // Draw grid rings
        for (let ring = 1; ring <= 5; ring++) {
            const r = (maxR / 5) * ring;
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Draw axes
        for (let i = 0; i < n; i++) {
            const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
            const x = cx + maxR * Math.cos(angle);
            const y = cy + maxR * Math.sin(angle);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x, y);
            ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Labels
            const labelR = maxR + 22;
            const lx = cx + labelR * Math.cos(angle);
            const ly = cy + labelR * Math.sin(angle);
            ctx.fillStyle = 'rgba(240,240,245,0.5)';
            ctx.font = '500 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(labels[i], lx, ly);
        }

        // Draw data polygon
        const gradient = ctx.createLinearGradient(50, 50, 350, 350);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
        gradient.addColorStop(1, 'rgba(6, 182, 212, 0.15)');

        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const idx = i % n;
            const angle = (Math.PI * 2 / n) * idx - Math.PI / 2;
            const r = maxR * values[idx] * eased;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw data points
        for (let i = 0; i < n; i++) {
            const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
            const r = maxR * values[i] * eased;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);

            // Glow
            const glowGrad = ctx.createRadialGradient(x, y, 0, x, y, 12);
            glowGrad.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
            glowGrad.addColorStop(1, 'rgba(99, 102, 241, 0)');
            ctx.fillStyle = glowGrad;
            ctx.fillRect(x - 12, y - 12, 24, 24);

            // Dot
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#6366f1';
            ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Center pulse
        const pulseR = 6 + 4 * Math.sin(timestamp / 1000);
        const centerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulseR * 3);
        centerGrad.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
        centerGrad.addColorStop(1, 'rgba(99, 102, 241, 0)');
        ctx.fillStyle = centerGrad;
        ctx.fillRect(cx - pulseR * 3, cy - pulseR * 3, pulseR * 6, pulseR * 6);

        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#6366f1';
        ctx.fill();

        requestAnimationFrame(drawFrame);
    }

    // Start after a delay to let the page load
    setTimeout(() => requestAnimationFrame(drawFrame), 500);
}

/* ─── FAIRNESS CHART ───────────────────────────────────────── */
function initFairnessChart() {
    const canvas = document.getElementById('fairnessCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 480 * dpr;
    canvas.height = 300 * dpr;
    ctx.scale(dpr, dpr);

    const W = 480, H = 300;
    const padding = { top: 40, right: 30, bottom: 50, left: 60 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;

    // Data: Before vs After mitigation
    const groups = ['Age < 30', 'Age 30-50', 'Age 50+', 'NA Region', 'EMEA', 'APAC'];
    const before = [0.22, 0.19, 0.31, 0.18, 0.21, 0.28];
    const after = [0.20, 0.19, 0.21, 0.19, 0.20, 0.20];
    const maxVal = 0.40;

    let animProgress = 0;
    let startTime = null;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                requestAnimationFrame(drawFrame);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    observer.observe(canvas);

    function drawFrame(timestamp) {
        if (!startTime) startTime = timestamp;
        animProgress = Math.min((timestamp - startTime) / 1500, 1);
        const eased = 1 - Math.pow(1 - animProgress, 3);

        ctx.clearRect(0, 0, W, H);

        // Title
        ctx.fillStyle = 'rgba(240,240,245,0.9)';
        ctx.font = '600 13px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Churn Prediction Rate by Demographic', padding.left, 24);

        ctx.fillStyle = 'rgba(240,240,245,0.4)';
        ctx.font = '400 10px Inter, sans-serif';
        ctx.fillText('Before vs. After Fairness Mitigation', padding.left, 38);

        // Grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + chartH - (chartH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + chartW, y);
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Y-axis labels
            const val = (maxVal / 4) * i;
            ctx.fillStyle = 'rgba(240,240,245,0.3)';
            ctx.font = '500 10px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.fillText((val * 100).toFixed(0) + '%', padding.left - 8, y + 3);
        }

        // Fairness threshold line
        const threshY = padding.top + chartH - (0.10 / maxVal) * chartH;

        // Bars
        const barGroupW = chartW / groups.length;
        const barW = barGroupW * 0.28;
        const gap = 4;

        for (let i = 0; i < groups.length; i++) {
            const groupX = padding.left + barGroupW * i + barGroupW / 2;

            // Before bar
            const bH = (before[i] / maxVal) * chartH * eased;
            const bX = groupX - barW - gap / 2;
            const bY = padding.top + chartH - bH;

            const bGrad = ctx.createLinearGradient(0, bY, 0, bY + bH);
            bGrad.addColorStop(0, 'rgba(244, 63, 94, 0.8)');
            bGrad.addColorStop(1, 'rgba(244, 63, 94, 0.3)');
            ctx.fillStyle = bGrad;
            roundedRect(ctx, bX, bY, barW, bH, 3);
            ctx.fill();

            // After bar
            const aH = (after[i] / maxVal) * chartH * eased;
            const aX = groupX + gap / 2;
            const aY = padding.top + chartH - aH;

            const aGrad = ctx.createLinearGradient(0, aY, 0, aY + aH);
            aGrad.addColorStop(0, 'rgba(16, 185, 129, 0.9)');
            aGrad.addColorStop(1, 'rgba(16, 185, 129, 0.4)');
            ctx.fillStyle = aGrad;
            roundedRect(ctx, aX, aY, barW, aH, 3);
            ctx.fill();

            // X-axis label
            ctx.fillStyle = 'rgba(240,240,245,0.4)';
            ctx.font = '500 9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(groups[i], groupX, padding.top + chartH + 18);
        }

        // Fairness band
        const bandTop = padding.top + chartH - (0.22 / maxVal) * chartH;
        const bandBot = padding.top + chartH - (0.18 / maxVal) * chartH;
        ctx.fillStyle = 'rgba(99, 102, 241, 0.06)';
        ctx.fillRect(padding.left, bandTop, chartW, bandBot - bandTop);

        // Parity line at ~0.20
        const parityY = padding.top + chartH - (0.20 / maxVal) * chartH;
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(padding.left, parityY);
        ctx.lineTo(padding.left + chartW, parityY);
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = 'rgba(99, 102, 241, 0.7)';
        ctx.font = '500 9px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Target Parity', padding.left + chartW + 4, parityY + 3);

        // Legend
        const legX = W - 140;
        const legY = 18;

        ctx.fillStyle = 'rgba(244, 63, 94, 0.7)';
        roundedRect(ctx, legX, legY, 10, 10, 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(240,240,245,0.5)';
        ctx.font = '500 10px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Before', legX + 16, legY + 9);

        ctx.fillStyle = 'rgba(16, 185, 129, 0.8)';
        roundedRect(ctx, legX + 70, legY, 10, 10, 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(240,240,245,0.5)';
        ctx.fillText('After', legX + 86, legY + 9);

        if (animProgress < 1) {
            requestAnimationFrame(drawFrame);
        }
    }
}

function roundedRect(ctx, x, y, w, h, r) {
    if (h < r * 2) r = h / 2;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}
